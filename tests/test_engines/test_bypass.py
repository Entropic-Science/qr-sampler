"""Tests for the per-request sampling bypass (``qr_bypass``).

Pins AGENTS.md invariant 8's one explicit exception: a request carrying
``qr_bypass=true`` passes through ``apply()`` untouched — zero entropy
drawn, no ``TokenSamplingRecord``, no perf telemetry — so vLLM's native
sampler applies the standard request params. Bare requests never bypass.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from qr_sampler.engines.vllm.adapter import VLLMAdapter, _BypassState, _RequestState
from qr_sampler.exceptions import ConfigValidationError
from tests.test_engines.test_vllm_adapter import (
    MockAddedRequest,
    MockBatchUpdate,
    MockSamplingParams,
    _make_adapter,
)


def _random_logits(rows: int, vocab: int = 10, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 2.0, size=(rows, vocab)).astype(np.float32)


def _assert_onehot(row: np.ndarray) -> None:
    assert np.count_nonzero(row == 0.0) == 1
    assert np.all(np.isneginf(row[row != 0.0]))


class TestBypassUpdateState:
    """Bypass requests build a _BypassState and skip all pipeline work."""

    def test_bypass_builds_bypass_state(self) -> None:
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(extra_args={"qr_bypass": True})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            assert isinstance(adapter._request_states[0], _BypassState)
        finally:
            adapter.close()

    def test_bypass_string_true_coerces(self) -> None:
        """The env-settable string form ("true") coerces to a bool."""
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(extra_args={"qr_bypass": "true"})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            assert isinstance(adapter._request_states[0], _BypassState)
        finally:
            adapter.close()

    def test_bypass_fires_no_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bypass addition never fires the first-token entropy prefetch."""
        adapter = _make_adapter()
        try:
            calls: list[Any] = []
            for pipeline in adapter._pipelines.values():
                monkeypatch.setattr(
                    pipeline.entropy_source,
                    "prefetch",
                    lambda *args, **kwargs: calls.append(args),
                )
            params = MockSamplingParams(extra_args={"qr_bypass": True})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            assert calls == []
            # Positive control: a non-bypass addition fires the prefetch
            # through the same spy (entropy_prefetch defaults True).
            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=1, sampling_params=MockSamplingParams())]
                )
            )
            assert len(calls) == 1
        finally:
            adapter.close()

    def test_bypass_plus_unpreinitialised_source_does_not_raise(self) -> None:
        """Bypass + anything is bypass: the short-circuit runs BEFORE the
        pipeline lookup, so an un-preinit'd source name in the same
        extra_args cannot raise in the engine worker (GL-01: an uncaught
        raise there kills the whole shared engine)."""
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(
                extra_args={"qr_bypass": True, "qr_entropy_source_type": "openentropy"}
            )
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            assert isinstance(adapter._request_states[0], _BypassState)
        finally:
            adapter.close()

    def test_bare_request_never_bypasses_under_preset(self) -> None:
        """A bare request on a QR_PRESET deployment routes to the full QR
        pipeline — presets never switch bypass on."""
        adapter = _make_adapter(preset="creative_sampling")
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
                )
            )
            state = adapter._request_states[0]
            assert isinstance(state, _RequestState)
            assert state.config.temperature_strategy == "hvh_drift"
            assert state.config.bypass is False
        finally:
            adapter.close()

    def test_routed_event_reports_native_bypass(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(extra_args={"qr_bypass": True})
            with caplog.at_level(logging.INFO, logger="qr_sampler"):
                adapter.update_state(
                    MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
                )
            routed = [
                r for r in caplog.records if getattr(r, "event", "") == "entropy.request.routed"
            ]
            assert len(routed) == 1
            assert routed[0].resolved_pipeline_source == "native"
            assert routed[0].bypass is True
        finally:
            adapter.close()

    def test_bypass_state_survives_tuple_abi_move_swap_remove(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_BypassState duck-types the removal-loop attributes, so vLLM V1
        tuple-ABI moves/swaps/removals need no special-casing and the
        completion event reports source=native."""

        class _Dir:
            def __init__(self, name: str) -> None:
                self.name = name

        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        (0, MockSamplingParams(extra_args={"qr_bypass": True}), None, []),
                        (1, MockSamplingParams(), None, []),
                    ]
                )
            )
            adapter.update_state(MockBatchUpdate(moved=[(0, 5, _Dir("UNIDIRECTIONAL"))]))
            assert 0 not in adapter._request_states
            assert isinstance(adapter._request_states[5], _BypassState)

            adapter.update_state(MockBatchUpdate(moved=[(5, 1, _Dir("SWAP"))]))
            assert isinstance(adapter._request_states[1], _BypassState)
            assert isinstance(adapter._request_states[5], _RequestState)

            with caplog.at_level(logging.INFO, logger="qr_sampler"):
                adapter.update_state(MockBatchUpdate(removed=[1]))
            assert 1 not in adapter._request_states
            completed = [
                r for r in caplog.records if getattr(r, "event", "") == "entropy.request.completed"
            ]
            assert len(completed) == 1
            assert completed[0].dominant_source == "native"
        finally:
            adapter.close()


class TestBypassApplyNumpy:
    """apply() partitioning on numpy batches."""

    def test_all_bypass_rows_bit_identical(self) -> None:
        """An all-bypass step returns the tensor untouched (and in-place)."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=i,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        )
                        for i in range(2)
                    ]
                )
            )
            logits = _random_logits(rows=2)
            before = logits.copy()
            out = adapter.apply(logits)
            assert out is logits
            assert np.array_equal(logits, before)
        finally:
            adapter.close()

    def test_mixed_batch_partitions_rows(self) -> None:
        """Bypass row untouched; QR row AND stateless default row one-hot."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        ),
                        MockAddedRequest(req_index=1, sampling_params=MockSamplingParams()),
                        # Row 2 gets NO state (ABI-wobble shape): samples via
                        # the default pipeline because the default is not bypass.
                    ]
                )
            )
            logits = _random_logits(rows=3)
            before = logits.copy()
            adapter.apply(logits)
            assert np.array_equal(logits[0], before[0])
            _assert_onehot(logits[1])
            _assert_onehot(logits[2])
        finally:
            adapter.close()

    def test_1d_bypass_passthrough(self) -> None:
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(extra_args={"qr_bypass": True})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
            before = logits.copy()
            out = adapter.apply(logits)
            assert out is logits
            assert np.array_equal(logits, before)
        finally:
            adapter.close()

    def test_bypass_draws_zero_entropy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bypass rows never touch an entropy source: no per-token fetch,
        no prefetch, on ANY pre-initialised pipeline."""
        adapter = _make_adapter()
        try:

            def _boom(*args: Any, **kwargs: Any) -> bytes:
                raise AssertionError("bypass request drew entropy")

            for pipeline in adapter._pipelines.values():
                monkeypatch.setattr(pipeline.entropy_source, "get_random_bytes", _boom)
                monkeypatch.setattr(pipeline.entropy_source, "prefetch", _boom)
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=i,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        )
                        for i in range(2)
                    ]
                )
            )
            logits = _random_logits(rows=2)
            before = logits.copy()
            adapter.apply(logits)
            adapter.apply(logits)
            assert np.array_equal(logits, before)
        finally:
            adapter.close()

    def test_tokens_generated_counts_steps_and_completion_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bypass rows still count steps, so the entropy.request.completed
        event honestly reports tokens=N source=native (K-5 diagnostic)."""
        adapter = _make_adapter()
        try:
            params = MockSamplingParams(extra_args={"qr_bypass": True})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            logits = _random_logits(rows=1)
            adapter.apply(logits)
            adapter.apply(logits)
            state = adapter._request_states[0]
            assert isinstance(state, _BypassState)
            assert state.tokens_generated == 2
            with caplog.at_level(logging.INFO, logger="qr_sampler"):
                adapter.update_state(MockBatchUpdate(removed=[0]))
            completed = [
                r for r in caplog.records if getattr(r, "event", "") == "entropy.request.completed"
            ]
            assert len(completed) == 1
            assert completed[0].tokens_generated == 2
            assert completed[0].dominant_source == "native"
        finally:
            adapter.close()

    def test_perf_telemetry_attributed_to_sampled_rows_only(self) -> None:
        """Bypass rows feed no perf telemetry, and the batch-wide to_numpy /
        onehot costs are divided by the SAMPLED row count (attribution fix:
        the historical divisor was the whole batch size)."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        ),
                        MockAddedRequest(req_index=1, sampling_params=MockSamplingParams()),
                    ]
                )
            )
            logits = _random_logits(rows=3)
            adapter.apply(logits)
            assert adapter._perf.tokens_total == 2
            assert len(adapter._perf._stages["to_numpy"]) == 2
        finally:
            adapter.close()

    def test_diagnostic_records_only_for_sampled_rows(self) -> None:
        adapter = _make_adapter(diagnostic_mode="true")
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        ),
                        MockAddedRequest(req_index=1, sampling_params=MockSamplingParams()),
                    ]
                )
            )
            logits = _random_logits(rows=3)
            adapter.apply(logits)
            records = adapter.sampling_logger.get_diagnostic_data()
            assert len(records) == 2
        finally:
            adapter.close()

    def test_env_process_wide_bypass(self) -> None:
        """QR_BYPASS=true turns the server into a vanilla vLLM passthrough:
        bare requests AND stateless (ABI-wobble) rows bypass, while
        qr_bypass=false opts back into the QR pipeline per-request."""
        adapter = _make_adapter(bypass="true")
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(req_index=0, sampling_params=MockSamplingParams()),
                        MockAddedRequest(
                            req_index=1,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": False}),
                        ),
                    ]
                )
            )
            assert isinstance(adapter._request_states[0], _BypassState)
            assert isinstance(adapter._request_states[1], _RequestState)
            logits = _random_logits(rows=3)
            before = logits.copy()
            adapter.apply(logits)
            assert np.array_equal(logits[0], before[0])  # bare request bypasses
            assert np.array_equal(logits[2], before[2])  # stateless row bypasses
            _assert_onehot(logits[1])  # explicit opt-back-in samples
        finally:
            adapter.close()


class TestBypassApplyTorch:
    """apply() partitioning on torch CPU tensors."""

    def test_all_bypass_torch_tensor_untouched(self) -> None:
        torch = pytest.importorskip("torch")
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=i,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        )
                        for i in range(2)
                    ]
                )
            )
            gen = torch.Generator().manual_seed(11)
            logits = torch.randn((2, 10), generator=gen, dtype=torch.float32)
            before = logits.clone()
            out = adapter.apply(logits)
            assert out is logits
            assert torch.equal(logits, before)
        finally:
            adapter.close()

    def test_mixed_batch_torch(self) -> None:
        """Torch mixed batch: subset gather + row-restricted one-hot force."""
        torch = pytest.importorskip("torch")
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=MockSamplingParams(extra_args={"qr_bypass": True}),
                        ),
                        MockAddedRequest(req_index=1, sampling_params=MockSamplingParams()),
                    ]
                )
            )
            gen = torch.Generator().manual_seed(13)
            logits = torch.randn((3, 10), generator=gen, dtype=torch.float32)
            before = logits.clone()
            adapter.apply(logits)
            assert torch.equal(logits[0], before[0])
            for i in (1, 2):
                row = logits[i]
                assert int((row == 0.0).sum()) == 1
                assert bool(torch.isinf(row[row != 0.0]).all())
        finally:
            adapter.close()

    def test_force_onehot_rows_matches_per_row_numpy(self) -> None:
        """_force_onehot_rows equals the per-row reference on a row subset
        and leaves unlisted rows untouched (mirrors TestBatchedOnehot)."""
        adapter = _make_adapter(vocab_size=16)
        try:
            row_indices = [0, 2, 3]
            token_ids = [3, 15, 7]
            batch = np.zeros((5, 16), dtype=np.float32)
            adapter._force_onehot_rows(batch, row_indices, token_ids, is_numpy=True)

            reference = np.zeros((5, 16), dtype=np.float32)
            for i, token in zip(row_indices, token_ids, strict=True):
                adapter._force_onehot_row(reference, i, token, is_numpy=True)
            assert np.array_equal(batch, reference)
        finally:
            adapter.close()

    def test_force_onehot_rows_matches_per_row_torch(self) -> None:
        torch = pytest.importorskip("torch")
        adapter = _make_adapter(vocab_size=16)
        try:
            row_indices = [1, 3]
            token_ids = [5, 9]
            batch = torch.zeros((4, 16), dtype=torch.float32)
            adapter._force_onehot_rows(batch, row_indices, token_ids, is_numpy=False)

            reference = torch.zeros((4, 16), dtype=torch.float32)
            for i, token in zip(row_indices, token_ids, strict=True):
                adapter._force_onehot_row(reference, i, token, is_numpy=False)
            assert torch.equal(batch, reference)
        finally:
            adapter.close()


class TestBypassValidateParams:
    """validate_params (API-server side) treatment of qr_bypass."""

    def test_accepts_qr_bypass_bool_and_string(self) -> None:
        VLLMAdapter.validate_params(MockSamplingParams(extra_args={"qr_bypass": True}))
        VLLMAdapter.validate_params(MockSamplingParams(extra_args={"qr_bypass": "true"}))

    def test_unknown_source_still_rejected_alongside_bypass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DELIBERATE asymmetry — do not 'fix' by exempting bypass requests:
        validate_params rejects an un-preinitialised qr_entropy_source_type
        VALUE even when qr_bypass=true would make the engine-side pipeline
        lookup unreachable. The API-side allowlist check stays unconditional
        (defense in depth, AUDIT A-1); only the engine-worker short-circuit
        is lenient, because raising THERE kills the shared engine."""
        monkeypatch.delenv("QR_PREINIT_ENTROPY_SOURCES", raising=False)
        monkeypatch.delenv("QR_ENTROPY_SOURCE_INSTANCES", raising=False)
        params = MockSamplingParams(
            extra_args={"qr_bypass": True, "qr_entropy_source_type": "nonexistent"}
        )
        with pytest.raises(ConfigValidationError, match="not pre-initialised"):
            VLLMAdapter.validate_params(params)
