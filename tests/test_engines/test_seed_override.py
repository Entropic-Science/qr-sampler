"""Tests for the deterministic lane's per-request seeded source override.

Pins docs/determinism.md §6 T3: a request carrying ``qr_seed`` (via the
``deterministic_prng`` preset) rides the DEFAULT pipeline with a
per-request ``SeededPrngSource`` override — deterministic across thread
scheduling, batch composition, and engine preemption re-adds, and drawing
ZERO entropy from the shared pipeline sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qr_sampler.engines.vllm.adapter import VLLMAdapter, _RequestState
from qr_sampler.entropy.seeded import SeededPrngSource
from tests.test_engines.test_vllm_adapter import (
    MockAddedRequest,
    MockBatchUpdate,
    MockSamplingParams,
    _make_adapter,
)

_SEED = 20260827


def _det_params(seed: int = _SEED, **extra: Any) -> MockSamplingParams:
    return MockSamplingParams(
        extra_args={"qr_preset": "deterministic_prng", "qr_seed": seed, **extra}
    )


def _flat_logits(rows: int, vocab: int = 10) -> np.ndarray:
    """Near-uniform logits so the selected token tracks u sensitively."""
    base = np.linspace(0.0, 0.5, vocab, dtype=np.float32)
    return np.tile(base, (rows, 1))


def _selected(row: np.ndarray) -> int:
    """Token id forced by a one-hot row (the single 0.0 entry)."""
    (ids,) = np.nonzero(row == 0.0)
    assert len(ids) == 1
    return int(ids[0])


class TestSeedOverrideUpdateState:
    def test_seeded_request_builds_override_on_default_pipeline(self) -> None:
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=_det_params())]
                )
            )
            state = adapter._request_states[0]
            assert isinstance(state, _RequestState)
            assert isinstance(state.entropy_override, SeededPrngSource)
            assert state.entropy_override.seed == _SEED
            assert state.entropy_override.next_step == 0
            assert state.pipeline is adapter._pipeline
            assert state.dominant_source_name == "seeded_prng"
        finally:
            adapter.close()

    def test_preemption_readd_resumes_counter_from_output_ids(self) -> None:
        """A re-add carrying k already-generated tokens resumes at counter k
        (determinism.md footgun #2 — the load-bearing line)."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=_det_params(),
                            output_tok_ids=[3, 1, 4, 1, 5],
                        )
                    ]
                )
            )
            state = adapter._request_states[0]
            assert isinstance(state, _RequestState)
            assert state.entropy_override is not None
            assert state.entropy_override.next_step == 5
        finally:
            adapter.close()

    def test_tuple_abi_readd_resumes_counter(self) -> None:
        """Same resume via the production tuple ABI
        ``(index, params, prompt_tok_ids, output_tok_ids)``."""
        adapter = _make_adapter()
        try:
            added = (0, _det_params(), [11, 12], [7, 8, 9])
            adapter.update_state(MockBatchUpdate(added=[added]))  # type: ignore[list-item]
            state = adapter._request_states[0]
            assert isinstance(state, _RequestState)
            assert state.entropy_override is not None
            assert state.entropy_override.next_step == 3
        finally:
            adapter.close()

    def test_seeded_add_draws_zero_shared_entropy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Request-add for the deterministic preset touches NO pipeline
        source: no calibration draw (zscore_calibration_samples=0), no
        step-0 prefetch."""
        adapter = _make_adapter()
        try:

            def _boom(*args: Any, **kwargs: Any) -> bytes:
                raise AssertionError("deterministic request drew shared entropy")

            for pipeline in adapter._pipelines.values():
                monkeypatch.setattr(pipeline.entropy_source, "get_random_bytes", _boom)
                monkeypatch.setattr(pipeline.entropy_source, "prefetch", _boom)
            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=_det_params())]
                )
            )
            # The apply path must keep drawing ONLY from the override.
            logits = _flat_logits(rows=1)
            adapter.apply(logits)
        finally:
            adapter.close()

    def test_validate_params_accepts_seeded_prng(self) -> None:
        """seeded_prng is always allowed through validate_params without a
        preinit entry (it is per-request-constructed by design)."""
        VLLMAdapter.validate_params(_det_params(qr_entropy_source_type="seeded_prng"))

    def test_validate_params_rejects_partial_deterministic_shapes(self) -> None:
        """Cross-field/value violations must be rejected API-SIDE by
        validate_params (clean per-request error), never first raise
        inside the engine worker's update_state (EngineDeadError;
        GL-01/A-1). validate_params dry-runs the full resolve_config."""
        from qr_sampler.exceptions import ConfigValidationError

        bad_shapes = [
            {"qr_seed": 42},  # seed without the deterministic envelope
            {"qr_preset": "deterministic_prng"},  # preset without a seed
            {"qr_preset": "deterministic_prng", "qr_seed": 1, "qr_entropy_source_type": "system"},
            {"qr_preset": "deterministic_prng", "qr_seed": -1},  # value bounds
            {"qr_preset": "deterministic_prng", "qr_seed": 2**63},
            {"qr_preset": "deterministic_prng", "qr_seed": 1, "qr_bypass": True},
        ]
        for extra in bad_shapes:
            with pytest.raises(ConfigValidationError) as excinfo:
                VLLMAdapter.validate_params(MockSamplingParams(extra_args=extra))
            # The boundary raise must ALSO be a ValueError: that is what
            # vLLM's OpenAI server maps to a clean per-request 400 —
            # anything else reaches the client as an opaque 500.
            assert isinstance(excinfo.value, ValueError)

    def test_update_state_never_raises_on_bad_config(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Defense in depth: a malformed request that somehow evades
        validate_params must NOT kill the engine worker — the request
        degrades to a loud native-sampler bypass."""
        import logging

        from qr_sampler.engines.vllm.adapter import _BypassState

        adapter = _make_adapter()
        try:
            with caplog.at_level(logging.ERROR, logger="qr_sampler"):
                adapter.update_state(
                    MockBatchUpdate(
                        added=[
                            MockAddedRequest(
                                req_index=0,
                                sampling_params=MockSamplingParams(
                                    extra_args={"qr_seed": 42}  # invalid combo
                                ),
                            )
                        ]
                    )
                )
            assert isinstance(adapter._request_states[0], _BypassState)
            rejected = [
                r
                for r in caplog.records
                if getattr(r, "event", "") == "entropy.request.rejected_config"
            ]
            assert len(rejected) == 1
        finally:
            adapter.close()


class TestSeedOverrideApply:
    def test_same_seed_rows_agree_across_threaded_steps(self) -> None:
        """Two same-seed rows mixed with a QRNG-mock row select identical
        tokens at every step, with the row thread pool enabled — the
        order-free property, end to end, over 50 steps."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(req_index=0, sampling_params=_det_params()),
                        MockAddedRequest(req_index=1, sampling_params=_det_params()),
                        MockAddedRequest(req_index=2, sampling_params=MockSamplingParams()),
                    ]
                )
            )
            for _ in range(50):
                logits = _flat_logits(rows=3)
                adapter.apply(logits)
                assert _selected(logits[0]) == _selected(logits[1])
        finally:
            adapter.close()

    def test_full_replay_is_identical(self) -> None:
        """Remove + re-add at the same seed replays the identical token
        sequence over fixed per-step logits."""
        adapter = _make_adapter()
        try:

            def _run(steps: int) -> list[int]:
                tokens = []
                for _ in range(steps):
                    logits = _flat_logits(rows=1)
                    adapter.apply(logits)
                    tokens.append(_selected(logits[0]))
                return tokens

            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=_det_params())]
                )
            )
            first = _run(12)

            adapter.update_state(MockBatchUpdate(removed=[0]))
            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=_det_params())]
                )
            )
            assert _run(12) == first
        finally:
            adapter.close()

    def test_resumed_replay_matches_uninterrupted_suffix(self) -> None:
        """A preemption-style re-add at k=5 replays exactly steps 5..11 of
        the uninterrupted run. The re-add hands the adapter a LIVE
        emitted-token list (production tuple/attr ABI), which the engine
        keeps appending to — the harness simulates that append."""
        adapter = _make_adapter()
        try:

            def _run(steps: int, live: list[int] | None) -> list[int]:
                tokens = []
                for _ in range(steps):
                    logits = _flat_logits(rows=1)
                    adapter.apply(logits)
                    tok = _selected(logits[0])
                    tokens.append(tok)
                    if live is not None:
                        live.append(tok)
                return tokens

            adapter.update_state(
                MockBatchUpdate(
                    added=[MockAddedRequest(req_index=0, sampling_params=_det_params())]
                )
            )
            full = _run(12, None)  # no live list -> internal counter path

            adapter.update_state(MockBatchUpdate(removed=[0]))
            resumed_live = full[:5]  # 5 tokens already emitted pre-preemption
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(
                            req_index=0,
                            sampling_params=_det_params(),
                            output_tok_ids=resumed_live,
                        )
                    ]
                )
            )
            assert _run(7, resumed_live) == full[5:]
        finally:
            adapter.close()

    def test_step_follows_live_output_list_not_draw_count(self) -> None:
        """The counter is a pure function of TOKEN POSITION: with a live
        output_tok_ids list attached (production tuple ABI), extra draws
        at the same position — vLLM's sampled-then-discarded
        partial-prefill rows — re-draw the SAME block, and the counter
        only advances when the emitted-token list grows."""
        adapter = _make_adapter()
        try:
            live_output: list[int] = []
            added = (0, _det_params(), [1, 2, 3], live_output)
            adapter.update_state(MockBatchUpdate(added=[added]))  # type: ignore[list-item]

            # Two applies WITHOUT growing the list (the discarded-row
            # shape): both must select the identical token from block 0.
            logits_a = _flat_logits(rows=1)
            adapter.apply(logits_a)
            first = _selected(logits_a[0])
            logits_b = _flat_logits(rows=1)
            adapter.apply(logits_b)
            assert _selected(logits_b[0]) == first

            # Growing the list advances the position -> a fresh block.
            live_output.append(first)
            state = adapter._request_states[0]
            assert isinstance(state, _RequestState)
            logits_c = _flat_logits(rows=1)
            adapter.apply(logits_c)
            assert state.entropy_override is not None
            assert state.entropy_override.next_step == 2  # drew block 1
        finally:
            adapter.close()

    def test_live_list_replay_immune_to_discarded_draws(self) -> None:
        """End-to-end: a run polluted by discarded-row extra draws still
        replays bitwise — the load-invariance property the position sync
        exists to restore."""
        adapter = _make_adapter()
        try:

            def _run(extra_draws_at: set[int]) -> list[int]:
                live: list[int] = []
                adapter.update_state(
                    MockBatchUpdate(added=[(0, _det_params(), [9], live)])  # type: ignore[list-item]
                )
                tokens = []
                for step in range(8):
                    if step in extra_draws_at:
                        # Simulate a discarded partial-prefill row: an
                        # apply whose result vLLM throws away (list does
                        # not grow).
                        adapter.apply(_flat_logits(rows=1))
                    logits = _flat_logits(rows=1)
                    adapter.apply(logits)
                    tok = _selected(logits[0])
                    tokens.append(tok)
                    live.append(tok)
                adapter.update_state(MockBatchUpdate(removed=[0]))
                return tokens

            assert _run(set()) == _run({0, 3, 5})
        finally:
            adapter.close()

    def test_different_seeds_diverge(self) -> None:
        """Different seeds produce different token sequences (u streams are
        keyed) — the collision-avoidance property motivating per-request
        opt-in."""
        adapter = _make_adapter()
        try:
            adapter.update_state(
                MockBatchUpdate(
                    added=[
                        MockAddedRequest(req_index=0, sampling_params=_det_params(seed=1)),
                        MockAddedRequest(req_index=1, sampling_params=_det_params(seed=2)),
                    ]
                )
            )
            sequences: tuple[list[int], list[int]] = ([], [])
            for _ in range(30):
                logits = _flat_logits(rows=2)
                adapter.apply(logits)
                sequences[0].append(_selected(logits[0]))
                sequences[1].append(_selected(logits[1]))
            assert sequences[0] != sequences[1]
        finally:
            adapter.close()
