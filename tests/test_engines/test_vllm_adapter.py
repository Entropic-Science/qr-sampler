"""Tests for VLLMAdapter — the vLLM engine adapter.

Verifies that VLLMAdapter delegates sampling to SamplingPipeline
and handles batch state management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import SamplingPipeline
from qr_sampler.engines.base import EngineAdapter
from qr_sampler.engines.vllm.adapter import _DEFAULT_VOCAB_SIZE, VLLMAdapter
from qr_sampler.exceptions import ConfigValidationError

# ---------------------------------------------------------------------------
# Mock objects simulating vLLM's batch management types
# ---------------------------------------------------------------------------


@dataclass
class MockVllmConfig:
    """Simulates vLLM's VllmConfig with vocab_size access."""

    vocab_size: int = 10


@dataclass
class MockModelConfig:
    """Simulates vLLM's model config nested structure."""

    hf_text_config: Any = None


@dataclass
class MockHfTextConfig:
    """Simulates the HuggingFace text config with vocab_size."""

    vocab_size: int = 10


@dataclass
class MockSamplingParams:
    """Simulates vLLM's SamplingParams."""

    extra_args: dict[str, Any] | None = None


@dataclass
class MockAddedRequest:
    """Simulates a BatchUpdate added request."""

    req_index: int
    sampling_params: MockSamplingParams | None = None


@dataclass
class MockMovedRequest:
    """Simulates a BatchUpdate moved request."""

    src_index: int
    dst_index: int


@dataclass
class MockBatchUpdate:
    """Simulates vLLM's BatchUpdate dataclass."""

    removed: list[int] | None = None
    moved: list[MockMovedRequest] | None = None
    added: list[MockAddedRequest] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


# ---------------------------------------------------------------------------
# Helper to create an adapter with MockUniformSource
# ---------------------------------------------------------------------------


def _make_adapter(
    vocab_size: int = 10,
    entropy_source_type: str = "mock_uniform",
    fallback_mode: str = "error",
    preinit_sources: str | None = None,
    **config_overrides: Any,
) -> VLLMAdapter:
    """Create an adapter using mock entropy (no gRPC, no GPU).

    Sets environment variables to configure, then instantiates.

    ``preinit_sources`` (mapped to ``QR_PREINIT_ENTROPY_SOURCES``) defaults
    to ``entropy_source_type`` so tests do not pay the gRPC connection cost
    that the production default ``"quantum_grpc,system"`` would incur.
    """
    import os

    env_vars = {
        "QR_ENTROPY_SOURCE_TYPE": entropy_source_type,
        "QR_FALLBACK_MODE": fallback_mode,
        "QR_LOG_LEVEL": "none",
        "QR_PREINIT_ENTROPY_SOURCES": (
            preinit_sources if preinit_sources is not None else entropy_source_type
        ),
    }
    for key, value in config_overrides.items():
        env_vars[f"QR_{key.upper()}"] = str(value)

    old_env: dict[str, str | None] = {}
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        vllm_config = MockVllmConfig(vocab_size=vocab_size)
        adapter = VLLMAdapter(
            vllm_config=vllm_config,
            device=None,
            is_pin_memory=False,
        )
    finally:
        for key, original in old_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    return adapter


# ---------------------------------------------------------------------------
# Tests: Adapter basics and EngineAdapter contract
# ---------------------------------------------------------------------------


class TestVLLMAdapterInit:
    """Test VLLMAdapter construction and EngineAdapter contract."""

    def test_is_engine_adapter(self) -> None:
        """VLLMAdapter is a subclass of EngineAdapter."""
        assert issubclass(VLLMAdapter, EngineAdapter)

    def test_init_with_mock_source(self) -> None:
        """Adapter initializes successfully with mock entropy source."""
        adapter = _make_adapter()
        assert adapter._vocab_size == 10
        assert adapter.is_argmax_invariant() is False

    def test_get_pipeline_returns_sampling_pipeline(self) -> None:
        """get_pipeline() returns a SamplingPipeline instance."""
        adapter = _make_adapter()
        pipeline = adapter.get_pipeline()
        assert isinstance(pipeline, SamplingPipeline)
        adapter.close()

    def test_init_with_none_vllm_config(self) -> None:
        """When vllm_config is None, uses default vocab size."""
        import os

        os.environ["QR_ENTROPY_SOURCE_TYPE"] = "mock_uniform"
        os.environ["QR_FALLBACK_MODE"] = "error"
        os.environ["QR_LOG_LEVEL"] = "none"
        os.environ["QR_PREINIT_ENTROPY_SOURCES"] = "mock_uniform"
        try:
            adapter = VLLMAdapter(vllm_config=None)
            assert adapter._vocab_size == _DEFAULT_VOCAB_SIZE
        finally:
            os.environ.pop("QR_ENTROPY_SOURCE_TYPE", None)
            os.environ.pop("QR_FALLBACK_MODE", None)
            os.environ.pop("QR_LOG_LEVEL", None)
            os.environ.pop("QR_PREINIT_ENTROPY_SOURCES", None)

    def test_init_with_nested_vllm_config(self) -> None:
        """Extracts vocab_size from nested vLLM config structure."""
        hf = MockHfTextConfig(vocab_size=256)
        model_cfg = MockModelConfig(hf_text_config=hf)

        @dataclass
        class NestedConfig:
            model_config: Any = None

        config = NestedConfig(model_config=model_cfg)
        vocab = VLLMAdapter._extract_vocab_size(config)
        assert vocab == 256

    def test_extract_vocab_size_fallback(self) -> None:
        """Falls back to default when config has no vocab_size."""

        class EmptyConfig:
            pass

        vocab = VLLMAdapter._extract_vocab_size(EmptyConfig())
        assert vocab == _DEFAULT_VOCAB_SIZE


# ---------------------------------------------------------------------------
# Tests: Pipeline delegation
# ---------------------------------------------------------------------------


class TestPipelineDelegation:
    """Test that VLLMAdapter delegates sampling to SamplingPipeline."""

    def test_apply_delegates_to_pipeline(self) -> None:
        """apply() uses pipeline.sample_token() internally."""
        adapter = _make_adapter()
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = adapter.apply(logits)

        # Verify one-hot output structure.
        row = result[0]
        assert np.sum(row == 0.0) == 1
        assert np.sum(np.isneginf(row)) == 9
        adapter.close()

    def test_entropy_source_property(self) -> None:
        """entropy_source property delegates to pipeline."""
        adapter = _make_adapter()
        assert adapter.entropy_source is adapter.get_pipeline().entropy_source
        adapter.close()

    def test_sampling_logger_property(self) -> None:
        """sampling_logger property delegates to pipeline."""
        adapter = _make_adapter()
        assert adapter.sampling_logger is adapter.get_pipeline().sampling_logger
        adapter.close()

    def test_default_config_property(self) -> None:
        """default_config property returns the adapter's config."""
        adapter = _make_adapter()
        assert isinstance(adapter.default_config, QRSamplerConfig)
        adapter.close()

    def test_qr_preset_expands_into_the_process_default(self) -> None:
        """QR_PRESET selects the whole sampling profile at process init — no
        per-request extra_args. ``qthought_purity`` makes the default config a
        server-integrated draw (the amplify-in-Qbert migrate), and the default
        pipeline's amplifier reports ``requires_server_draw``."""
        import os

        for key, val in {
            "QR_PRESET": "qthought_purity",
            "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
            "QR_FALLBACK_MODE": "error",
            "QR_LOG_LEVEL": "none",
            "QR_PREINIT_ENTROPY_SOURCES": "mock_uniform",
        }.items():
            os.environ[key] = val
        try:
            adapter = VLLMAdapter(vllm_config=None)
            cfg = adapter.default_config
            assert cfg.signal_amplifier_type == "server"
            assert cfg.draw_block_bytes == 102400
            assert cfg.temperature_strategy == "coherence_gate"
            # The preset field is cleared so per-request resolve does not re-expand.
            assert cfg.preset == ""
            assert getattr(adapter._pipeline.amplifier, "requires_server_draw", False) is True
            adapter.close()
        finally:
            for key in (
                "QR_PRESET",
                "QR_ENTROPY_SOURCE_TYPE",
                "QR_FALLBACK_MODE",
                "QR_LOG_LEVEL",
                "QR_PREINIT_ENTROPY_SOURCES",
            ):
                os.environ.pop(key, None)

    def test_close_delegates_to_pipeline(self) -> None:
        """close() delegates to pipeline.close()."""
        adapter = _make_adapter()
        adapter.close()  # Should not raise.

    def test_close_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        adapter = _make_adapter()
        adapter.close()
        adapter.close()  # Should not raise.


# ---------------------------------------------------------------------------
# Tests: Batch processing
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    """Test apply() with various batch shapes."""

    def test_single_row_onehot(self) -> None:
        """apply() produces one-hot output for a single-row batch."""
        adapter = _make_adapter()
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = adapter.apply(logits)
        assert result is logits  # In-place modification.
        row = result[0]
        assert np.sum(row == 0.0) == 1
        assert np.sum(np.isneginf(row)) == 9

    def test_multi_row_batch(self) -> None:
        """apply() processes all rows in a batch."""
        adapter = _make_adapter()
        logits = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    10.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                ],
            ]
        )
        result = adapter.apply(logits)
        for i in range(3):
            row = result[i]
            assert np.sum(row == 0.0) == 1
            assert np.sum(np.isneginf(row)) == 9

    def test_1d_logits(self) -> None:
        """apply() handles 1-D logits (single request, no batch dim)."""
        adapter = _make_adapter()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = adapter.apply(logits)
        assert np.sum(result == 0.0) == 1
        assert np.sum(np.isneginf(result)) == 9

    def test_empty_batch(self) -> None:
        """apply() short-circuits on empty batch."""
        adapter = _make_adapter()
        logits = np.empty((0, 10))
        result = adapter.apply(logits)
        assert result.shape == (0, 10)

    def test_dominant_token_selected(self) -> None:
        """A very dominant logit is always selected."""
        adapter = _make_adapter()
        logits = np.array(
            [
                [
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                    -100.0,
                ]
            ]
        )
        result = adapter.apply(logits)
        assert result[0, 4] == 0.0

    def test_inplace_modification(self) -> None:
        """apply() modifies the logits array in-place and returns it."""
        adapter = _make_adapter()
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = adapter.apply(logits)
        assert result is logits


# ---------------------------------------------------------------------------
# Tests: update_state
# ---------------------------------------------------------------------------


class TestUpdateState:
    """Test update_state() batch management."""

    def test_add_request(self) -> None:
        """Adding a request creates per-request state."""
        adapter = _make_adapter()
        batch = MockBatchUpdate(
            added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
        )
        adapter.update_state(batch)
        assert 0 in adapter._request_states

    def test_add_request_with_overrides(self) -> None:
        """Added request with extra_args gets resolved config."""
        adapter = _make_adapter()
        params = MockSamplingParams(extra_args={"qr_top_k": 100})
        batch = MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
        adapter.update_state(batch)
        assert adapter._request_states[0].config.top_k == 100

    def test_remove_request(self) -> None:
        """Removing a request cleans up per-request state."""
        adapter = _make_adapter()
        adapter.update_state(
            MockBatchUpdate(
                added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
            )
        )
        assert 0 in adapter._request_states
        adapter.update_state(MockBatchUpdate(removed=[0]))
        assert 0 not in adapter._request_states

    def test_move_request(self) -> None:
        """Moving a request updates state index."""
        adapter = _make_adapter()
        adapter.update_state(
            MockBatchUpdate(
                added=[MockAddedRequest(req_index=0, sampling_params=MockSamplingParams())]
            )
        )
        adapter.update_state(MockBatchUpdate(moved=[MockMovedRequest(src_index=0, dst_index=5)]))
        assert 0 not in adapter._request_states
        assert 5 in adapter._request_states

    # ── vLLM V1 TUPLE ABI ──────────────────────────────────────────────────
    # The object-shaped mocks above masked an ABI drift: real vLLM V1 passes
    # AddedRequest as a tuple (index, params, prompt_tok_ids, output_tok_ids)
    # and MovedRequest as (src, dst, MoveDirectionality). Reading .req_index /
    # .sampling_params off a tuple returns None, so EVERY addition was skipped,
    # no per-request state was built, and per-request extra_args silently died
    # (all tokens routed to the process default). These pin the real shape.

    def test_add_request_tuple_abi_applies_extra_args(self) -> None:
        adapter = _make_adapter()
        params = MockSamplingParams(extra_args={"qr_top_k": 100})
        adapter.update_state(MockBatchUpdate(added=[(0, params, None, [])]))
        assert 0 in adapter._request_states
        # The whole point: the per-request override actually reached the config.
        assert adapter._request_states[0].config.top_k == 100

    def test_remove_and_move_tuple_abi(self) -> None:
        class _Dir:
            def __init__(self, name: str) -> None:
                self.name = name

        adapter = _make_adapter()
        adapter.update_state(MockBatchUpdate(added=[(0, MockSamplingParams(), None, [])]))
        assert 0 in adapter._request_states
        adapter.update_state(MockBatchUpdate(moved=[(0, 5, _Dir("UNIDIRECTIONAL"))]))
        assert 0 not in adapter._request_states
        assert 5 in adapter._request_states
        adapter.update_state(MockBatchUpdate(removed=[5]))
        assert 5 not in adapter._request_states

    def test_swap_tuple_abi_exchanges_states(self) -> None:
        class _Dir:
            def __init__(self, name: str) -> None:
                self.name = name

        adapter = _make_adapter()
        adapter.update_state(
            MockBatchUpdate(
                added=[
                    (0, MockSamplingParams(extra_args={"qr_top_k": 11}), None, []),
                    (1, MockSamplingParams(extra_args={"qr_top_k": 22}), None, []),
                ]
            )
        )
        adapter.update_state(MockBatchUpdate(moved=[(0, 1, _Dir("SWAP"))]))
        assert adapter._request_states[0].config.top_k == 22
        assert adapter._request_states[1].config.top_k == 11

    def test_none_batch_update(self) -> None:
        """None batch_update is a no-op."""
        adapter = _make_adapter()
        adapter.update_state(None)  # Should not raise.

    def test_per_request_config_in_apply(self) -> None:
        """Per-request config affects token selection parameters."""
        adapter = _make_adapter()
        params = MockSamplingParams(extra_args={"qr_top_k": 1})
        adapter.update_state(
            MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
        )
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        result = adapter.apply(logits)
        # With top_k=1, only the highest logit (index 0) should be selected.
        assert result[0, 0] == 0.0


# ---------------------------------------------------------------------------
# Tests: validate_params
# ---------------------------------------------------------------------------


class TestValidateParams:
    """Test validate_params() classmethod."""

    def test_valid_extra_args(self) -> None:
        """Valid qr_ keys pass validation."""
        params = MockSamplingParams(extra_args={"qr_top_k": 100})
        VLLMAdapter.validate_params(params)

    def test_invalid_key_raises(self) -> None:
        """Unknown qr_ key raises ConfigValidationError."""
        params = MockSamplingParams(extra_args={"qr_nonexistent": 42})
        with pytest.raises(ConfigValidationError):
            VLLMAdapter.validate_params(params)

    def test_non_overridable_field_raises(self) -> None:
        """Infrastructure field raises ConfigValidationError."""
        params = MockSamplingParams(extra_args={"qr_grpc_server_address": "foo"})
        with pytest.raises(ConfigValidationError):
            VLLMAdapter.validate_params(params)

    def test_empty_extra_args(self) -> None:
        """Empty extra_args passes validation."""
        params = MockSamplingParams(extra_args={})
        VLLMAdapter.validate_params(params)

    def test_no_extra_args(self) -> None:
        """Missing extra_args passes validation."""
        params = MockSamplingParams(extra_args=None)
        VLLMAdapter.validate_params(params)

    def test_qr_preset_accepted(self) -> None:
        """A known qr_preset must pass validation.

        Regression guard: validate_params is vLLM's pre-batch validation
        hook. Without preset awareness it would reject every request that
        opts into a preset via ``extra_args={"qr_preset": "..."}``,
        breaking the documented per-request preset flow (README curl
        example, Open WebUI UserValves toggle, Python client examples).
        """
        params = MockSamplingParams(extra_args={"qr_preset": "creative_sampling"})
        VLLMAdapter.validate_params(params)

        params = MockSamplingParams(extra_args={"qr_preset": "normal_t1"})
        VLLMAdapter.validate_params(params)

    def test_qr_preset_with_extra_overrides_accepted(self) -> None:
        """A preset alongside per-request overrides must pass validation."""
        params = MockSamplingParams(
            extra_args={"qr_preset": "creative_sampling", "qr_hvh_t_base": 1.2},
        )
        VLLMAdapter.validate_params(params)

    def test_unknown_qr_preset_raises(self) -> None:
        """Unknown preset name fails with a helpful error message."""
        params = MockSamplingParams(extra_args={"qr_preset": "not_a_real_preset"})
        with pytest.raises(ConfigValidationError, match="Unknown preset"):
            VLLMAdapter.validate_params(params)


# ---------------------------------------------------------------------------
# Tests: Diagnostic logging
# ---------------------------------------------------------------------------


class TestDiagnosticLogging:
    """Test that the adapter produces valid diagnostic records."""

    def test_diagnostic_records_stored(self) -> None:
        """With diagnostic_mode=True, records are stored."""
        adapter = _make_adapter(diagnostic_mode=True)
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        adapter.apply(logits)

        records = adapter.sampling_logger.get_diagnostic_data()
        assert len(records) == 1

        record = records[0]
        assert record.token_id >= 0
        assert record.token_id < 10
        assert 0.0 < record.u_value < 1.0
        assert record.token_rank >= 0
        assert record.token_prob > 0.0
        assert record.num_candidates > 0
        assert record.entropy_fetch_ms >= 0.0
        assert record.total_sampling_ms > 0.0
        assert len(record.config_hash) == 16
        assert record.temperature_used > 0.0

    def test_entropy_source_tracking(self) -> None:
        """Diagnostic records track which entropy source was used."""
        adapter = _make_adapter(diagnostic_mode=True)
        logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
        adapter.apply(logits)

        record = adapter.sampling_logger.get_diagnostic_data()[0]
        assert record.entropy_source_used == "mock_uniform"
        assert record.entropy_is_fallback is False


# ---------------------------------------------------------------------------
# Tests: Public exports
# ---------------------------------------------------------------------------


class TestPublicExports:
    """Test the top-level package export surface."""

    def test_exports_available(self) -> None:
        """Core exports are available from the top-level package."""
        from qr_sampler import (
            EngineAdapter,
            SamplingPipeline,
            SamplingResult,
            build_pipeline,
        )

        assert EngineAdapter is not None
        assert SamplingPipeline is not None
        assert SamplingResult is not None
        assert build_pipeline is not None


# ---------------------------------------------------------------------------
# Tests: Per-request entropy source override
# ---------------------------------------------------------------------------


class TestPerRequestEntropySourceOverride:
    """Tests for the per-request ``qr_entropy_source_type`` override.

    Comparison mode depends on this: the OWUI pipe issues two parallel
    requests against one vLLM engine, one with ``quantum_grpc`` entropy
    and one with ``system`` entropy. The same model weights and KV cache
    are shared; only the entropy source differs.
    """

    def test_preinit_sources_from_env(self) -> None:
        """The adapter pre-initialises one pipeline per source in the env."""
        adapter = _make_adapter(
            entropy_source_type="mock_uniform",
            preinit_sources="mock_uniform,system",
        )
        assert set(adapter._pipelines.keys()) == {"mock_uniform", "system"}
        # Each pipeline points at the right entropy source class.
        assert adapter._pipelines["mock_uniform"].entropy_source.name == "mock_uniform"
        assert adapter._pipelines["system"].entropy_source.name == "system"
        adapter.close()

    def test_default_source_always_present(self) -> None:
        """If the env list omits the default source, it is auto-included."""
        adapter = _make_adapter(
            entropy_source_type="mock_uniform",
            preinit_sources="system",
        )
        # mock_uniform is the default → must be present even though the env
        # list only mentioned 'system'.
        assert "mock_uniform" in adapter._pipelines
        assert "system" in adapter._pipelines
        adapter.close()

    def test_preinit_whitespace_tolerated(self) -> None:
        """Whitespace around commas in the env list is stripped."""
        adapter = _make_adapter(
            entropy_source_type="mock_uniform",
            preinit_sources=" mock_uniform , system ,",
        )
        assert set(adapter._pipelines.keys()) == {"mock_uniform", "system"}
        adapter.close()

    def test_per_request_source_override(self) -> None:
        """Two requests in one batch with different qr_entropy_source_type
        end up routed to pipelines with the correct entropy sources."""
        from qr_sampler.entropy.mock import MockUniformSource
        from qr_sampler.entropy.system import SystemEntropySource

        adapter = _make_adapter(
            entropy_source_type="mock_uniform",
            preinit_sources="mock_uniform,system",
        )

        batch = MockBatchUpdate(
            added=[
                MockAddedRequest(
                    req_index=0,
                    sampling_params=MockSamplingParams(
                        extra_args={"qr_entropy_source_type": "mock_uniform"},
                    ),
                ),
                MockAddedRequest(
                    req_index=1,
                    sampling_params=MockSamplingParams(
                        extra_args={"qr_entropy_source_type": "system"},
                    ),
                ),
            ]
        )
        adapter.update_state(batch)

        # Per the plan test description: assert each used the correct source
        # via _request_states[i].source.__class__.__name__.
        assert adapter._request_states[0].source.__class__ is MockUniformSource
        assert adapter._request_states[1].source.__class__ is SystemEntropySource

        # Both pipelines are distinct — fan-out is not aliasing one pipeline.
        assert adapter._request_states[0].pipeline is not adapter._request_states[1].pipeline

        # And apply() still produces valid one-hot output for both rows.
        logits = np.array(
            [
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
                [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
            ]
        )
        result = adapter.apply(logits)
        for i in range(2):
            row = result[i]
            assert np.sum(row == 0.0) == 1
            assert np.sum(np.isneginf(row)) == 9

        adapter.close()

    def test_rejects_uninit_source(self) -> None:
        """Requesting an entropy source that was not pre-initialised raises
        a clean ConfigValidationError, not a KeyError or AttributeError."""
        adapter = _make_adapter(
            entropy_source_type="system",
            preinit_sources="system",
        )

        batch = MockBatchUpdate(
            added=[
                MockAddedRequest(
                    req_index=0,
                    sampling_params=MockSamplingParams(
                        extra_args={"qr_entropy_source_type": "openentropy"},
                    ),
                ),
            ]
        )
        with pytest.raises(ConfigValidationError, match="not pre-initialised"):
            adapter.update_state(batch)
        adapter.close()

    def test_validate_params_accepts_entropy_source_type(self) -> None:
        """validate_params() accepts qr_entropy_source_type — the field is
        per-request overridable, and 'system' is in the default preinit
        allowlist. update_state() keeps the authoritative pipeline check."""
        params = MockSamplingParams(extra_args={"qr_entropy_source_type": "system"})
        VLLMAdapter.validate_params(params)  # Should not raise.

    def test_validate_params_rejects_unknown_source_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AUDIT A-1: an un-preinitialised qr_entropy_source_type VALUE is
        rejected at request validation (API-server side) — before the request
        can reach the engine worker, where a raise kills the shared engine."""
        monkeypatch.delenv("QR_PREINIT_ENTROPY_SOURCES", raising=False)
        monkeypatch.delenv("QR_ENTROPY_SOURCE_INSTANCES", raising=False)
        params = MockSamplingParams(extra_args={"qr_entropy_source_type": "nonexistent"})
        with pytest.raises(ConfigValidationError, match="not pre-initialised"):
            VLLMAdapter.validate_params(params)

    def test_validate_params_accepts_declared_instance_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Instance names declared via QR_ENTROPY_SOURCE_INSTANCES pass
        request validation; undeclared ones are rejected in the same call."""
        monkeypatch.setenv(
            "QR_ENTROPY_SOURCE_INSTANCES",
            '{"qbert_prng_uniform":{"type":"quantum_grpc","grpc_api_key":"k"}}',
        )
        VLLMAdapter.validate_params(
            MockSamplingParams(extra_args={"qr_entropy_source_type": "qbert_prng_uniform"})
        )
        with pytest.raises(ConfigValidationError, match="not pre-initialised"):
            VLLMAdapter.validate_params(
                MockSamplingParams(extra_args={"qr_entropy_source_type": "qbert_prng_markov"})
            )

    def test_validate_params_default_env_allowlist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No-instances regression: with env unset the allowlist is exactly
        the default preinit set plus the process-default source."""
        monkeypatch.delenv("QR_PREINIT_ENTROPY_SOURCES", raising=False)
        monkeypatch.delenv("QR_ENTROPY_SOURCE_INSTANCES", raising=False)
        monkeypatch.delenv("QR_ENTROPY_SOURCE_TYPE", raising=False)
        for name in ("quantum_grpc", "system"):
            VLLMAdapter.validate_params(
                MockSamplingParams(extra_args={"qr_entropy_source_type": name})
            )
        with pytest.raises(ConfigValidationError, match="not pre-initialised"):
            VLLMAdapter.validate_params(
                MockSamplingParams(extra_args={"qr_entropy_source_type": "mock_uniform"})
            )

    def test_validate_params_tolerates_malformed_instances_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed QR_ENTROPY_SOURCE_INSTANCES contributes no names (the
        engine would refuse to start on it) — builtins still validate."""
        monkeypatch.setenv("QR_ENTROPY_SOURCE_INSTANCES", "{not json")
        VLLMAdapter.validate_params(
            MockSamplingParams(extra_args={"qr_entropy_source_type": "system"})
        )

    def test_no_override_uses_default_source(self) -> None:
        """A request with no qr_entropy_source_type override routes to the
        pipeline matching the default source."""
        from qr_sampler.entropy.mock import MockUniformSource

        adapter = _make_adapter(
            entropy_source_type="mock_uniform",
            preinit_sources="mock_uniform,system",
        )

        batch = MockBatchUpdate(
            added=[
                MockAddedRequest(req_index=0, sampling_params=MockSamplingParams()),
            ]
        )
        adapter.update_state(batch)

        assert adapter._request_states[0].source.__class__ is MockUniformSource
        adapter.close()


# ---------------------------------------------------------------------------
# Tests: named entropy-source instances
# ---------------------------------------------------------------------------

_INSTANCES_JSON = '{"qbert_prng_uniform": {"type": "mock_uniform"}}'


class TestEntropySourceInstances:
    """Preinit + per-request routing for named entropy-source instances."""

    def test_declared_instances_are_preinitialised(self) -> None:
        """Declared instances get their own pipeline even when absent from
        QR_PREINIT_ENTROPY_SOURCES (union semantics)."""
        adapter = _make_adapter(
            preinit_sources="mock_uniform",
            entropy_source_instances=_INSTANCES_JSON,
        )
        try:
            assert sorted(adapter._pipelines) == ["mock_uniform", "qbert_prng_uniform"]
            instance_pipeline = adapter._pipelines["qbert_prng_uniform"]
            assert instance_pipeline.entropy_source.name == "qbert_prng_uniform"
        finally:
            adapter.close()

    def test_per_request_instance_routing_and_record_label(self) -> None:
        """A request selecting the instance routes to its pipeline, and the
        TokenSamplingRecord carries the INSTANCE name (loud PRNG labelling)."""
        adapter = _make_adapter(
            preinit_sources="mock_uniform",
            entropy_source_instances=_INSTANCES_JSON,
            diagnostic_mode="true",
        )
        try:
            params = MockSamplingParams(extra_args={"qr_entropy_source_type": "qbert_prng_uniform"})
            adapter.update_state(
                MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
            )
            state = adapter._request_states[0]
            assert state.pipeline is adapter._pipelines["qbert_prng_uniform"]
            assert state.dominant_source_name == "qbert_prng_uniform"

            logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
            adapter.apply(logits)
            records = adapter._pipelines["qbert_prng_uniform"].sampling_logger.get_diagnostic_data()
            assert records, "instance pipeline logged no records"
            assert records[-1].entropy_source_used == "qbert_prng_uniform"
        finally:
            adapter.close()

    def test_unpreinitialised_instance_name_rejected(self) -> None:
        """An instance name that was never declared keeps the existing clean
        rejection at request-add time."""
        adapter = _make_adapter(
            preinit_sources="mock_uniform",
            entropy_source_instances=_INSTANCES_JSON,
        )
        try:
            params = MockSamplingParams(extra_args={"qr_entropy_source_type": "undeclared_lane"})
            with pytest.raises(ConfigValidationError, match="not pre-initialised"):
                adapter.update_state(
                    MockBatchUpdate(added=[MockAddedRequest(req_index=0, sampling_params=params)])
                )
        finally:
            adapter.close()

    def test_no_instances_regression(self) -> None:
        """Risk §8.1 pin: with QR_ENTROPY_SOURCE_INSTANCES unset, preinit
        resolution and pipeline set are identical to the pre-instances
        adapter."""
        import os

        assert "QR_ENTROPY_SOURCE_INSTANCES" not in os.environ
        adapter = _make_adapter()
        try:
            assert adapter.default_config.entropy_source_instances == {}
            assert sorted(adapter._pipelines) == ["mock_uniform"]
        finally:
            adapter.close()
        # The one-argument call (no instances) is the legacy shape.
        old_env = os.environ.get("QR_PREINIT_ENTROPY_SOURCES")
        os.environ["QR_PREINIT_ENTROPY_SOURCES"] = "quantum_grpc,system"
        try:
            assert VLLMAdapter._resolve_preinit_sources("system") == [
                "quantum_grpc",
                "system",
            ]
        finally:
            if old_env is None:
                os.environ.pop("QR_PREINIT_ENTROPY_SOURCES", None)
            else:
                os.environ["QR_PREINIT_ENTROPY_SOURCES"] = old_env

    def test_preinit_union_order(self) -> None:
        """Env entries first (documented order), then declared instances,
        then the default source — first occurrence wins."""
        import os

        old_env = os.environ.get("QR_PREINIT_ENTROPY_SOURCES")
        os.environ["QR_PREINIT_ENTROPY_SOURCES"] = "quantum_grpc,lane_a"
        try:
            resolved = VLLMAdapter._resolve_preinit_sources("system", ["lane_a", "lane_b"])
            assert resolved == ["quantum_grpc", "lane_a", "lane_b", "system"]
        finally:
            if old_env is None:
                os.environ.pop("QR_PREINIT_ENTROPY_SOURCES", None)
            else:
                os.environ["QR_PREINIT_ENTROPY_SOURCES"] = old_env
