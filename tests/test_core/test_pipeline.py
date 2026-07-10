"""Tests for core/pipeline.py — SamplingPipeline and factory functions.

Uses MockUniformSource and numpy arrays. No torch or GPU required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import (
    SamplingPipeline,
    accepts_config,
    build_entropy_source,
    build_pipeline,
    config_hash,
)
from qr_sampler.core.types import SamplingResult
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.temperature.base import TemperatureResult, TemperatureStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides: Any) -> QRSamplerConfig:
    """Create a config with mock source defaults, preventing env interference."""
    defaults: dict[str, Any] = {
        "entropy_source_type": "mock_uniform",
        "fallback_mode": "error",
        "log_level": "none",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


def _make_pipeline(
    vocab_size: int = 10,
    **config_overrides: Any,
) -> SamplingPipeline:
    """Build a SamplingPipeline using mock entropy for testing."""
    config = _make_config(**config_overrides)
    return build_pipeline(config, vocab_size)


# ---------------------------------------------------------------------------
# Tests: config_hash
# ---------------------------------------------------------------------------


class TestConfigHash:
    """Test config_hash determinism and format."""

    def test_deterministic(self) -> None:
        """Same config produces the same hash."""
        cfg = _make_config()
        h1 = config_hash(cfg)
        h2 = config_hash(cfg)
        assert h1 == h2

    def test_length_16(self) -> None:
        """Hash is exactly 16 hex characters."""
        cfg = _make_config()
        h = config_hash(cfg)
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_configs_different_hashes(self) -> None:
        """Different configs produce different hashes."""
        cfg1 = _make_config(top_k=10)
        cfg2 = _make_config(top_k=20)
        assert config_hash(cfg1) != config_hash(cfg2)


# ---------------------------------------------------------------------------
# Tests: accepts_config
# ---------------------------------------------------------------------------


class TestAcceptsConfig:
    """Test accepts_config detection."""

    def test_detects_config_by_annotation(self) -> None:
        """Detects QRSamplerConfig annotation on first param."""

        class WithConfig:
            def __init__(self, config: QRSamplerConfig) -> None:
                pass

        assert accepts_config(WithConfig) is True

    def test_detects_config_by_name(self) -> None:
        """Detects param named 'config' without annotation."""

        class WithConfigName:
            def __init__(self, config) -> None:  # type: ignore[no-untyped-def]
                pass

        assert accepts_config(WithConfigName) is True

    def test_rejects_no_config(self) -> None:
        """Returns False when no config param exists."""

        class NoConfig:
            def __init__(self) -> None:
                pass

        assert accepts_config(NoConfig) is False

    def test_rejects_different_first_param(self) -> None:
        """Returns False when first param is not config-like."""

        class DifferentParam:
            def __init__(self, vocab_size: int) -> None:
                pass

        assert accepts_config(DifferentParam) is False


# ---------------------------------------------------------------------------
# Tests: build_entropy_source
# ---------------------------------------------------------------------------


class TestBuildEntropySource:
    """Test build_entropy_source factory."""

    def test_builds_mock_source(self) -> None:
        """Builds a MockUniformSource when configured."""
        cfg = _make_config(entropy_source_type="mock_uniform", fallback_mode="error")
        source = build_entropy_source(cfg)
        assert source.name == "mock_uniform"

    def test_system_fallback_wrapping(self) -> None:
        """With fallback_mode=system, wraps primary in FallbackEntropySource."""
        cfg = _make_config(entropy_source_type="mock_uniform", fallback_mode="system")
        source = build_entropy_source(cfg)
        assert isinstance(source, FallbackEntropySource)
        assert "+" in source.name

    def test_mock_fallback_wrapping(self) -> None:
        """With fallback_mode=mock_uniform, wraps in FallbackEntropySource."""
        cfg = _make_config(entropy_source_type="system", fallback_mode="mock_uniform")
        source = build_entropy_source(cfg)
        assert isinstance(source, FallbackEntropySource)
        assert "+" in source.name

    def test_error_mode_no_wrapping(self) -> None:
        """With fallback_mode=error, returns unwrapped source."""
        cfg = _make_config(entropy_source_type="mock_uniform", fallback_mode="error")
        source = build_entropy_source(cfg)
        assert not isinstance(source, FallbackEntropySource)

    def test_source_provides_bytes(self) -> None:
        """Built source actually provides random bytes."""
        cfg = _make_config()
        source = build_entropy_source(cfg)
        data = source.get_random_bytes(100)
        assert len(data) == 100
        source.close()


# ---------------------------------------------------------------------------
# Tests: build_pipeline
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """Test build_pipeline factory."""

    def test_builds_successfully(self) -> None:
        """build_pipeline returns a SamplingPipeline."""
        pipeline = _make_pipeline()
        assert isinstance(pipeline, SamplingPipeline)
        pipeline.close()

    def test_pipeline_has_correct_config(self) -> None:
        """Pipeline stores the provided config."""
        cfg = _make_config(top_k=42)
        pipeline = build_pipeline(cfg, vocab_size=10)
        assert pipeline.default_config.top_k == 42
        pipeline.close()

    def test_pipeline_has_entropy_source(self) -> None:
        """Pipeline exposes the entropy source."""
        pipeline = _make_pipeline()
        assert pipeline.entropy_source.name == "mock_uniform"
        pipeline.close()

    def test_pipeline_has_sampling_logger(self) -> None:
        """Pipeline exposes the sampling logger."""
        pipeline = _make_pipeline()
        assert pipeline.sampling_logger is not None
        pipeline.close()


# ---------------------------------------------------------------------------
# Tests: SamplingPipeline.sample_token
# ---------------------------------------------------------------------------


class TestSampleToken:
    """Test the core sampling pipeline."""

    def test_returns_sampling_result(self) -> None:
        """sample_token returns a SamplingResult."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert isinstance(result, SamplingResult)
        pipeline.close()

    def test_token_id_in_range(self) -> None:
        """Selected token_id is within vocab range."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert 0 <= result.token_id < 10
        pipeline.close()

    def test_one_hot_shape(self) -> None:
        """one_hot array matches logits shape."""
        pipeline = _make_pipeline(vocab_size=10)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert result.one_hot.shape == (10,)
        pipeline.close()

    def test_one_hot_values(self) -> None:
        """one_hot has 0.0 at token_id and -inf elsewhere."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert result.one_hot[result.token_id] == 0.0
        assert np.sum(result.one_hot == 0.0) == 1
        assert np.sum(np.isneginf(result.one_hot)) == 9
        pipeline.close()

    def test_record_populated(self) -> None:
        """Diagnostic record has valid values."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        record = result.record
        assert record.token_id == result.token_id
        assert 0.0 < record.u_value < 1.0
        assert record.token_rank >= 0
        assert record.token_prob > 0.0
        assert record.num_candidates > 0
        assert record.entropy_fetch_ms >= 0.0
        assert record.total_sampling_ms > 0.0
        assert len(record.config_hash) == 16
        assert record.temperature_used > 0.0
        pipeline.close()

    def test_dominant_token_selected(self) -> None:
        """Overwhelmingly dominant logit is always selected."""
        pipeline = _make_pipeline()
        logits = np.array(
            [-100.0, -100.0, -100.0, 100.0, -100.0, -100.0, -100.0, -100.0, -100.0, -100.0]
        )
        for _ in range(5):
            result = pipeline.sample_token(logits.copy())
            assert result.token_id == 3
        pipeline.close()

    def test_per_request_config_override(self) -> None:
        """Per-request config changes sampling parameters."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])

        override_config = _make_config(top_k=1)
        result = pipeline.sample_token(logits, config=override_config)
        # With top_k=1, only the highest logit (index 0) should be selected.
        assert result.token_id == 0
        pipeline.close()

    def test_per_request_config_hash(self) -> None:
        """Pre-computed config_hash is used in the record."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        custom_hash = "custom_hash_1234"
        result = pipeline.sample_token(logits, config_hash_str=custom_hash)
        assert result.record.config_hash == custom_hash
        pipeline.close()

    def test_entropy_source_tracking(self) -> None:
        """Record tracks which entropy source was used."""
        pipeline = _make_pipeline()
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert result.record.entropy_source_used == "mock_uniform"
        assert result.record.entropy_is_fallback is False
        pipeline.close()

    def test_diagnostic_mode(self) -> None:
        """With diagnostic_mode=True, records are stored in the logger."""
        pipeline = _make_pipeline(diagnostic_mode=True)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        pipeline.sample_token(logits)
        records = pipeline.sampling_logger.get_diagnostic_data()
        assert len(records) == 1
        pipeline.close()

    def test_multiple_tokens(self) -> None:
        """Sampling multiple tokens accumulates diagnostic records."""
        pipeline = _make_pipeline(diagnostic_mode=True)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        for _ in range(3):
            pipeline.sample_token(logits.copy())
        records = pipeline.sampling_logger.get_diagnostic_data()
        assert len(records) == 3
        pipeline.close()


# ---------------------------------------------------------------------------
# Tests: SamplingPipeline.close
# ---------------------------------------------------------------------------


class _StaticTempStrategy(TemperatureStrategy):
    """Returns a fixed TemperatureResult with caller-supplied diagnostics."""

    def __init__(self, temperature: float, diagnostics: dict[str, Any]) -> None:
        self._temperature = temperature
        self._diagnostics = diagnostics

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        return TemperatureResult(
            temperature=self._temperature,
            shannon_entropy=1.0,
            diagnostics=dict(self._diagnostics),
        )


class TestMinPThreading:
    """Pipeline reads ``min_p`` from temperature diagnostics with config fallback."""

    def test_min_p_from_temp_diagnostics_overrides_config_default(self) -> None:
        """Strategy diagnostics ``min_p`` wins over ``config.min_p_base``."""
        pipeline = _make_pipeline(min_p_base=0.0)
        # Highly skewed logits; min_p=0.1 means only the dominant token survives.
        logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        strategy = _StaticTempStrategy(temperature=1.0, diagnostics={"min_p": 0.1})
        result = pipeline.sample_token(logits, strategy=strategy)
        assert result.record.min_p_used == 0.1
        pipeline.close()

    def test_min_p_falls_back_to_config_when_strategy_omits_it(self) -> None:
        """Without ``min_p`` in diagnostics, the selector sees ``config.min_p_base``."""
        pipeline = _make_pipeline(min_p_base=0.07)
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        strategy = _StaticTempStrategy(temperature=1.0, diagnostics={})
        result = pipeline.sample_token(logits, strategy=strategy)
        assert result.record.min_p_used == 0.07
        pipeline.close()


class TestRecordOptionalFields:
    """Optional HVH/preset diagnostic fields on TokenSamplingRecord."""

    def test_record_populated_with_hvh_diagnostics(self) -> None:
        """Using ``hvh_drift`` strategy populates varentropy/h_ema/vh_ema/min_p_used."""
        # Importing the module registers HVHDriftStrategy in the registry.
        import qr_sampler.temperature.hvh_drift  # noqa: F401

        pipeline = _make_pipeline(temperature_strategy="hvh_drift")
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)

        assert result.record.varentropy is not None
        assert result.record.varentropy >= 0.0
        assert result.record.h_ema is not None
        assert result.record.vh_ema is not None
        assert result.record.min_p_used is not None
        pipeline.close()

    def test_record_unaffected_for_fixed_strategy(self) -> None:
        """Fixed strategy leaves all five optional fields at sensible defaults."""
        pipeline = _make_pipeline(temperature_strategy="fixed")
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)

        # HVH-only diagnostics are absent → None.
        assert result.record.varentropy is None
        assert result.record.h_ema is None
        assert result.record.vh_ema is None
        # No env-var preset configured → None.
        assert result.record.preset_active is None
        # min_p_used always populated from selector diagnostics; defaults to 0.0.
        assert result.record.min_p_used == 0.0
        pipeline.close()


class TestPipelineClose:
    """Test pipeline resource cleanup."""

    def test_close_delegates_to_source(self) -> None:
        """close() delegates to entropy source."""
        pipeline = _make_pipeline()
        pipeline.close()  # Should not raise.

    def test_close_idempotent(self) -> None:
        """close() can be called multiple times safely."""
        pipeline = _make_pipeline()
        pipeline.close()
        pipeline.close()  # Should not raise.


# ---------------------------------------------------------------------------
# Tests: Import constraint validation
# ---------------------------------------------------------------------------


class TestImportConstraints:
    """Verify that core/pipeline.py has no forbidden imports."""

    def test_no_torch_import(self) -> None:
        """core/pipeline.py must not import torch."""
        import qr_sampler.core.pipeline as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        # Check for direct 'import torch' or 'from torch' in source.
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("import torch"), f"Forbidden import found: {stripped}"
            assert not stripped.startswith("from torch"), f"Forbidden import found: {stripped}"

    def test_no_vllm_import(self) -> None:
        """core/pipeline.py must not import vllm."""
        import qr_sampler.core.pipeline as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")  # type: ignore[arg-type]
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("import vllm"), f"Forbidden import found: {stripped}"
            assert not stripped.startswith("from vllm"), f"Forbidden import found: {stripped}"


# ---------------------------------------------------------------------------
# Tests: named entropy-source instances in build_entropy_source
# ---------------------------------------------------------------------------


class TestBuildEntropySourceInstances:
    """Instance names resolve to their underlying type with overrides applied,
    and the built source reports the INSTANCE name end-to-end."""

    @staticmethod
    def _register_probe() -> type[Any]:
        """Register a temporary config-capturing source under 'cfg_probe'."""
        from qr_sampler.entropy.base import EntropySource
        from qr_sampler.entropy.registry import EntropySourceRegistry

        @EntropySourceRegistry.register("cfg_probe")
        class ConfigProbeSource(EntropySource):
            def __init__(self, config: QRSamplerConfig) -> None:
                self.captured_config = config

            @property
            def name(self) -> str:
                return "cfg_probe"

            @property
            def is_available(self) -> bool:
                return True

            def get_random_bytes(self, n: int) -> bytes:
                return b"\x7f" * n

            def close(self) -> None:
                pass

        return ConfigProbeSource

    @staticmethod
    def _unregister_probe() -> None:
        from qr_sampler.entropy.registry import EntropySourceRegistry

        EntropySourceRegistry._registry.pop("cfg_probe", None)

    def test_instance_resolves_type_and_overrides(self) -> None:
        """The instance's underlying source is built against a config copy
        carrying the allowlisted infrastructure overrides."""
        from qr_sampler.entropy.named import InstanceNamedSource

        self._register_probe()
        try:
            config = _make_config(
                entropy_source_type="prng_lane",
                entropy_source_instances={
                    "prng_lane": {
                        "type": "cfg_probe",
                        "grpc_api_key": "lane-key",
                        "grpc_server_address": "unix:///lane.sock",
                    }
                },
            )
            source = build_entropy_source(config)
            assert isinstance(source, InstanceNamedSource)
            assert source.name == "prng_lane"
            captured = source.inner.captured_config  # type: ignore[attr-defined]
            assert captured.entropy_source_type == "cfg_probe"
            assert captured.grpc_api_key == "lane-key"
            assert captured.grpc_server_address == "unix:///lane.sock"
        finally:
            self._unregister_probe()

    def test_instance_with_fallback_labels_primary_leg(self) -> None:
        """With fallback wrapping, the PRIMARY leg carries the instance name —
        the status-file/log leg labels read primary.name."""
        config = _make_config(
            entropy_source_type="prng_lane",
            fallback_mode="system",
            entropy_source_instances={"prng_lane": {"type": "mock_uniform"}},
        )
        source = build_entropy_source(config)
        assert isinstance(source, FallbackEntropySource)
        assert source.primary_name == "prng_lane"
        assert source.name == "prng_lane+system"

    def test_record_carries_instance_name(self) -> None:
        """TokenSamplingRecord.entropy_source_used reports the instance name."""
        pipeline = _make_pipeline(
            entropy_source_type="prng_lane",
            entropy_source_instances={"prng_lane": {"type": "mock_uniform"}},
        )
        logits = np.array([3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0, -4.0])
        result = pipeline.sample_token(logits)
        assert result.record.entropy_source_used == "prng_lane"

    def test_non_instance_path_unchanged(self) -> None:
        """Regression: without declared instances the built source is the
        plain type-named source, exactly as before."""
        config = _make_config()
        source = build_entropy_source(config)
        assert source.name == "mock_uniform"
