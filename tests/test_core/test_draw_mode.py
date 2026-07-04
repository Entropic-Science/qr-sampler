"""Tests for the server-integrated draw mode (spec FR-S3).

Covers the four layers the draw branch spans:

- ``EntropySource`` base defaults (``supports_server_draw`` /
  ``get_draw`` raises / ``prefetch_draw`` returns ``None``).
- ``FallbackEntropySource``: draws delegate to the PRIMARY only and
  failures raise upward (no byte-failover bookkeeping).
- ``ServerDrawAmplifier``: registry name ``"server"``, marker flag, and
  the deliberately dead local ``amplify()``.
- ``SamplingPipeline.sample_token``: server ``u`` consumed verbatim
  (differential vs a local amplifier producing the same ``u``), DrawMeta
  on result + record, the ``observe_draw_meta`` hook, and the fail-safe
  degradation path (fallback bytes + lazily-cached local zscore_mean).
"""

from __future__ import annotations

import math
import os
from typing import Any, ClassVar

import numpy as np
import pytest

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.server_side import ServerDrawAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import SamplingPipeline
from qr_sampler.entropy.base import DrawMeta, EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.system import SystemEntropySource
from qr_sampler.exceptions import EntropyUnavailableError, SignalAmplificationError
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

VOCAB = 64


def _meta(**overrides: Any) -> DrawMeta:
    """A fully-populated DrawMeta with overridable fields."""
    fields: dict[str, Any] = {
        "z": 1.75,
        "coherence_z": 4.2,
        "coherence_valid": True,
        "coherence_r": 0.31,
        "purity_label": "quantum/intact/raw/qf:device",
        "integrated_bytes": 2_097_152,
        "integrator": "bit_z",
        "source_id": "qrng-a",
        "generation_timestamp_ns": 1_234_567,
        "echo_verified": None,
    }
    fields.update(overrides)
    return DrawMeta(**fields)


class FakeDrawSource(EntropySource):
    """Draw-capable in-memory source returning a scripted (u, meta)."""

    supports_server_draw: ClassVar[bool] = True

    def __init__(self, u: float = 0.734, meta: DrawMeta | None = None) -> None:
        self.u = u
        self.meta = meta if meta is not None else _meta()
        self.draw_calls: list[tuple[int, str, Any]] = []
        self.byte_calls: list[int] = []
        self._closed = False

    @property
    def name(self) -> str:
        return "fake_draw"

    @property
    def is_available(self) -> bool:
        return not self._closed

    def get_random_bytes(self, n: int) -> bytes:
        self.byte_calls.append(n)
        return os.urandom(n)

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: Any | None = None
    ) -> tuple[float, DrawMeta]:
        self.draw_calls.append((block_bytes, source_id, ticket))
        return self.u, self.meta

    def close(self) -> None:
        self._closed = True


class DrawlessSource(FakeDrawSource):
    """Draw-capable on paper, but the server never serves a draw."""

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: Any | None = None
    ) -> tuple[float, DrawMeta]:
        self.draw_calls.append((block_bytes, source_id, ticket))
        raise EntropyUnavailableError("PurityService down")


class FixedUAmplifier(SignalAmplifier):
    """Local amplifier producing a scripted u (differential-test twin)."""

    def __init__(self, u: float) -> None:
        self._u = u

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        return AmplificationResult(u=self._u, diagnostics={"sample_mean": 127.5, "z_score": 0.0})


class MetaRecordingStrategy:
    """Wraps a real strategy; records every observe_draw_meta call."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.observed: list[DrawMeta] = []

    def observe_draw_meta(self, meta: DrawMeta) -> None:
        self.observed.append(meta)

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> Any:
        return self._inner.compute_temperature(logits, config)


def _config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, sample_count=128, **overrides)  # type: ignore[call-arg]


def _pipeline(
    source: EntropySource,
    amplifier: SignalAmplifier,
    config: QRSamplerConfig,
    strategy: Any | None = None,
) -> SamplingPipeline:
    return SamplingPipeline(
        entropy_source=source,
        amplifier=amplifier,
        strategy=strategy
        if strategy is not None
        else (TemperatureStrategyRegistry.build(config, VOCAB)),
        selector=TokenSelector(),
        sampling_logger=SamplingLogger(config),
        config=config,
    )


def _logits() -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=VOCAB).astype(np.float32)


# ---------------------------------------------------------------------------
# EntropySource base defaults
# ---------------------------------------------------------------------------


class TestBaseDefaults:
    def test_supports_server_draw_defaults_false(self) -> None:
        assert EntropySource.supports_server_draw is False
        assert SystemEntropySource.supports_server_draw is False
        assert FakeDrawSource.supports_server_draw is True

    def test_default_get_draw_raises_entropy_unavailable(self) -> None:
        source = SystemEntropySource()
        with pytest.raises(EntropyUnavailableError, match="server-integrated draws"):
            source.get_draw(0, "")

    def test_default_prefetch_draw_returns_none(self) -> None:
        source = SystemEntropySource()
        assert source.prefetch_draw(0, "", nonce=42) is None


# ---------------------------------------------------------------------------
# FallbackEntropySource delegation
# ---------------------------------------------------------------------------


class TestFallbackDrawDelegation:
    def test_get_draw_delegates_to_primary(self) -> None:
        primary = FakeDrawSource(u=0.42)
        wrapper = FallbackEntropySource(primary, SystemEntropySource())
        u, meta = wrapper.get_draw(1024, "qrng-a", None)
        assert u == 0.42
        assert meta is primary.meta
        assert primary.draw_calls == [(1024, "qrng-a", None)]

    def test_draw_failure_raises_upward_without_failover_bookkeeping(self) -> None:
        wrapper = FallbackEntropySource(DrawlessSource(), SystemEntropySource())
        with pytest.raises(EntropyUnavailableError):
            wrapper.get_draw(0, "")
        # Failover bookkeeping means "who provided BYTES" — untouched.
        assert wrapper.fallback_count == 0
        assert wrapper.currently_degraded is False

    def test_prefetch_draw_delegates_and_never_raises(self) -> None:
        class _ExplodingDrawPrefetch(FakeDrawSource):
            def prefetch_draw(
                self, block_bytes: int, source_id: str, nonce: int | None = None
            ) -> Any | None:
                raise RuntimeError("boom")

        wrapper = FallbackEntropySource(_ExplodingDrawPrefetch(), SystemEntropySource())
        assert wrapper.prefetch_draw(0, "", nonce=7) is None
        # A primary without draw support yields None too (base default).
        plain = FallbackEntropySource(SystemEntropySource(), SystemEntropySource())
        assert plain.prefetch_draw(0, "") is None


# ---------------------------------------------------------------------------
# ServerDrawAmplifier
# ---------------------------------------------------------------------------


class TestServerDrawAmplifier:
    def test_registered_under_server(self) -> None:
        assert AmplifierRegistry.get("server") is ServerDrawAmplifier
        assert "server" in AmplifierRegistry.list_registered()

    def test_build_from_config(self) -> None:
        config = _config(signal_amplifier_type="server")
        amplifier = AmplifierRegistry.build(config)
        assert isinstance(amplifier, ServerDrawAmplifier)
        assert amplifier.requires_server_draw is True

    def test_local_amplify_raises(self) -> None:
        amplifier = ServerDrawAmplifier(_config())
        with pytest.raises(SignalAmplificationError, match="no local amplify path"):
            amplifier.amplify(b"\x00" * 16)

    def test_local_amplifiers_do_not_carry_the_flag(self) -> None:
        config = _config()
        local = AmplifierRegistry.build(config)
        assert getattr(local, "requires_server_draw", False) is False


# ---------------------------------------------------------------------------
# Pipeline draw branch
# ---------------------------------------------------------------------------


class TestPipelineDrawMode:
    def test_server_u_consumed_and_meta_on_result_and_record(self) -> None:
        meta = _meta()
        source = FakeDrawSource(u=0.734, meta=meta)
        config = _config(
            signal_amplifier_type="server",
            draw_source_id="qrng-a",
            draw_block_bytes=2_097_152,
        )
        pipeline = _pipeline(source, ServerDrawAmplifier(config), config)

        result = pipeline.sample_token(_logits(), build_onehot=False)

        # One draw round trip, no local byte fetch/amplification.
        assert source.draw_calls == [(2_097_152, "qrng-a", None)]
        assert source.byte_calls == []
        # DrawMeta rides the result AND the record.
        assert result.draw_meta is meta
        record = result.record
        assert record.u_value == 0.734
        assert record.z_score == meta.z
        assert math.isnan(record.sample_mean)  # no byte mean exists
        assert record.draw_z == meta.z
        assert record.draw_coherence_z == meta.coherence_z
        assert record.draw_coherence_valid is True
        assert record.draw_coherence_r == meta.coherence_r
        assert record.purity_label == meta.purity_label
        assert record.integrated_bytes == 2_097_152
        assert record.integrator == "bit_z"
        assert record.draw_source_id == "qrng-a"
        assert record.entropy_is_fallback is False

    def test_differential_selector_identical_to_local_amplifier_same_u(self) -> None:
        """The selector's output is byte-identical for a server draw and a
        local amplifier producing the same u — the draw path changes WHERE
        u comes from, never how it selects."""
        u = 0.61803398875
        logits = _logits()

        draw_config = _config(signal_amplifier_type="server")
        draw_pipeline = _pipeline(
            FakeDrawSource(u=u), ServerDrawAmplifier(draw_config), draw_config
        )
        local_config = _config()
        local_pipeline = _pipeline(FakeDrawSource(u=u), FixedUAmplifier(u), local_config)

        draw_result = draw_pipeline.sample_token(logits.copy())
        local_result = local_pipeline.sample_token(logits.copy())

        assert draw_result.token_id == local_result.token_id
        assert draw_result.record.token_rank == local_result.record.token_rank
        assert draw_result.record.token_prob == local_result.record.token_prob
        assert draw_result.record.num_candidates == local_result.record.num_candidates
        np.testing.assert_array_equal(draw_result.one_hot, local_result.one_hot)

    def test_observe_draw_meta_hook_invoked(self) -> None:
        meta = _meta()
        config = _config(signal_amplifier_type="server")
        strategy = MetaRecordingStrategy(TemperatureStrategyRegistry.build(config, VOCAB))
        pipeline = _pipeline(
            FakeDrawSource(meta=meta), ServerDrawAmplifier(config), config, strategy=strategy
        )

        pipeline.sample_token(_logits(), build_onehot=False)
        assert strategy.observed == [meta]

    def test_hook_receives_none_on_degraded_draw(self) -> None:
        # Degraded draw (no meta): a hook-bearing strategy is explicitly
        # signalled with None so it clears any stale evidence (review fix —
        # an outage must hard-reset a coherence gate, not replay old meta).
        config = _config(signal_amplifier_type="server")
        strategy = MetaRecordingStrategy(TemperatureStrategyRegistry.build(config, VOCAB))
        source = FallbackEntropySource(DrawlessSource(), SystemEntropySource())
        pipeline = _pipeline(source, ServerDrawAmplifier(config), config, strategy=strategy)
        pipeline.sample_token(_logits(), build_onehot=False)
        assert strategy.observed == [None]
        # Hook-less strategies simply aren't called (no AttributeError).
        plain = _pipeline(FakeDrawSource(), ServerDrawAmplifier(config), config)
        plain.sample_token(_logits(), build_onehot=False)

    def test_byte_path_leaves_draw_fields_none(self) -> None:
        config = _config()
        pipeline = _pipeline(FakeDrawSource(), FixedUAmplifier(0.5), config)
        record = pipeline.sample_token(_logits(), build_onehot=False).record
        assert record.draw_z is None
        assert record.draw_coherence_z is None
        assert record.purity_label is None
        assert record.integrated_bytes is None
        assert record.integrator is None
        assert record.draw_source_id is None
        assert record.gate_open is None
        assert record.gate_boost is None


# ---------------------------------------------------------------------------
# Degradation path
# ---------------------------------------------------------------------------


class TestDrawDegradation:
    def _degraded_pipeline(self) -> tuple[SamplingPipeline, DrawlessSource]:
        primary = DrawlessSource()
        source = FallbackEntropySource(primary, SystemEntropySource())
        config = _config(signal_amplifier_type="server")
        return _pipeline(source, ServerDrawAmplifier(config), config), primary

    def test_degrades_to_fallback_bytes_and_local_zscore(self) -> None:
        pipeline, primary = self._degraded_pipeline()
        result = pipeline.sample_token(_logits(), build_onehot=False)

        record = result.record
        assert result.draw_meta is None
        assert record.entropy_is_fallback is True
        assert record.draw_z is None
        # Local zscore_mean statistics — a real byte mean, not the NaN sentinel.
        assert not math.isnan(record.sample_mean)
        assert 0.0 < record.u_value < 1.0
        # The substitute bytes were fetched at sample_count through the wrapper.
        assert primary.draw_calls  # the draw WAS attempted first

    def test_degraded_local_amplifier_is_cached(self) -> None:
        pipeline, _ = self._degraded_pipeline()
        pipeline.sample_token(_logits(), build_onehot=False)
        first = pipeline._draw_fallback_amp
        assert first is not None
        pipeline.sample_token(_logits(), build_onehot=False)
        assert pipeline._draw_fallback_amp is first

    def test_fallback_mode_error_reraises(self) -> None:
        config = _config(signal_amplifier_type="server", fallback_mode="error")
        pipeline = _pipeline(DrawlessSource(), ServerDrawAmplifier(config), config)
        with pytest.raises(EntropyUnavailableError):
            pipeline.sample_token(_logits(), build_onehot=False)

    def test_bare_primary_bytes_still_degrade_as_fallback(self) -> None:
        """Even when the byte substitute comes from the draw source's own
        byte path (no wrapper), the token is marked degraded — the draw
        itself failed."""
        source = DrawlessSource()
        config = _config(signal_amplifier_type="server", fallback_mode="system")
        pipeline = _pipeline(source, ServerDrawAmplifier(config), config)
        record = pipeline.sample_token(_logits(), build_onehot=False).record
        assert record.entropy_is_fallback is True
        assert source.byte_calls == [128]
