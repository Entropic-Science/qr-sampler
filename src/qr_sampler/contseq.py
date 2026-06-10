"""Contseq roller — one quantum byte per roll, for the thought engine.

The contseq "thought engine" (qr-llm-chat) consumes entropy in the same
shape as a token-sampling step: one full-size fetch (``sample_count``
bytes, the same ``QR_SAMPLE_COUNT`` the GPU lanes read), reduced through
the configured amplifier to a uniform float, then mapped to a byte code
in [0, 255]. Parity with token sampling is deliberate — the roller is
the entropy half of ``SamplingPipeline.sample_token`` with the logits
half cut away, so the "thought" rolls and the token samples draw from
the same statistical machinery.

Two rolls happen per engine tick (one for the word, one for the
action); the ``contseq`` preset documents that consumption pattern.

This module has zero vLLM/torch imports — it composes the entropy
stack exactly the way ``core/pipeline.py`` does.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config
from qr_sampler.core.pipeline import build_entropy_source, config_hash
from qr_sampler.entropy.fallback import FallbackEntropySource

if TYPE_CHECKING:
    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.entropy.base import EntropySource

logger = logging.getLogger("qr_sampler")


@dataclass(frozen=True)
class RollResult:
    """One contseq roll: a byte code plus its entropy provenance."""

    code: int
    """Selected code in [0, 255]."""

    u: float
    """Amplified uniform float the code was derived from, for logs."""

    source: str
    """Name of the entropy source that actually served the fetch."""

    is_fallback: bool
    """True when the fetch came from the fallback leg, not the primary."""

    latency_ms: float
    """Wall-clock duration of fetch + amplify + map."""


class ContseqRoller:
    """Rolls byte codes from the entropy stack, token-sampling style.

    Construction without a config resolves the ``contseq`` preset on top
    of the environment defaults (pinning ``quantum_grpc`` + ``zscore_mean``),
    so the lineage shows up in logs via ``config_hash``. Passing an
    explicit config skips preset resolution — tests use this to run
    against ``mock_uniform``.

    ``roll()`` is sync (the underlying ``_fetch_sync`` gRPC path); async
    callers run it via ``asyncio.to_thread`` so their event loop never
    blocks on the network.
    """

    def __init__(self, config: QRSamplerConfig | None = None) -> None:
        """Build the entropy source + amplifier from config.

        Args:
            config: Explicit sampler configuration. ``None`` loads the
                environment defaults and applies the ``contseq`` preset.
        """
        if config is None:
            config = resolve_config(QRSamplerConfig(preset="contseq"), None)
        self._config = config
        self._config_hash = config_hash(config)
        self._source: EntropySource = build_entropy_source(config)
        self._amplifier: SignalAmplifier = AmplifierRegistry.build(config)
        if hasattr(self._amplifier, "calibrate"):
            self._amplifier.calibrate(self._source, config)
        logger.info(
            "contseq.roller.init: source=%s amplifier=%s sample_count=%d config_hash=%s",
            self._source.name,
            config.signal_amplifier_type,
            config.sample_count,
            self._config_hash,
            extra={
                "event": "contseq.roller.init",
                "source": self._source.name,
                "config_hash": self._config_hash,
            },
        )

    @property
    def config(self) -> QRSamplerConfig:
        """The resolved configuration this roller was built with."""
        return self._config

    def warmup(self) -> None:
        """Eagerly open the entropy source's connection (gRPC channel)."""
        self._source.warmup()

    def roll(self) -> RollResult:
        """Fetch one full-size entropy sample and reduce it to a byte code.

        Exactly the entropy half of a token-sampling step:
        ``get_random_bytes(sample_count)`` -> ``amplify()`` ->
        ``code = min(int(u * 256), 255)``.

        Returns:
            RollResult with the code, the uniform it came from, and
            which source leg served the fetch.

        Raises:
            EntropyUnavailableError: Only when both primary and fallback
                fail (or ``fallback_mode='error'`` disables the wrapper).
        """
        t_start = time.perf_counter_ns()
        raw = self._source.get_random_bytes(self._config.sample_count)

        source_name = self._source.name
        is_fallback = False
        if isinstance(self._source, FallbackEntropySource):
            source_name = self._source.last_source_used
            is_fallback = source_name != self._source.primary_name

        u = self._amplifier.amplify(raw).u
        code = min(int(u * 256), 255)
        latency_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        return RollResult(
            code=code,
            u=u,
            source=source_name,
            is_fallback=is_fallback,
            latency_ms=latency_ms,
        )

    def status(self) -> dict[str, Any]:
        """Degradation telemetry for the engine's entropy honesty block.

        Returns:
            Dict with ``source``, ``currently_degraded`` and
            ``fallback_count``. A roller built with ``fallback_mode='error'``
            has no wrapper to ask, so it reports never-degraded.
        """
        if isinstance(self._source, FallbackEntropySource):
            return {
                "source": self._source.name,
                "currently_degraded": self._source.currently_degraded,
                "fallback_count": self._source.fallback_count,
            }
        return {
            "source": self._source.name,
            "currently_degraded": False,
            "fallback_count": 0,
        }

    def close(self) -> None:
        """Release entropy source resources."""
        self._source.close()
