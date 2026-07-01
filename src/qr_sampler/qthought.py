"""Qthought roller — fresh quantum draws for arbitrary-arity grammar decisions.

The ``qthought`` "automated mind" consumes entropy in the same shape as the
:class:`~qr_sampler.contseq.ContseqRoller` (one full-size ``sample_count`` fetch
reduced through the configured amplifier to a uniform float ``u``), but where the
contseq roller maps every fetch onto a single 256-way byte code, the qthought
roller exposes a *typed family* of decisions that a case-frame grammar needs:

- :meth:`QthoughtRoller.choose` — uniform pick of one of ``k`` options.
- :meth:`QthoughtRoller.choose_weighted` — CDF pick over a weight vector.
- :meth:`QthoughtRoller.coin` — a biased Bernoulli (slot-presence gate).
- :meth:`QthoughtRoller.bind_int` — a piecewise-uniform integer bind
  (``time``/``age``/``year`` and other mixture domains).

Every decision performs exactly **one fresh** ``get_random_bytes(sample_count)``
→ ``amplify`` → ``u`` and records a :class:`ChoiceProvenance` into an internal
buffer, so the entropy is just-in-time and never cached across decisions
(invariant 4). :meth:`QthoughtRoller.drain` returns and clears that buffer.

Parity with token sampling is deliberate — like the contseq roller, this is the
entropy half of ``SamplingPipeline.sample_token`` with the logits half cut away,
so the grammar decisions and the GPU token samples draw from the same statistical
machinery and the same fallback/honesty labelling.

This module has zero vLLM/torch imports — it composes the entropy stack exactly
the way ``core/pipeline.py`` and ``contseq.py`` do. ``QthoughtRoller`` methods are
SYNC (the underlying gRPC fetch is blocking); async callers run them via
``asyncio.to_thread`` so their event loop never blocks on the network.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config
from qr_sampler.core.pipeline import build_entropy_source, config_hash
from qr_sampler.entropy.fallback import FallbackEntropySource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.entropy.base import EntropySource

logger = logging.getLogger("qr_sampler")


@dataclass(frozen=True)
class ChoiceProvenance:
    """One qthought decision plus the entropy provenance it was derived from.

    This is the :class:`~qr_sampler.contseq.RollResult` analogue, extended with
    the *decision context* the grammar needs to audit a decoded thought: which
    kind of decision was made (``kind``), what it resolved to (``value``), and
    the per-draw bias diagnostics (``z_score``, ``bias``) alongside the entropy
    honesty labels (``source``, ``is_fallback``) and freshness/latency stamps.
    """

    kind: str
    """Decision kind: ``'choose'`` | ``'choose_weighted'`` | ``'coin'`` | ``'bind_int'``."""

    value: int | bool
    """The decision result (an index/integer; a ``bool`` for :meth:`QthoughtRoller.coin`)."""

    u: float
    """Amplified uniform float in (eps, 1-eps) the decision was derived from."""

    z_score: float
    """Z-score of this draw's sample mean under the null (from amplifier diagnostics)."""

    bias: float
    """``sample_mean - population_mean`` for this draw — the raw measured bias."""

    source: str
    """Name of the entropy source that actually served the fetch."""

    is_fallback: bool
    """True when the fetch came from the fallback leg, not the primary."""

    generation_timestamp: float
    """Wall-clock (``time.time()``) instant the entropy for this decision was drawn."""

    latency_ms: float
    """Wall-clock duration of fetch + amplify + map for this decision."""

    thought_aggregate: dict[str, Any] | None = None
    """Optional thought-level z-score/bias aggregate, folded in at :meth:`QthoughtRoller.drain`.

    Populated only when the active amplifier exposes the duck-typed thought
    protocol (``zscore_thought``) and a thought scope was opened with
    :meth:`QthoughtRoller.begin_thought`; ``None`` for the fallback-safe
    ``zscore_mean`` path, which omits it entirely.
    """


@dataclass(frozen=True)
class IntRange:
    """One inclusive integer interval ``[low, high]`` with a selection weight."""

    low: int
    high: int
    weight: float = 1.0


@dataclass(frozen=True)
class BindSpec:
    """A piecewise-uniform integer domain: a weighted mixture of inclusive ranges.

    :meth:`QthoughtRoller.bind_int` maps one fresh amplified uniform onto this
    domain by inverse-CDF: the uniform first selects a component range in
    proportion to its weight, then positions uniformly within that range. A
    single range is the degenerate (plain-uniform) case.

    The ``mode`` label is carried for documentation/telemetry only; the three
    constructors below build the common semantic domains a thought grammar binds
    (a clock time, a human age, a calendar year) as small, defensible mixtures.
    """

    mode: str
    ranges: tuple[IntRange, ...]

    @classmethod
    def for_time(cls) -> BindSpec:
        """Hour-of-day ``[0, 23]``, weighted toward waking hours."""
        return cls(
            mode="time",
            ranges=(
                IntRange(0, 5, 0.5),
                IntRange(6, 11, 1.5),
                IntRange(12, 17, 1.5),
                IntRange(18, 23, 1.0),
            ),
        )

    @classmethod
    def for_age(cls) -> BindSpec:
        """Human age ``[0, 99]``, weighted toward adults."""
        return cls(
            mode="age",
            ranges=(
                IntRange(0, 17, 1.0),
                IntRange(18, 64, 3.0),
                IntRange(65, 99, 1.0),
            ),
        )

    @classmethod
    def for_year(cls, *, low: int = 1900, high: int = 2099) -> BindSpec:
        """Calendar year, weighted toward the most recent century."""
        midpoint = (low + high) // 2
        return cls(
            mode="year",
            ranges=(
                IntRange(low, midpoint, 1.0),
                IntRange(midpoint + 1, high, 2.0),
            ),
        )


@dataclass(frozen=True)
class _Draw:
    """One fresh entropy draw reduced to a uniform plus its provenance fields."""

    u: float
    z_score: float
    bias: float
    source: str
    is_fallback: bool
    generation_timestamp: float
    latency_ms: float


def _ensure_source_importable(config: QRSamplerConfig) -> QRSamplerConfig:
    """Trigger the lazy import that registers ``quantum_grpc``, or degrade.

    ``EntropySourceRegistry`` is populated by module-import side effects:
    ``qr_sampler.entropy.quantum`` self-registers via decorator, but
    ``qr_sampler.entropy.__init__`` deliberately does NOT import it so
    grpcio stays an optional dependency. The vLLM containers import it
    through the serving stack; a host that only runs the qthought roller
    does not — without this nudge, building a roller there dies with
    ``KeyError: Unknown entropy source: 'quantum_grpc'``.

    If the import itself fails (grpcio genuinely absent), swap the
    config to the fallback source instead of crashing: an automated mind
    on labeled system entropy beats no mind at all, and the
    ``is_fallback``/source labels keep the degradation honest downstream.
    """
    if config.entropy_source_type != "quantum_grpc":
        return config
    try:
        import qr_sampler.entropy.quantum  # noqa: F401 — registers "quantum_grpc"
    except ImportError as exc:
        degraded = config.fallback_mode if config.fallback_mode != "error" else "system"
        logger.warning(
            "qthought.entropy.import_failed: quantum_grpc unavailable (%s); "
            "degrading to %s entropy",
            exc,
            degraded,
            extra={"event": "qthought.entropy.import_failed", "degraded_to": degraded},
        )
        return config.model_copy(update={"entropy_source_type": degraded})
    return config


class QthoughtRoller:
    """Rolls typed grammar decisions from the entropy stack, token-sampling style.

    Construction without a config resolves the ``qthought`` preset on top of the
    environment defaults (pinning ``quantum_grpc`` + ``zscore_thought``), so the
    lineage shows up in logs via ``config_hash``. Passing an explicit config
    skips preset resolution — tests use this to run against ``mock_uniform``.

    Every public decision method performs exactly one fresh fetch + amplify and
    records a :class:`ChoiceProvenance`; :meth:`drain` hands the buffer back to
    the caller and clears it. Methods are sync (the underlying gRPC path is
    blocking); async callers wrap them in ``asyncio.to_thread``.
    """

    def __init__(self, config: QRSamplerConfig | None = None) -> None:
        """Build the entropy source + amplifier from config.

        Args:
            config: Explicit sampler configuration. ``None`` loads the
                environment defaults and applies the ``qthought`` preset.
        """
        if config is None:
            config = resolve_config(QRSamplerConfig(preset="qthought"), None)
        config = _ensure_source_importable(config)
        self._config = config
        self._config_hash = config_hash(config)
        self._source: EntropySource = build_entropy_source(config)
        self._amplifier: SignalAmplifier = AmplifierRegistry.build(config)
        if hasattr(self._amplifier, "calibrate"):
            self._amplifier.calibrate(self._source, config)
        self._buffer: list[ChoiceProvenance] = []
        self._thought_active = False
        logger.info(
            "qthought.roller.init: source=%s amplifier=%s sample_count=%d config_hash=%s",
            self._source.name,
            config.signal_amplifier_type,
            config.sample_count,
            self._config_hash,
            extra={
                "event": "qthought.roller.init",
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

    def begin_thought(self) -> None:
        """Open a thought scope, resetting the amplifier's thought accumulator.

        When the active amplifier exposes the optional, duck-typed thought
        protocol (``zscore_thought``), this resets its thought-scoped byte
        accumulator so the next sequence of decisions folds into one
        thought-level aggregate, surfaced by :meth:`drain`. With the
        fallback-safe ``zscore_mean`` amplifier the protocol is absent and this
        is a no-op beyond marking the scope open (the aggregate is omitted).
        """
        if hasattr(self._amplifier, "begin_thought"):
            self._amplifier.begin_thought()
        self._thought_active = True

    def choose(self, k: int) -> int:
        """Uniformly pick one of ``k`` options as an index in ``[0, k-1]``.

        Args:
            k: Number of options (must be >= 1).

        Returns:
            ``min(int(u * k), k - 1)`` — the amplified-uniform index.
        """
        if k < 1:
            raise ValueError(f"choose(k) requires k >= 1, got {k}")
        draw = self._draw()
        value = min(int(draw.u * k), k - 1)
        self._record("choose", value, draw)
        return value

    def choose_weighted(self, weights: Sequence[float]) -> int:
        """Pick an index by inverse-CDF over a (non-negative) weight vector.

        Args:
            weights: One weight per option; must be non-empty, non-negative,
                and sum to a positive total.

        Returns:
            The selected index in ``[0, len(weights) - 1]``.
        """
        if not weights:
            raise ValueError("choose_weighted requires at least one weight")
        if any(w < 0.0 for w in weights):
            raise ValueError("choose_weighted requires non-negative weights")
        total = math.fsum(weights)
        if total <= 0.0:
            raise ValueError("choose_weighted requires a positive total weight")

        draw = self._draw()
        target = draw.u * total
        cumulative = 0.0
        index = len(weights) - 1
        for i, weight in enumerate(weights):
            cumulative += weight
            if target < cumulative:
                index = i
                break
        self._record("choose_weighted", index, draw)
        return index

    def coin(self, p: float) -> bool:
        """Flip a biased coin: ``True`` with probability ``p`` (slot-presence gate).

        Args:
            p: Success probability in ``[0, 1]``.

        Returns:
            ``u < p``.
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"coin(p) requires 0 <= p <= 1, got {p}")
        draw = self._draw()
        value = draw.u < p
        self._record("coin", value, draw)
        return value

    def bind_int(self, spec: BindSpec) -> int:
        """Bind one fresh uniform onto a piecewise-uniform integer domain.

        The uniform selects a component range in proportion to its weight
        (inverse-CDF over the mixture), then positions uniformly within that
        range. Single-range specs reduce to a plain uniform bind.

        Args:
            spec: The mixture of inclusive integer ranges to bind into.

        Returns:
            An integer inside one of ``spec``'s ranges.
        """
        if not spec.ranges:
            raise ValueError("bind_int requires at least one range")
        for r in spec.ranges:
            if r.high < r.low:
                raise ValueError(f"IntRange high {r.high} < low {r.low}")
            if r.weight <= 0.0:
                raise ValueError(f"IntRange weight must be positive, got {r.weight}")

        total = math.fsum(r.weight for r in spec.ranges)
        draw = self._draw()
        target = draw.u * total

        cumulative = 0.0
        chosen = spec.ranges[-1]
        component_lo = total - spec.ranges[-1].weight
        for r in spec.ranges:
            nxt = cumulative + r.weight
            if target < nxt:
                chosen = r
                component_lo = cumulative
                break
            cumulative = nxt

        span = chosen.high - chosen.low + 1
        residual = (target - component_lo) / chosen.weight
        offset = min(int(residual * span), span - 1)
        value = chosen.low + offset
        self._record("bind_int", value, draw)
        return value

    def drain(self) -> tuple[ChoiceProvenance, ...]:
        """Return and clear the buffered per-decision provenance.

        When a thought scope is open (:meth:`begin_thought`) and the amplifier
        exposes the thought protocol, the thought-level aggregate is folded into
        every returned provenance so the caller can audit the decoded thought's
        aggregate bias alongside the per-decision draws. The scope is closed on
        drain.

        Returns:
            The buffered :class:`ChoiceProvenance` entries, oldest first.
        """
        items = tuple(self._buffer)
        self._buffer.clear()
        if self._thought_active and hasattr(self._amplifier, "thought_aggregate"):
            aggregate: dict[str, Any] = dict(self._amplifier.thought_aggregate())
            items = tuple(replace(item, thought_aggregate=aggregate) for item in items)
        self._thought_active = False
        return items

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

    def _draw(self) -> _Draw:
        """Perform one fresh fetch + amplify and capture its provenance fields.

        Exactly the entropy half of a token-sampling step:
        ``get_random_bytes(sample_count)`` → ``amplify()`` → ``u``. The fallback
        source name + flag are read immediately after the fetch (the wrapper
        records the leg it just used), mirroring ``SamplingPipeline.sample_token``
        and ``ContseqRoller.roll``.
        """
        t_start = time.perf_counter_ns()
        raw = self._source.get_random_bytes(self._config.sample_count)

        source_name = self._source.name
        is_fallback = False
        if isinstance(self._source, FallbackEntropySource):
            source_name = self._source.last_source_used
            is_fallback = source_name != self._source.primary_name

        result = self._amplifier.amplify(raw)
        sample_mean = float(result.diagnostics.get("sample_mean", self._config.population_mean))
        z_score = float(result.diagnostics.get("z_score", 0.0))
        latency_ms = (time.perf_counter_ns() - t_start) / 1_000_000.0

        return _Draw(
            u=result.u,
            z_score=z_score,
            bias=sample_mean - self._config.population_mean,
            source=source_name,
            is_fallback=is_fallback,
            generation_timestamp=time.time(),
            latency_ms=latency_ms,
        )

    def _record(self, kind: str, value: int | bool, draw: _Draw) -> None:
        """Append one :class:`ChoiceProvenance` for a completed decision."""
        self._buffer.append(
            ChoiceProvenance(
                kind=kind,
                value=value,
                u=draw.u,
                z_score=draw.z_score,
                bias=draw.bias,
                source=draw.source,
                is_fallback=draw.is_fallback,
                generation_timestamp=draw.generation_timestamp,
                latency_ms=draw.latency_ms,
            )
        )
