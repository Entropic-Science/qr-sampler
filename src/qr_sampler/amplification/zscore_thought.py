"""Z-score thought-level signal amplifier.

Per-token sampling is **byte-identical** to :class:`ZScoreMeanAmplifier`: each
``amplify()`` call maps one buffer of raw entropy bytes to a uniform float via
exactly the same z-score statistics. This amplifier adds an *optional,
duck-typed* thought-level protocol on top of that unchanged per-token path:

    begin_thought()      reset a thought-scoped byte-statistics accumulator
    amplify(raw_bytes)   (unchanged output) + fold this call's bytes in
    thought_aggregate()  report one thought-level z-score / bias / uniform

The protocol is **not** part of the :class:`SignalAmplifier` ABC. Consumers
feature-detect it with ``hasattr(amplifier, "begin_thought")`` — mirroring the
existing ``hasattr(amplifier, "calibrate")`` precedent for the ECDF amplifier —
so the shipped ``zscore_mean`` path is unaffected and costs nothing when the
thought protocol is unused.

The accumulator is a pure side-channel: it never influences the value returned
by ``amplify()``, so per-call statelessness of the amplified signal (invariant
5) is preserved. A given byte buffer always amplifies to the same ``u``,
independent of how many calls preceded it within a thought.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier

if TYPE_CHECKING:
    from qr_sampler.amplification.base import AmplificationResult
    from qr_sampler.config import QRSamplerConfig

_SQRT2 = math.sqrt(2.0)


@AmplifierRegistry.register("zscore_thought")
class ZScoreThoughtAmplifier(ZScoreMeanAmplifier):
    """Z-score amplifier with an optional thought-level bias aggregate.

    Per-token amplification is inherited verbatim from
    :class:`ZScoreMeanAmplifier` — same raw ``uint8`` interpretation, same
    ``z = (M - population_mean) / SEM``, same ``erf``-based CDF mapping, same
    ``(eps, 1-eps)`` clamp, same diagnostics key set. The only addition is a
    thought-scoped accumulator folded by ``amplify()`` and summarised by
    ``thought_aggregate()``.

    The thought protocol is duck-typed (``begin_thought`` / ``thought_aggregate``
    are absent from the ABC and from ``ZScoreMeanAmplifier`` / ``ECDFAmplifier``),
    so it integrates one thought-level signal without changing the shipped
    per-token path.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """Initialize population parameters and an empty thought accumulator.

        Args:
            config: Configuration providing population_mean, population_std,
                and uniform_clamp_epsilon.
        """
        super().__init__(config)
        # Thought-scoped byte-statistics accumulator (a side-channel that never
        # affects amplify()'s return value). Tracked as an exact integer running
        # sum and count so the thought-level mean is computed without drift.
        self._thought_sum: int = 0
        self._thought_count: int = 0

    def begin_thought(self) -> None:
        """Reset the thought-scoped accumulator to start a fresh thought."""
        self._thought_sum = 0
        self._thought_count = 0

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Amplify one buffer (byte-identical to ZScoreMeanAmplifier) and fold.

        The returned result is exactly what :meth:`ZScoreMeanAmplifier.amplify`
        produces — the thought accumulator is updated only as a side effect and
        never participates in computing ``u`` (invariant 5).

        Args:
            raw_bytes: Raw entropy bytes from an entropy source.

        Returns:
            AmplificationResult with u in (eps, 1-eps) and diagnostics.

        Raises:
            SignalAmplificationError: If raw_bytes is empty. (Nothing is folded
                when the parent raises, since the raise precedes the fold.)
        """
        result = super().amplify(raw_bytes)

        # Fold this call's exact byte statistics into the thought accumulator.
        # uint8 sums auto-promote to a wide platform integer, so no overflow.
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        self._thought_sum += int(np.sum(samples))
        self._thought_count += len(samples)

        return result

    def thought_aggregate(self) -> dict[str, Any]:
        """Summarise the folded thought as one z-score / bias / uniform.

        Computes the same z-score statistics as a single ``amplify()`` call, but
        over *all* bytes folded since the last :meth:`begin_thought` — yielding a
        thought-level view of any aggregate bias. When no bytes have been folded
        the aggregate is neutral (``z_score`` 0, ``u`` 0.5).

        Returns:
            Mapping with keys ``sample_mean``, ``z_score``, ``sem``,
            ``sample_count``, ``bias`` (sample_mean - population_mean), and ``u``
            (the thought-level uniform in (eps, 1-eps)).
        """
        n = self._thought_count
        if n == 0:
            return {
                "sample_mean": float(self._population_mean),
                "z_score": 0.0,
                "sem": 0.0,
                "sample_count": 0,
                "bias": 0.0,
                "u": 0.5,
            }

        sample_mean = self._thought_sum / n
        bias = sample_mean - self._population_mean

        # SEM is derived, never stored (invariant 5).
        sem = self._population_std / math.sqrt(n)
        z_score = bias / sem

        u = 0.5 * (1.0 + math.erf(z_score / _SQRT2))
        eps = self._clamp_epsilon
        u = max(eps, min(1.0 - eps, u))

        return {
            "sample_mean": sample_mean,
            "z_score": z_score,
            "sem": sem,
            "sample_count": n,
            "bias": bias,
            "u": u,
        }
