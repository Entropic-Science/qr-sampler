"""Z-score mean signal amplifier.

Converts raw entropy bytes into a uniform float via z-score statistics.
Under the null hypothesis (unbiased entropy), the output is uniformly
distributed on (0, 1). Any systematic bias in the entropy source shifts
the output away from 0.5, enabling weak-signal detection.

Baseline calibration (``zscore_calibration_samples > 0``): a real device's
byte mean is never exactly the ideal 127.5, and at ``sample_count`` ~ 10^4
the SEM is small enough (~0.7) that even a fraction-of-a-byte static offset
saturates every z into the CDF clamp — every ``u`` pins to the same extreme
and every downstream ``choose(k)`` returns the same index. Calibration draws
N blocks from the *actual* source at build time and replaces the population
baseline with the device's empirical block-mean / block-SEM, so ``u`` is
uniform under the device's own null and only *departures from the device's
baseline* register as signal. Baseline correction, not censoring — the same
rationale as the server-integrated draw path (F7.2).
"""

from __future__ import annotations

import logging
import math
import weakref
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.exceptions import SignalAmplificationError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

logger = logging.getLogger("qr_sampler")

_SQRT2 = math.sqrt(2.0)

#: Process-wide calibration cache: per source *instance*, keyed by the
#: (sample_count, calibration_samples) pair, holding the derived
#: (baseline_mean, byte_std). Per-request amplifier builds (the vLLM adapter
#: constructs one amplifier per request config) would otherwise re-pay the
#: full N-block calibration fetch on every request against the same source.
#: Weak keys: the cache never extends an entropy source's lifetime.
_CALIBRATION_CACHE: weakref.WeakKeyDictionary[object, dict[tuple[int, int], tuple[float, float]]]
_CALIBRATION_CACHE = weakref.WeakKeyDictionary()


class ZScoreMeanAmplifier(SignalAmplifier):
    """Z-score signal amplification.

    Algorithm:
        1. Interpret raw_bytes as uint8 array.
        2. Compute sample mean M.
        3. Derive SEM = population_std / sqrt(N).
        4. Compute z-score: z = (M - population_mean) / SEM.
        5. Map to uniform via normal CDF: u = 0.5 * (1 + erf(z / sqrt(2))).
        6. Clamp to (eps, 1-eps).

    Under the null hypothesis (no systematic bias), z ~ N(0, 1)
    and u ~ Uniform(0, 1). A small per-byte bias (e.g., +0.003) accumulates
    over thousands of samples, producing a detectable shift in u.

    Example with 20,480 bytes and +0.003 mean shift per byte:
        M ~ 127.56, SEM ~ 0.5143, z ~ 0.12, u ~ 0.548
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """Initialize with population parameters from config.

        Args:
            config: Configuration providing population_mean, population_std,
                and uniform_clamp_epsilon.
        """
        self._population_mean = config.population_mean
        self._population_std = config.population_std
        self._clamp_epsilon = config.uniform_clamp_epsilon

    def calibrate(
        self,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> None:
        """Replace the ideal-population baseline with the source's empirical one.

        Gated on ``config.zscore_calibration_samples``: 0 (the default) is a
        no-op, keeping the historical ideal baseline byte-identical. N > 0
        draws N blocks of ``config.sample_count`` bytes, then sets

        * ``population_mean`` ← the grand mean of the block means (the
          device's real baseline), and
        * ``population_std`` ← the *effective* per-byte std derived from the
          observed block-mean spread (``std(means, ddof=1) * sqrt(n)``), so
          the derived SEM matches the device's true block-mean variance even
          when its bytes are not i.i.d. ideal-uniform.

        ``amplify()`` is untouched — SEM stays derived, never stored, and a
        given byte buffer still always maps to the same ``u`` after
        calibration completes (per-call statelessness is preserved).

        Results are cached per source *instance* (weakly) so per-request
        amplifier builds against a shared source calibrate once per process.

        Calibration draws through whatever leg the source serves at build
        time; if a fallback wrapper is degraded to the system leg the learned
        baseline is the system's (≈ ideal), which is exactly the historical
        behaviour — never worse than uncalibrated.

        Args:
            entropy_source: Source to draw calibration blocks from.
            config: Configuration providing zscore_calibration_samples and
                sample_count.

        Raises:
            SignalAmplificationError: If the calibration blocks have zero
                variance (a stuck source cannot define a baseline).
        """
        n_blocks = config.zscore_calibration_samples
        if n_blocks <= 0:
            return

        cache_key = (config.sample_count, n_blocks)
        per_source = _CALIBRATION_CACHE.setdefault(entropy_source, {})
        cached = per_source.get(cache_key)
        if cached is not None:
            self._population_mean, self._population_std = cached
            return

        means = np.empty(n_blocks, dtype=np.float64)
        for i in range(n_blocks):
            raw = entropy_source.get_random_bytes(config.sample_count)
            means[i] = float(np.frombuffer(raw, dtype=np.uint8).mean())

        block_sem = float(np.std(means, ddof=1)) if n_blocks > 1 else 0.0
        if block_sem == 0.0:
            raise SignalAmplificationError(
                "z-score calibration produced zero block-mean variance — "
                "the source is stuck; cannot define an empirical baseline"
            )

        baseline_mean = float(np.mean(means))
        byte_std = block_sem * math.sqrt(config.sample_count)
        self._population_mean = baseline_mean
        self._population_std = byte_std
        per_source[cache_key] = (baseline_mean, byte_std)
        logger.info(
            "z-score calibration complete: %d blocks x %d bytes, "
            "baseline_mean=%.4f (ideal 127.5, offset %+0.4f), "
            "effective byte_std=%.3f (ideal %.3f)",
            n_blocks,
            config.sample_count,
            baseline_mean,
            baseline_mean - 127.5,
            byte_std,
            config.population_std,
        )

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw entropy bytes into a uniform float.

        Args:
            raw_bytes: Raw entropy bytes from an entropy source.

        Returns:
            AmplificationResult with u in (eps, 1-eps) and diagnostics.

        Raises:
            SignalAmplificationError: If raw_bytes is empty.
        """
        if not raw_bytes:
            raise SignalAmplificationError("Cannot amplify empty byte sequence")

        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(samples)
        sample_mean = float(np.mean(samples))

        # SEM is derived, never stored (invariant 5).
        sem = self._population_std / math.sqrt(n)
        z_score = (sample_mean - self._population_mean) / sem

        # Normal CDF via error function: phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
        u = 0.5 * (1.0 + math.erf(z_score / _SQRT2))

        # Clamp to avoid degenerate CDF extremes.
        eps = self._clamp_epsilon
        u = max(eps, min(1.0 - eps, u))

        return AmplificationResult(
            u=u,
            diagnostics={
                "sample_mean": sample_mean,
                "z_score": z_score,
                "sem": sem,
                "sample_count": n,
            },
        )
