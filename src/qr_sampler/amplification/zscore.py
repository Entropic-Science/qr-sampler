"""Z-score mean signal amplifier.

Converts raw entropy bytes into a uniform float via z-score statistics.
Under the null hypothesis (unbiased entropy), the output is uniformly
distributed on (0, 1). Any systematic bias in the entropy source shifts
the output away from 0.5, enabling weak-signal detection.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.exceptions import SignalAmplificationError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

_SQRT2 = math.sqrt(2.0)


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
