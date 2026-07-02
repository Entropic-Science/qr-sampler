"""ECDF-based signal amplifier.

Converts raw entropy bytes into a uniform float via an empirical cumulative
distribution function (ECDF) built from calibration samples. Unlike the
z-score amplifier, this approach makes no distributional assumptions about
the entropy source — it learns the distribution empirically.

The calibration phase collects N samples from the entropy source, computes
the byte-mean of each, and sorts them. At runtime, the sample mean is
mapped to a uniform float via binary search (Hazen plotting position).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.exceptions import SignalAmplificationError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

logger = logging.getLogger("qr_sampler")


class ECDFAmplifier(SignalAmplifier):
    """ECDF-based signal amplification.

    Algorithm:
        Calibration (one-time):
            1. Collect N samples of ``sample_count`` bytes each from the
               entropy source.
            2. Compute the byte-mean of each sample.
            3. Sort the means to form the empirical CDF.

        Runtime (per token):
            1. Interpret raw_bytes as uint8 array, compute sample mean M.
            2. Binary search M in the sorted calibration means.
            3. Map to uniform via Hazen plotting position:
               u = (rank + 1) / (N + 1).
            4. Clamp to (eps, 1-eps).

    The Hazen formula guarantees u in (0, 1) for any input, avoiding
    degenerate CDF extremes. No distributional assumptions are required.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """Initialize with calibration parameters from config.

        Args:
            config: Configuration providing ecdf_calibration_samples,
                sample_count, and uniform_clamp_epsilon.
        """
        self._ecdf_calibration_samples = config.ecdf_calibration_samples
        self._sample_count = config.sample_count
        self._clamp_epsilon = config.uniform_clamp_epsilon
        self._sorted_means: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
        self._calibrated: bool = False

    def calibrate(
        self,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> None:
        """Build the empirical CDF from calibration samples.

        Collects ``ecdf_calibration_samples`` samples from the entropy source,
        computes the byte-mean of each, and sorts them. Calibration is
        idempotent — calling again replaces the sorted array.

        Args:
            entropy_source: Source to draw calibration bytes from.
            config: Configuration providing sample_count.

        Raises:
            SignalAmplificationError: If all calibration samples are identical
                (zero variance).
        """
        n = self._ecdf_calibration_samples
        means: list[float] = []
        for _ in range(n):
            raw = entropy_source.get_random_bytes(config.sample_count)
            sample_mean = float(np.frombuffer(raw, dtype=np.uint8).mean())
            means.append(sample_mean)

        self._sorted_means = np.sort(np.array(means))

        if np.std(self._sorted_means) == 0.0:
            raise SignalAmplificationError(
                "ECDF calibration produced zero variance — all samples identical"
            )

        self._calibrated = True
        logger.info(
            "ECDF calibration complete: %d samples, mean range [%.2f, %.2f]",
            n,
            float(self._sorted_means[0]),
            float(self._sorted_means[-1]),
        )

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Convert raw entropy bytes into a uniform float via ECDF lookup.

        Args:
            raw_bytes: Raw entropy bytes from an entropy source.

        Returns:
            AmplificationResult with u in (eps, 1-eps) and diagnostics.

        Raises:
            SignalAmplificationError: If not calibrated or raw_bytes is empty.
        """
        if not self._calibrated:
            raise SignalAmplificationError(
                "ECDF amplifier has not been calibrated. Call calibrate() first."
            )
        if not raw_bytes:
            raise SignalAmplificationError("Cannot amplify empty byte sequence")

        sample_mean = float(np.frombuffer(raw_bytes, dtype=np.uint8).mean())

        # Binary search in the sorted calibration means.
        sorted_means = self._sorted_means
        if sorted_means is None:
            raise SignalAmplificationError("Calibration data missing")
        rank = int(np.searchsorted(sorted_means, sample_mean, side="right"))
        n = len(sorted_means)

        # Hazen plotting position: u = (rank + 1) / (N + 1).
        u = (rank + 1) / (n + 1)

        # Clamp to avoid degenerate CDF extremes.
        eps = self._clamp_epsilon
        u = max(eps, min(1.0 - eps, u))

        return AmplificationResult(
            u=u,
            diagnostics={
                "sample_mean": sample_mean,
                "ecdf_rank": rank,
                "calibration_size": n,
            },
        )
