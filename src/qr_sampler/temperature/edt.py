"""Entropy-based Dynamic Temperature (EDT) strategy.

Dynamically adjusts sampling temperature based on the Shannon entropy
of the logit distribution. Low-entropy (peaked) distributions get lower
temperature, while high-entropy (flat) distributions get higher temperature.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_shannon_entropy,
)

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.config import QRSamplerConfig


class EDTTemperatureStrategy(TemperatureStrategy):
    """Entropy-based Dynamic Temperature.

    Formula::

        H_norm = shannon_entropy / ln(vocab_size)   # Normalized to [0, 1]
        T = edt_base_temp * H_norm ^ edt_exponent
        T = clamp(T, edt_min_temp, edt_max_temp)

    Behaviour:
        - Low entropy (peaked) -> low T -> sharper sampling
        - High entropy (flat) -> high T -> more exploration
        - ``edt_exponent < 1`` (concave): T rises quickly with entropy
        - ``edt_exponent > 1`` (convex): T rises slowly with entropy

    The constructor requires ``vocab_size`` to compute the maximum possible
    entropy ``H_max = ln(V)``, which is used for normalization.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize with the model's vocabulary size.

        Args:
            vocab_size: Number of tokens in the model vocabulary.

        Raises:
            ValueError: If vocab_size < 2 (entropy normalization is undefined).
        """
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self._vocab_size = vocab_size
        self._max_entropy = math.log(vocab_size)  # H_max = ln(V)

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute dynamic temperature based on logit entropy.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing EDT parameters.

        Returns:
            TemperatureResult with the computed temperature and Shannon entropy.
        """
        h = compute_shannon_entropy(logits)

        # Normalize entropy to [0, 1].
        h_norm = h / self._max_entropy if self._max_entropy > 0 else 0.0

        # Power-law scaling.
        raw_temp = config.edt_base_temp * (h_norm**config.edt_exponent)

        # Clamp to configured bounds.
        temp = max(config.edt_min_temp, min(config.edt_max_temp, raw_temp))

        return TemperatureResult(
            temperature=temp,
            shannon_entropy=h,
            diagnostics={
                "strategy": "edt",
                "h_norm": h_norm,
                "pre_clamp_temp": raw_temp,
                "vocab_size": self._vocab_size,
            },
        )
