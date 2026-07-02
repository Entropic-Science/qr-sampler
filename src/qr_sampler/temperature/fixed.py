"""Fixed temperature strategy.

Returns a constant temperature regardless of the logit distribution.
The simplest strategy -- useful as a baseline and when the user wants
direct control over sampling temperature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_shannon_entropy,
)

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.config import QRSamplerConfig


class FixedTemperatureStrategy(TemperatureStrategy):
    """Returns ``config.fixed_temperature`` for every token.

    Shannon entropy is always computed and included in the result
    for logging purposes, even though it does not affect the temperature.
    """

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute temperature (constant) and Shannon entropy.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing ``fixed_temperature``.

        Returns:
            TemperatureResult with the fixed temperature and Shannon entropy.
        """
        h = compute_shannon_entropy(logits)
        return TemperatureResult(
            temperature=config.fixed_temperature,
            shannon_entropy=h,
            diagnostics={"strategy": "fixed"},
        )
