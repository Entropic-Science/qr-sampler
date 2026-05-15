"""Temperature strategy subsystem for qr-sampler.

Computes per-token sampling temperature from the logit distribution.
Supports fixed and entropy-based dynamic temperature (EDT) strategies.
"""

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_shannon_entropy,
)
from qr_sampler.temperature.edt import EDTTemperatureStrategy
from qr_sampler.temperature.fixed import FixedTemperatureStrategy
from qr_sampler.temperature.hvh_drift import HVHDriftStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

__all__ = [
    "EDTTemperatureStrategy",
    "FixedTemperatureStrategy",
    "HVHDriftStrategy",
    "TemperatureResult",
    "TemperatureStrategy",
    "TemperatureStrategyRegistry",
    "compute_shannon_entropy",
]
