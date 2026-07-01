"""Signal amplification subsystem for qr-sampler.

Converts raw entropy bytes into a uniform float u in (eps, 1-eps) that drives
CDF-based token selection. The amplification preserves even tiny biases
in the entropy source.
"""

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.ecdf import ECDFAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier
from qr_sampler.amplification.zscore_thought import ZScoreThoughtAmplifier

__all__ = [
    "AmplificationResult",
    "AmplifierRegistry",
    "ECDFAmplifier",
    "SignalAmplifier",
    "ZScoreMeanAmplifier",
    "ZScoreThoughtAmplifier",
]
