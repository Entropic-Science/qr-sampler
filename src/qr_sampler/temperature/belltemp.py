"""BellTemp: bell-curve temperature on normalized entropy.

Stateless per-token strategy ported from createmp-evalsuite's V5
``BellTempProcessor`` (v4v6 competitiveness assessment §3.5). Unlike
monotonic methods (EDT, DynaTemp) the temperature-entropy relationship is
non-monotonic — sharpen when confident, explore in the mid-entropy sweet
spot, stabilize when the model is lost:

    H_norm  = clip(H / ln(V), 0, 1)
    VH_norm = 1 - exp(-VH / belltemp_lambda_vh)
    bell    = belltemp_t_peak * exp(-(H_norm - belltemp_mu)^2
                                    / (2 * belltemp_sigma^2))
    T_t     = clip(belltemp_t_base + bell + belltemp_vh_weight * VH_norm,
                   0.3, 2.2)

    t_frac  = clip((T_t - 0.3) / (2.2 - 0.3), 0, 1)
    min_p_t = clip(belltemp_min_p_base + belltemp_min_p_scale * t_frac,
                   0, 0.15)

The per-token ``min_p`` is published via ``TemperatureResult.diagnostics``
(key ``"min_p"``). Port note: the legacy adaptive min-p normalized
``t_frac`` over configurable ``T_min``/``T_max`` knobs (defaults 0.1/3.0);
this port pins the normalization to the repo-wide temperature guardrail
box ``[0.3, 2.2]`` — legacy configs using the coupling must be re-derived
(documented in the v7 program).

Family hypothesis (assessment §3.5): the conservative arm — the V5
``BELLTEMP_R02_03`` posted the 2nd-highest ConvJ overall with GPQA 0.439,
the best "coherent-and-capable" adaptive preset in the corpus.
``belltemp_t_peak`` is hard-capped at 1.5 in the config model (shares
GDT's ablation-located coherence cliff).

**Static-clone parameterisation (FR-8.5):** ``belltemp_t_peak = 0``,
``belltemp_vh_weight = 0`` and ``belltemp_min_p_scale = 0`` reduce the
strategy exactly to fixed ``T = clip(belltemp_t_base)`` with constant
``min_p = clip(belltemp_min_p_base)`` on every token.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_entropy_varentropy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

# Repo-wide V6 guardrail box — hard clamps applied AFTER the formulas.
# The box also fixes the adaptive min-p normalization (see module doc).
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class BellTempStrategy(TemperatureStrategy):
    """Stateless BellTemp temperature strategy.

    Carries no per-request state. ``vocab_size`` is required: normalized
    entropy uses ``H_max = ln(V)``.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize with the model's vocabulary size.

        Args:
            vocab_size: Number of tokens in the model vocabulary.

        Raises:
            ValueError: If vocab_size < 2 (entropy normalization undefined).
        """
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self._vocab_size = vocab_size
        self._max_entropy = math.log(vocab_size)

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute (T, min_p) from the bell curve + varentropy boost.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 8 BellTemp
                hyperparameters (``belltemp_t_base``, ``belltemp_t_peak``,
                ``belltemp_mu``, ``belltemp_sigma``, ``belltemp_vh_weight``,
                ``belltemp_lambda_vh``, ``belltemp_min_p_base``,
                ``belltemp_min_p_scale``).

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p``, ``varentropy``, ``h_norm``,
            ``vh_norm``, ``bell``.
        """
        h, vh = compute_entropy_varentropy(logits)

        h_norm = h / self._max_entropy
        h_norm = max(0.0, min(1.0, h_norm))
        vh_norm = 1.0 - math.exp(-vh / config.belltemp_lambda_vh)

        bell = config.belltemp_t_peak * math.exp(
            -((h_norm - config.belltemp_mu) ** 2) / (2.0 * config.belltemp_sigma**2)
        )
        raw_temp = config.belltemp_t_base + bell + config.belltemp_vh_weight * vh_norm
        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))

        # Adaptive min-p, normalized over the guardrail box (module doc).
        t_frac = (temperature - _TEMP_CLAMP[0]) / (_TEMP_CLAMP[1] - _TEMP_CLAMP[0])
        t_frac = max(0.0, min(1.0, t_frac))
        raw_min_p = config.belltemp_min_p_base + config.belltemp_min_p_scale * t_frac
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "belltemp: H=%.4f VH=%.4f bell=%.4f T=%.4f min_p=%.4f",
                h,
                vh,
                bell,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "belltemp",
                "min_p": min_p,
                "varentropy": vh,
                "h_norm": h_norm,
                "vh_norm": vh_norm,
                "bell": bell,
                "pre_clamp_temp": raw_temp,
                "pre_clamp_min_p": raw_min_p,
            },
        )
