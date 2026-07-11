"""GDT: Gaussian Dynamic Temperature.

Stateless per-token strategy ported from createmp-evalsuite's V5
``GDTTempProcessor`` (v4v6 competitiveness assessment §3.2). Bell-curve
temperature mapping on normalized entropy, with an entropy-tapered
varentropy boost — varentropy gets full voice at low entropy (the
"creative goldmine") and is shut off as ``H_norm -> 1`` (model lost):

    H_norm   = clip(H / ln(V), 0, 1)
    VH_norm  = 1 - exp(-VH / gdt_lambda_vh)
    bell     = gdt_t_peak * exp(-(H_norm - gdt_mu)^2 / (2 * gdt_sigma^2))
    vh_boost = gdt_alpha * VH_norm * (1 - H_norm)
    T_t      = clip(gdt_t_base + bell + vh_boost, 0.3, 2.2)

    excess   = max(0, T_pre_clamp - gdt_t_base)
    min_p_t  = clip(gdt_min_p_base + gdt_min_p_scale * excess / gdt_t_peak,
                    0, 0.15)          # base only when gdt_t_peak == 0

The per-token ``min_p`` is published via ``TemperatureResult.diagnostics``
(key ``"min_p"``); the pipeline forwards it to the TokenSelector. The
min-p coupling scales with excess temperature above ``gdt_t_base``,
normalized by ``gdt_t_peak`` — at ``T = T_base`` (no boost) it is exactly
``gdt_min_p_base``.

Family hypothesis (assessment §3.2): entropy-*conditioned* heat protects
low-entropy (answer-token) contexts — GDT's taper is the structural reason
its V5 configs held GPQA 0.42-0.44 where EVDT fell to 0.09-0.22.
``gdt_t_peak`` is hard-capped at 1.5 in the config model (the
ablation-located family coherence cliff).

Defaults are the V5_GDT_R00_00 winning configuration.

**Static-clone parameterisation (FR-8.5):** ``gdt_t_peak = 0`` and
``gdt_alpha = 0`` reduce the strategy exactly to fixed
``T = clip(gdt_t_base)`` with constant ``min_p = clip(gdt_min_p_base)``
on every token.
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

# Repo-wide V6 guardrail box — hard clamps applied AFTER the formulas,
# same values as hvh_drift / evdt_tt / tt_exchange.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class GDTStrategy(TemperatureStrategy):
    """Stateless Gaussian Dynamic Temperature strategy.

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
        """Compute (T, min_p) from the bell curve + tapered varentropy boost.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 8 GDT hyperparameters
                (``gdt_t_base``, ``gdt_t_peak``, ``gdt_mu``, ``gdt_sigma``,
                ``gdt_alpha``, ``gdt_lambda_vh``, ``gdt_min_p_base``,
                ``gdt_min_p_scale``).

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p``, ``varentropy``, ``h_norm``,
            ``vh_norm``, ``bell``, ``vh_boost``.
        """
        h, vh = compute_entropy_varentropy(logits)

        h_norm = h / self._max_entropy
        h_norm = max(0.0, min(1.0, h_norm))
        vh_norm = 1.0 - math.exp(-vh / config.gdt_lambda_vh)

        bell = config.gdt_t_peak * math.exp(
            -((h_norm - config.gdt_mu) ** 2) / (2.0 * config.gdt_sigma**2)
        )
        vh_boost = config.gdt_alpha * vh_norm * (1.0 - h_norm)

        raw_temp = config.gdt_t_base + bell + vh_boost

        excess = max(0.0, raw_temp - config.gdt_t_base)
        if config.gdt_t_peak > 0.0:
            raw_min_p = config.gdt_min_p_base + config.gdt_min_p_scale * excess / config.gdt_t_peak
        else:
            raw_min_p = config.gdt_min_p_base

        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "gdt: H=%.4f VH=%.4f bell=%.4f vh_boost=%.4f T=%.4f min_p=%.4f",
                h,
                vh,
                bell,
                vh_boost,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "gdt",
                "min_p": min_p,
                "varentropy": vh,
                "h_norm": h_norm,
                "vh_norm": vh_norm,
                "bell": bell,
                "vh_boost": vh_boost,
                "pre_clamp_temp": raw_temp,
                "pre_clamp_min_p": raw_min_p,
            },
        )
