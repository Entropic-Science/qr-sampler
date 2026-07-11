"""DynaTemp: entropy-linear dynamic temperature (llama.cpp lineage).

Stateless per-token strategy ported from createmp-evalsuite's V5
``DynaTempProcessor`` (v4v6 competitiveness assessment §3.3), standard
direction only (low entropy -> low T, high entropy -> high T; the
inverted direction is retired):

    H_norm  = clip(H / ln(V), 0, 1)
    T_t     = clip(dynatemp_t_center - dynatemp_t_range
                   + 2 * dynatemp_t_range * H_norm^dynatemp_exponent,
                   0.3, 2.2)
    min_p_t = dynatemp_min_p                      (constant per token)

The constant ``min_p`` is still published via
``TemperatureResult.diagnostics`` (key ``"min_p"``) so the family fully
owns its truncation — ``qr_min_p_base`` is not consulted.

Family hypothesis (assessment §3.3): min_p is the family's safety valve,
not a creativity lever; the near-golden ``V5_ABL_DYN_08`` recipe is *hot +
very hard truncation* (T_center=1.875, T_range=0.80, min_p=0.12) — the
same recipe as the best statics — and is reachable within this port's
bounds by design (min_p cap 0.15, temperature guardrail ceiling 2.2).

Defaults are the balanced V5_DYNATEMP_R00_09 member.

**Static-clone parameterisation (FR-8.5):** ``dynatemp_t_range = 0``
reduces the strategy exactly to fixed ``T = clip(dynatemp_t_center)`` with
constant ``min_p = dynatemp_min_p`` on every token.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_shannon_entropy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

# Repo-wide V6 guardrail box — hard clamps applied AFTER the formula.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class DynaTempStrategy(TemperatureStrategy):
    """Stateless DynaTemp temperature strategy.

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
        """Compute T linear in (powered) normalized entropy.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 4 DynaTemp
                hyperparameters (``dynatemp_t_center``, ``dynatemp_t_range``,
                ``dynatemp_exponent``, ``dynatemp_min_p``).

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p`` and ``h_norm``.
        """
        h = compute_shannon_entropy(logits)

        h_norm = h / self._max_entropy
        h_norm = max(0.0, min(1.0, h_norm))

        raw_temp = (
            config.dynatemp_t_center
            - config.dynatemp_t_range
            + 2.0 * config.dynatemp_t_range * (h_norm**config.dynatemp_exponent)
        )

        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))
        min_p = float(np.clip(config.dynatemp_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "dynatemp: H=%.4f H_norm=%.4f T=%.4f min_p=%.4f",
                h,
                h_norm,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "dynatemp",
                "min_p": min_p,
                "h_norm": h_norm,
                "pre_clamp_temp": raw_temp,
            },
        )
