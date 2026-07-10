"""EVDT-TT: truncate-first-then-temperature.

Stateless per-token strategy ported from the V6 research spec (§7.1):

    min_p_t = clip(evdt_min_p_base + evdt_min_p_scale*H + evdt_min_p_vh*VH, 0, 0.15)
    T_t     = clip(evdt_t_base + evdt_alpha*H + evdt_beta*VH, 0.3, 2.2)

Both formulas are linear in the RAW entropy ``H`` and varentropy ``VH``
of the current token's logit distribution. The strategy publishes
``min_p_t`` via ``TemperatureResult.diagnostics`` (key ``"min_p"``); the
pipeline forwards it to the TokenSelector.

The family's defining property — truncating the raw distribution BEFORE
temperature is applied, which yields a support set unreachable by any
static ``(T, min_p)`` configuration — requires the selector-order option
``qr_truncate_first: true`` (per-request flag, default ``false``; see
``selection/selector.py`` and AGENTS.md invariant 15). Without the flag
this strategy is a scale-then-truncate EVDT variant sharing its support
set with static configurations.

Defaults are the V6 predicted-best values (research spec §7.1):
``T_base=1.25, alpha=0.35, beta=-0.10, min_p_base=0.008,
min_p_scale=0.015, min_p_vh=0.005``. The spec text clips min-p at 0.2;
this port uses the repo-wide V6 guardrail box (min-p <= 0.15), matching
the createmp reference implementation's ``MP_CLAMP``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

# V6 guardrail box (research spec §8.5 / createmp _common.py T_CLAMP,
# MP_CLAMP). Hard clamps applied AFTER the linear formulas, same values
# as hvh_drift.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class EVDTTTStrategy(TemperatureStrategy):
    """Stateless EVDT-TT temperature strategy.

    Carries no per-request state; ``vocab_size`` is accepted for registry
    parity and is not used in the math.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the strategy.

        Args:
            vocab_size: Accepted for registry parity; unused in the math.
        """
        self._vocab_size = vocab_size

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute (T, min_p) linear in raw entropy and varentropy.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 6 EVDT-TT
                hyperparameters (``evdt_t_base``, ``evdt_alpha``,
                ``evdt_beta``, ``evdt_min_p_base``, ``evdt_min_p_scale``,
                ``evdt_min_p_vh``).

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p`` and ``varentropy``.
        """
        # Stable softmax with a single exp pass (same shape as hvh_drift).
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        sum_exp = float(np.sum(exp_shifted))
        log_probs = shifted - np.log(sum_exp)
        probs = exp_shifted / sum_exp

        # H = -sum(p * log p); VH = sum(p * (-log p - H)^2).
        h = max(0.0, float(-np.sum(probs * log_probs)))
        vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))

        raw_min_p = config.evdt_min_p_base + config.evdt_min_p_scale * h + config.evdt_min_p_vh * vh
        raw_temp = config.evdt_t_base + config.evdt_alpha * h + config.evdt_beta * vh

        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "evdt_tt: H=%.4f VH=%.4f T=%.4f min_p=%.4f",
                h,
                vh,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "evdt_tt",
                "min_p": min_p,
                "varentropy": vh,
                "pre_clamp_temp": raw_temp,
                "pre_clamp_min_p": raw_min_p,
            },
        )
