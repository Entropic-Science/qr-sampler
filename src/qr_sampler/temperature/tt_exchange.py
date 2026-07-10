"""TT-Entropy-Exchange: entropy removed by truncation is redeployed as T.

Stateless per-token strategy ported from the V6 research spec (§7.3):

    1. ``min_p_t = clip(tt_min_p_base + tt_min_p_scale * H, 0, 0.15)``
    2. Truncate the RAW probability distribution with ``min_p_t``;
       renormalise to ``p_kept`` with entropy ``H_kept``.
    3. ``T_t = clip(tt_t_base + tt_gamma * max(0, H - H_kept), 0.3, 2.2)``
       — ``H - H_kept`` is the entropy *removed* by truncation. Light
       truncation keeps ``T`` near ``tt_t_base``; aggressive truncation
       scales ``T`` up to compensate for the information removed.

The strategy publishes ``min_p_t`` via ``TemperatureResult.diagnostics``
(key ``"min_p"``); the pipeline forwards it to the TokenSelector. On the
default selector order (invariant 15: temperature is applied before the
min-p mask) this is a scale-then-truncate port of the family. The exact
V6 order — min-p on the raw distribution, THEN temperature on the kept
support — is available by additionally setting the per-request flag
``qr_truncate_first: true`` (see ``selection/selector.py``).

The internal truncation in step 2 is a *measurement* only (to obtain
``H_kept``); the strategy never modifies logits. Defaults are the V6
predicted-best values (research spec §7.3): ``T_base=1.0, gamma=0.6,
min_p_base=0.005, min_p_scale=0.025``.
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
# MP_CLAMP). Hard clamps applied AFTER the formulas, same values as
# hvh_drift.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class TTExchangeStrategy(TemperatureStrategy):
    """Stateless TT-Entropy-Exchange temperature strategy.

    Carries no per-request state (each token is computed from the current
    logits alone), so instances are safe to rebuild per request like every
    other strategy; ``vocab_size`` is accepted for registry parity and is
    not used in the math.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the strategy.

        Args:
            vocab_size: Accepted for registry parity; unused in the math.
        """
        self._vocab_size = vocab_size

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute temperature from the entropy removed by min-p truncation.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 4 TT-Exchange
                hyperparameters (``tt_t_base``, ``tt_gamma``,
                ``tt_min_p_base``, ``tt_min_p_scale``).

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p``, ``h_kept``,
            ``entropy_removed``, ``n_kept``.
        """
        # Stable softmax with a single exp pass (same shape as hvh_drift).
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        sum_exp = float(np.sum(exp_shifted))
        log_probs = shifted - np.log(sum_exp)
        probs = exp_shifted / sum_exp

        h = max(0.0, float(-np.sum(probs * log_probs)))

        # 1. Classic entropy-linear min-p, clamped to the guardrail box.
        raw_min_p = config.tt_min_p_base + config.tt_min_p_scale * h
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        # 2. Measure the entropy of the kept (renormalised) distribution.
        #    This is a measurement only — logits are never modified here;
        #    the selector applies the published min_p downstream.
        if min_p > 0.0:
            mask = probs >= min_p * float(probs.max())
            kept = probs[mask]
            kept_sum = float(kept.sum())
            # mask always retains the argmax (probs.max() >= min_p * max
            # for min_p <= 1), so kept_sum > 0 by construction.
            kept = kept / kept_sum
            h_kept = max(0.0, float(-np.sum(kept * np.log(kept))))
            n_kept = int(mask.sum())
        else:
            h_kept = h
            n_kept = int(np.sum(probs > 0))

        # 3. Redeploy the removed entropy as temperature.
        entropy_removed = max(0.0, h - h_kept)
        raw_temp = config.tt_t_base + config.tt_gamma * entropy_removed
        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "tt_exchange: H=%.4f H_kept=%.4f removed=%.4f T=%.4f min_p=%.4f",
                h,
                h_kept,
                entropy_removed,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "tt_exchange",
                "min_p": min_p,
                "h_kept": h_kept,
                "entropy_removed": entropy_removed,
                "n_kept": n_kept,
                "pre_clamp_temp": raw_temp,
                "pre_clamp_min_p": raw_min_p,
            },
        )
