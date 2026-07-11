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
import math
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
        # Entropy via the shared dot-product formulation (perf tranche
        # 2026-07): with s = logits - max and Z = sum(exp(s)),
        # H = ln Z - dot(exp(s), s) / Z — no full-vocab log pass and no
        # ``probs``/``log_probs`` materialisation. Degenerate inputs
        # (non-finite max, NaN dot from a -inf logit) reproduce the
        # historical outputs: h = 0.0 and an empty kept set.
        max_logit = float(np.max(logits))
        degenerate = not math.isfinite(max_logit)
        if degenerate:
            h = 0.0
            exp_shifted = shifted = None
            sum_exp = 0.0
        else:
            shifted = logits - max_logit
            exp_shifted = np.exp(shifted)
            sum_exp = float(np.sum(exp_shifted))
            # NaN (exp(-inf) * -inf from a masked logit) is the deliberate
            # degenerate probe; suppress the numpy warning it triggers.
            with np.errstate(invalid="ignore"):
                m1 = float(np.dot(exp_shifted, shifted)) / sum_exp
            h = 0.0 if math.isnan(m1) else max(0.0, math.log(sum_exp) - m1)

        # 1. Classic entropy-linear min-p, clamped to the guardrail box.
        raw_min_p = config.tt_min_p_base + config.tt_min_p_scale * h
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        # 2. Measure the entropy of the kept (renormalised) distribution.
        #    This is a measurement only — logits are never modified here;
        #    the selector applies the published min_p downstream. The mask
        #    ``p >= min_p * max(p)`` is evaluated on the exp values
        #    directly: max(exp(s)) == 1.0 exactly (the max logit shifts to
        #    0), so the condition is ``exp(s) >= min_p`` — the same set up
        #    to division-rounding at the exact threshold boundary.
        if degenerate or exp_shifted is None or shifted is None:
            h_kept = 0.0
            n_kept = 0
        elif min_p > 0.0:
            mask = exp_shifted >= min_p
            kept_e = exp_shifted[mask]
            kept_s = shifted[mask]
            kept_sum = float(kept_e.sum())
            # mask always retains the argmax (exp == 1.0 >= min_p for
            # min_p <= 1), so kept_sum > 0 by construction. H of the
            # renormalised kept distribution q = e / K is
            # ln K - dot(e, s) / K (same identity as above).
            h_kept = max(0.0, math.log(kept_sum) - float(np.dot(kept_e, kept_s)) / kept_sum)
            n_kept = int(np.count_nonzero(mask))
        else:
            h_kept = h
            n_kept = int(np.count_nonzero(exp_shifted))

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
