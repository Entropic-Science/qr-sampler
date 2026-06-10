"""HVH-Drift: entropy-history aware temperature strategy.

Stateful, per-request strategy ported from createmp-evalsuite V6 (research
spec §8). Two scalar EMAs track recent entropy ``H`` and varentropy ``VH``;
the instantaneous-vs-smoothed gap (``dH``, ``dVH``) feeds the temperature
and min-p formulas:

    T_t     = T_base + alpha_H*H + alpha_VH*VH + gamma_dH*dH + delta_dVH*dVH
    min_p_t = min_p_base + kappa_H*H + nu_dH*dH

Both outputs are clamped to the V6 guardrail box (see ``_TEMP_CLAMP`` and
``_MIN_P_CLAMP``). On the first token, EMAs are seeded with the current
``H``/``VH``, so ``dH = dVH = 0`` -- there is no cold-start branch.

The per-token ``min_p`` is published into ``TemperatureResult.diagnostics``
under the key ``"min_p"``; the pipeline reads it and forwards it to the
TokenSelector. Defaults for the 9 hyperparameters live on
``QRSamplerConfig`` and trace back to the V6_HVD_R01_01 winning round in
``createmp-evalsuite/results/v6/round_final``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
)
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


# V6 reference math (createmp-evalsuite/samplers/v6/_common.py:26-35).
# Hard guardrail clamps applied AFTER the linear formulas.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


@TemperatureStrategyRegistry.register("hvh_drift")
class HVHDriftStrategy(TemperatureStrategy):
    """Per-request stateful HVH-Drift temperature strategy.

    Each instance owns its own ``H_ema``/``VH_ema`` state; engine adapters
    must build a fresh instance per request so state does not leak across
    sequences. The pipeline uses ``TemperatureStrategyRegistry.build()``,
    which calls this constructor with ``vocab_size`` -- the value is
    accepted for protocol parity but is not used in the math.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize EMA state.

        Args:
            vocab_size: Accepted for registry parity with EDT; unused here.
        """
        self._vocab_size = vocab_size
        self.H_ema: float = 0.0
        self.VH_ema: float = 0.0
        self._first_call: bool = True

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute temperature and min-p from current logits + EMA state.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 9 HVH hyperparameters.

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics containing ``min_p``, ``varentropy``, ``h_ema``,
            ``vh_ema``, ``d_h``, ``d_vh``.
        """
        # Stable softmax: shift by max, then log-normalize. iter-55: one
        # exp pass over the vocab instead of two — ``probs`` is derived
        # from the already-computed ``exp_shifted`` rather than
        # re-exponentiating ``log_probs`` (~0.7 ms/token saved at 152k
        # vocab). ``log_probs`` keeps the exact prior formula; ``probs``
        # is mathematically identical (may differ in the last ulp).
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        sum_exp = float(np.sum(exp_shifted))
        log_probs = shifted - np.log(sum_exp)
        probs = exp_shifted / sum_exp

        # H = -sum(p * log p); VH = sum(p * (-log p - H)^2).
        h = float(-np.sum(probs * log_probs))
        h = max(0.0, h)  # guard tiny negative float artifacts
        vh = float(np.sum(probs * (-log_probs - h) ** 2))
        vh = max(0.0, vh)

        # Compute drifts BEFORE EMA update (matches V6 reference order at
        # hvh_drift.py:127-137). On the first call, seed EMAs with current
        # values so dH = dVH = 0 (no cold-start branch).
        if self._first_call:
            self.H_ema = h
            self.VH_ema = vh
            self._first_call = False
            d_h = 0.0
            d_vh = 0.0
        else:
            lam = config.hvh_lambda_ema
            self.H_ema = (1.0 - lam) * self.H_ema + lam * h
            self.VH_ema = (1.0 - lam) * self.VH_ema + lam * vh
            d_h = h - self.H_ema
            d_vh = vh - self.VH_ema

        raw_temp = (
            config.hvh_t_base
            + config.hvh_alpha_h * h
            + config.hvh_alpha_vh * vh
            + config.hvh_gamma_dh * d_h
            + config.hvh_delta_dvh * d_vh
        )
        raw_min_p = config.hvh_min_p_base + config.hvh_kappa_h * h + config.hvh_nu_dh * d_h

        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))
        min_p = float(np.clip(raw_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "hvh_drift: H=%.4f VH=%.4f dH=%.4f dVH=%.4f T=%.4f min_p=%.4f",
                h,
                vh,
                d_h,
                d_vh,
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "hvh_drift",
                "min_p": min_p,
                "varentropy": vh,
                "h_ema": self.H_ema,
                "vh_ema": self.VH_ema,
                "d_h": d_h,
                "d_vh": d_vh,
                "pre_clamp_temp": raw_temp,
                "pre_clamp_min_p": raw_min_p,
            },
        )
