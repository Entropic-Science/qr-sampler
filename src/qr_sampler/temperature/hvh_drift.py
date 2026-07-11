"""HVH-Drift: entropy-history aware temperature strategy.

Stateful, per-request strategy ported from createmp-evalsuite V6 (research
spec §8). Two scalar EMAs track recent entropy ``H`` and varentropy ``VH``;
the instantaneous-vs-smoothed gap (``dH``, ``dVH``) feeds the temperature
and min-p formulas.

v7 semantics alignment (v4v6 competitiveness assessment §7.4): drift is
computed against the *previous* EMA (``dH = H_t - ema_{t-1}``), and the
EMA updates afterwards. The V6 reference updated first, making
``dH = (1 - lambda) * (H_t - ema_{t-1})`` — coupling ``hvh_lambda_ema`` to
the drift gains and degenerating smoothly to zero drift as lambda -> 1.
The decoupled form searches cleanly down to ``lambda_ema ~ 0.01`` (the V6
champion sat at 0.02, outside the old GP bounds).

Formulas:

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
    compute_entropy_varentropy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


# V6 reference math (createmp-evalsuite/samplers/v6/_common.py:26-35).
# Hard guardrail clamps applied AFTER the linear formulas.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


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
        # H = -sum(p * log p); VH = sum(p * (-log p - H)^2). Fused
        # dot-product formulation (perf tranche 2026-07): no full-vocab
        # ``probs``/``log_probs`` materialisation — see
        # ``compute_entropy_varentropy`` for the equivalence argument.
        h, vh = compute_entropy_varentropy(logits)

        # Drift vs the PREVIOUS EMA, then update (v7 alignment; v4v6
        # competitiveness assessment §7.4). The V6 reference updated the
        # EMA first and measured drift against the post-update value,
        # which scales the drift signal by (1 - lambda) and couples
        # ``hvh_lambda_ema`` to ``hvh_gamma_dh``/``hvh_nu_dh`` — warping
        # the BO space (lambda -> 1 shrinks drift to 0 smoothly).
        # Measuring against the previous EMA decouples them:
        # dH = H_t - ema_{t-1}, independent of lambda at fixed history.
        # On the first call, seed EMAs with current values so
        # dH = dVH = 0 (no cold-start branch).
        if self._first_call:
            self.H_ema = h
            self.VH_ema = vh
            self._first_call = False
            d_h = 0.0
            d_vh = 0.0
        else:
            d_h = h - self.H_ema
            d_vh = vh - self.VH_ema
            lam = config.hvh_lambda_ema
            self.H_ema = (1.0 - lam) * self.H_ema + lam * h
            self.VH_ema = (1.0 - lam) * self.VH_ema + lam * vh

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
