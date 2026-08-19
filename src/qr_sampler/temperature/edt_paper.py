"""EDT (paper form): entropy-based dynamic temperature, arXiv:2403.14541.

Faithful port of the published EDT formula (Zhang et al. 2024, "EDT:
Improving Large Language Models' Generation by Entropy-based Dynamic
Temperature Sampling")::

    T = edtp_t0 * edtp_n ** (edtp_theta / H)

where ``H`` is the (unnormalized) Shannon entropy of the token distribution
in nats and ``0 < edtp_n < 1``. Behaviour: as ``H -> 0`` (model confident)
the exponent blows up and ``T -> 0`` (near-greedy); as ``H -> inf`` (model
uncertain) ``T -> edtp_t0`` from below — temperature RISES with entropy
toward the ``T0`` ceiling. Paper-recommended starting hyperparameters
(their experiments): ``T0 = 0.6``, ``theta = 0.1``, ``N = 0.8`` (fixed).

This is a *different function family* from the existing ``edt`` strategy
(createmp-evalsuite lineage, ``T = base * H_norm^exponent`` on normalized
entropy) — the two are not inter-expressible, hence the separate strategy.

Port notes (documented deviations — verified against arXiv:2403.14541v2,
2026-08-19; any figure/caption naming this strategy must say "paper formula
under the v7 guardrail box", NOT "as published"):

- The repo-wide guardrail box clamps T to ``[0.3, 2.2]`` AFTER the formula,
  like every other strategy. The paper applies NO lower bound ("T0 indicates
  the upper bound of the temperature") and its near-greedy ``T -> 0`` regime
  for confident tokens is the behaviour that motivates the method; under
  this port those tokens sample at ``T = 0.3``. With the paper's
  ``T0 = 0.6`` the effective range is ``[0.3, 0.6]``.
- ``H`` is guarded below by a small epsilon so ``theta / H`` never
  divides by zero (a delta distribution ends at the clamp floor anyway).
- **The paper runs under top-p = 0.95** ("We fix the Top-p (p=0.95) ...
  during our experiments for dynamic strategy") — the earlier claim here
  that it pairs no truncation sampler was wrong. This strategy emits no
  per-token ``min_p`` diagnostic, so with the default ``min_p_base = 0.0``
  the port is UNtruncated; a faithful-companion run must set top-p 0.95
  at the selector level.
- Eq. 5 states ``Entropy = -sum p_i log(p_i)`` with **no base given**
  anywhere in the paper (no official code release to check). This port
  assumes nats (the PyTorch/llama-ecosystem default); if the authors meant
  log2, the exponent is off by ln(2) ~ 0.693 — equivalent to running
  ``theta ~ 0.069`` instead of 0.1 (well inside the paper's own 0.1-5.0
  theta sweep, so the function family is unchanged).
- ``T0=0.6 / theta=0.1 / N=0.8`` is the paper's *recommended starting
  point* ("we can start with a basic set of hyperparameters, such as
  T0=0.6 and theta=0.1"), not its per-task tuned best (their MS MARCO
  ablations preferred theta=1.0 at T0=1.0, and T0=0.9 at theta=0.1).

Static-clone parameterisation (FR-8.5): ``edtp_theta = 0`` reduces exactly
to fixed ``T = clip(edtp_t0)`` on every token (``N^0 = 1``).

Per-request override::

    "extra_args": {
        "qr_temperature_strategy": "edt_paper",
        "qr_edtp_t0": 0.6,
        "qr_edtp_theta": 0.1
    }
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_shannon_entropy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

# Repo-wide V6 guardrail box — hard clamp applied AFTER the formula.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)

#: Entropy floor: below this the distribution is effectively a delta and the
#: formula's T is ~0 (clamped to the box floor); guards the division.
_H_EPS: float = 1e-6

_logger = logging.getLogger("qr_sampler")


class EDTPaperStrategy(TemperatureStrategy):
    """Stateless paper-form EDT: ``T = T0 * N^(theta / H)``.

    Carries no per-request state and needs no vocabulary size — the paper
    formula consumes the raw (unnormalized) Shannon entropy.
    """

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute the paper-form EDT temperature for one token.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing ``edtp_t0``,
                ``edtp_theta`` and ``edtp_n``.

        Returns:
            TemperatureResult with temperature, Shannon entropy, and
            diagnostics (``pre_clamp_temp``, the raw exponent input).
        """
        h = compute_shannon_entropy(logits)

        h_safe = max(h, _H_EPS)
        raw_temp = config.edtp_t0 * (config.edtp_n ** (config.edtp_theta / h_safe))

        temperature = float(np.clip(raw_temp, *_TEMP_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "edt_paper: H=%.4f T=%.4f (raw %.4f)",
                h,
                temperature,
                raw_temp,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics={
                "strategy": "edt_paper",
                "pre_clamp_temp": raw_temp,
            },
        )
