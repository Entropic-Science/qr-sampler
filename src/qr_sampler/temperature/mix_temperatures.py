"""Mixture-of-Temperatures: convex mix of two temperature-sampled distributions.

Stateless per-token strategy ported from createmp-evalsuite's V6
``MixtureOfTemperaturesProcessor`` (V6 research spec §7.4):

    p_cool = softmax(l / mix_t_cool)
    p_hot  = softmax(l / mix_t_hot)
    alpha  = sigmoid(mix_gate_a * (H - mix_gate_b)
                     + mix_gate_c * (VH - mix_gate_d))
    p_mix  = alpha * p_hot + (1 - alpha) * p_cool

Reading: the cool arm makes grammar, the hot arm makes interesting words;
the gate routes by current (H, VH). A convex mixture of two softmax powers
is **not** a power — this is the one V6 family that genuinely leaves the
single-temperature family (assessment §7.4), so the strategy publishes the
fully transformed logits ``ln(p_mix)`` via
``TemperatureResult.diagnostics["transformed_logits"]`` (the pipeline's
distribution seam) and returns ``temperature = 1.0`` for the selector —
the mixing already encodes both temperatures. The min-p threshold
(``diagnostics["min_p"] = mix_min_p``) then applies to the mixed
distribution itself, exactly as in the V6 reference. The practical tail
effect: the mixture tail is dominated by ``alpha * p_hot`` (a *tail
floor*), so min_p interacts with both arms at once.

Bounds note (assessment §8.2 item 6): ``mix_t_hot`` is re-widened to the
repo guardrail ceiling (2.2) — the V6 re-centering to <= 1.6 partially
fenced the family out of the region where the best statics live. The
*sharp-gate hot* variant (``mix_gate_a`` large, ``mix_t_hot`` ~1.7-1.8:
rare-but-genuinely-hot tokens) is reachable by construction. Pair hot arms
with a hard truncation floor.

``diagnostics["t_mix"]`` records the mix-weighted temperature
``alpha * T_hot + (1 - alpha) * T_cool`` — not a physically sampled T, but
the most informative scalar summary for logging.

Defaults are the V6 §7.4 predicted values.

**Static-clone parameterisation (FR-8.5):** ``mix_t_cool = mix_t_hot = T``
makes ``p_mix = softmax(l / T)`` exactly, for every gate value — a fixed-T
sampler with constant ``min_p = mix_min_p``. The clone still routes
through the transformed-logits seam (deliberately: clone checks exercise
the mixture plumbing end-to-end), and matches the static twin's
distribution exactly up to floating-point rounding.
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

_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


def _stable_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable ``softmax(logits / temperature)``.

    Args:
        logits: 1-D logit array (may contain ``-inf`` for masked tokens).
        temperature: Positive sampling temperature.

    Returns:
        Probability array of the same shape. Degenerate inputs (no finite
        logit) return the uniform distribution, mirroring the selector's
        degenerate-softmax contract.
    """
    scaled = logits / temperature
    max_logit = float(np.max(scaled))
    if not math.isfinite(max_logit):
        finite = scaled[np.isfinite(scaled)]
        if finite.size == 0:
            return np.full(scaled.size, 1.0 / scaled.size)
        max_logit = float(np.max(finite))
    exp_shifted = np.exp(scaled - max_logit)
    total = float(np.sum(exp_shifted))
    result: np.ndarray = exp_shifted / total
    return result


class MixTemperaturesStrategy(TemperatureStrategy):
    """Stateless Mixture-of-Temperatures strategy.

    Carries no per-request state; ``vocab_size`` is accepted for registry
    parity and is not used in the math (the gate operates on raw nats).
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize the strategy.

        Args:
            vocab_size: Accepted for registry parity; unused in the math.
        """
        self._vocab_size = vocab_size

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Mix cool and hot softmax arms under the (H, VH) gate.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 7 mixture
                hyperparameters (``mix_t_cool``, ``mix_t_hot``,
                ``mix_gate_a``, ``mix_gate_b``, ``mix_gate_c``,
                ``mix_gate_d``, ``mix_min_p``).

        Returns:
            TemperatureResult with ``temperature = 1.0`` (the mixed
            distribution rides ``diagnostics["transformed_logits"]``),
            Shannon entropy of the RAW distribution, and diagnostics
            containing ``min_p``, ``varentropy``, ``alpha``, ``t_mix``.
        """
        h, vh = compute_entropy_varentropy(logits)

        # Gate: higher H or VH pulls toward the hot arm.
        gate_logit = config.mix_gate_a * (h - config.mix_gate_b) + config.mix_gate_c * (
            vh - config.mix_gate_d
        )
        # Guard exp overflow for extreme gate inputs.
        alpha = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, gate_logit))))

        p_cool = _stable_softmax(logits, config.mix_t_cool)
        p_hot = _stable_softmax(logits, config.mix_t_hot)
        p_mix = alpha * p_hot + (1.0 - alpha) * p_cool

        # ln(0) = -inf keeps masked tokens masked — exactly right, so the
        # divide-by-zero warning is suppressed rather than clamped away.
        with np.errstate(divide="ignore"):
            mixed_logits = np.log(p_mix)

        t_mix = alpha * config.mix_t_hot + (1.0 - alpha) * config.mix_t_cool
        min_p = float(np.clip(config.mix_min_p, *_MIN_P_CLAMP))

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "mix_temperatures: H=%.4f VH=%.4f alpha=%.4f t_mix=%.4f min_p=%.4f",
                h,
                vh,
                alpha,
                t_mix,
                min_p,
            )

        return TemperatureResult(
            temperature=1.0,
            shannon_entropy=h,
            diagnostics={
                "strategy": "mix_temperatures",
                "min_p": min_p,
                "varentropy": vh,
                "alpha": alpha,
                "t_mix": t_mix,
                "transformed_logits": mixed_logits,
            },
        )
