"""Base classes for temperature strategies.

Defines the abstract interface, result type, and shared utility for
computing Shannon entropy from logit distributions.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


@dataclass(frozen=True, slots=True)
class TemperatureResult:
    """Result of a temperature strategy computation.

    Attributes:
        temperature: Temperature to use for this token's sampling.
        shannon_entropy: Shannon entropy H of the logit distribution (nats).
        diagnostics: Additional info (strategy-specific details).
    """

    temperature: float
    shannon_entropy: float
    diagnostics: dict[str, Any]


class TemperatureStrategy(ABC):
    """Abstract base class for temperature strategies.

    Implementations compute a per-token temperature from the logit
    distribution. All strategies must compute and return Shannon entropy
    even if it is not used in the temperature formula, because the
    logging subsystem depends on it.
    """

    @abstractmethod
    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute temperature for a single token.

        Args:
            logits: 1-D logit array for the current token (vocab_size,).
            config: Active configuration for this request.

        Returns:
            TemperatureResult with temperature, Shannon entropy, and diagnostics.
        """


def compute_shannon_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy H = -sum(p_i * ln(p_i)) using numerically stable softmax.

    Uses the shift-by-max trick for numerical stability. Returns 0.0 for
    degenerate distributions where only one token has non-zero probability.

    Hot-path formulation (perf tranche 2026-07): with ``s_i = logit_i - max``
    and ``Z = sum(exp(s_i))``, ``ln p_i = s_i - ln Z``, so

        H = -sum(p_i * (s_i - ln Z)) = ln Z - dot(exp(s), s) / Z

    which needs one BLAS dot instead of a full-vocab ``log`` pass plus two
    boolean fancy-index copies (~40% of the per-token sampling budget at a
    152k vocabulary). Masked logits (``-inf``) make the dot produce NaN via
    ``0 * -inf``; that rare case falls back to the exact masked formula.

    Args:
        logits: 1-D logit array (vocab_size,).

    Returns:
        Shannon entropy in nats (natural log base).
    """
    max_logit = float(np.max(logits))
    if not math.isfinite(max_logit):
        # All-(-inf), or NaN/+inf contamination: the historical path
        # propagated NaN probabilities into an all-False mask and
        # returned 0.0 for every one of these degenerate shapes.
        return 0.0

    # Numerically stable softmax: shift by max to prevent overflow.
    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))

    if sum_exp == 0.0:
        return 0.0

    # dot(exp(s), s) = sum over tokens of exp(s_i) * s_i. A NaN result is
    # the deliberate probe for masked (-inf) logits, so the corresponding
    # numpy warning is suppressed.
    with np.errstate(invalid="ignore"):
        weighted = float(np.dot(exp_shifted, shifted))
    if math.isnan(weighted):
        # A masked token contributed exp(-inf) * -inf = NaN. Take the
        # exact masked path (identical to the historical implementation).
        probs = exp_shifted / sum_exp
        mask = probs > 0
        log_probs = np.log(probs[mask])
        entropy = -float(np.sum(probs[mask] * log_probs))
        return max(0.0, entropy)

    entropy = math.log(sum_exp) - weighted / sum_exp

    # Guard against floating-point artifacts producing tiny negatives.
    return max(0.0, entropy)


def compute_entropy_varentropy(logits: np.ndarray) -> tuple[float, float]:
    """Compute Shannon entropy H and varentropy VH of ``softmax(logits)``.

    ``VH = sum(p_i * (-ln p_i - H)^2)`` — the variance of the surprisal.
    Shared hot-path helper for the drift-family strategies (``hvh_drift``,
    ``evdt_tt``), which previously each materialised full-vocab ``probs``
    and ``log_probs`` arrays per token.

    Formulation: with ``s_i = logit_i - max`` and ``m1 = E_p[s]``,
    ``H = ln Z - m1`` and ``-ln p_i - H = m1 - s_i``, so
    ``VH = E_p[(s - m1)^2]`` — computed with a centered second moment (two
    BLAS dots) for the same numerical conditioning as the direct formula.

    Degenerate inputs (any ``-inf``/NaN logit, all ``-inf``) return
    ``(0.0, 0.0)`` — exactly what the historical strategy implementations
    produced (``0 * -inf = NaN`` folded to 0.0 through their
    ``max(0.0, ...)`` guards).

    Args:
        logits: 1-D logit array (vocab_size,).

    Returns:
        Tuple ``(shannon_entropy, varentropy)`` in nats (both >= 0.0).
    """
    max_logit = float(np.max(logits))
    if not math.isfinite(max_logit):
        return 0.0, 0.0

    shifted = logits - max_logit
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))
    if sum_exp == 0.0:
        return 0.0, 0.0

    with np.errstate(invalid="ignore"):
        m1 = float(np.dot(exp_shifted, shifted)) / sum_exp
    if math.isnan(m1):
        # A masked (-inf) logit contributed exp(-inf) * -inf = NaN — the
        # historical implementations returned (0.0, 0.0) for this shape.
        return 0.0, 0.0

    entropy = max(0.0, math.log(sum_exp) - m1)

    centered = shifted - m1
    np.multiply(centered, centered, out=centered)
    varentropy = float(np.dot(exp_shifted, centered)) / sum_exp
    return entropy, max(0.0, varentropy)
