"""Equivalence tests for the dot-product entropy/varentropy fast paths.

The 2026-07 perf tranche reformulated ``compute_shannon_entropy`` (and the
new shared ``compute_entropy_varentropy``) from the direct
``-sum(p * log p)`` form into ``ln Z - dot(exp(s), s) / Z`` — removing the
full-vocab ``log`` pass. These tests pin the numerical equivalence against
straightforward reference implementations, plus the documented degenerate
behaviours.
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.temperature.base import (
    compute_entropy_varentropy,
    compute_shannon_entropy,
)


def _reference_entropy(logits: np.ndarray) -> float:
    """The historical masked implementation of Shannon entropy."""
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0.0:
        return 0.0
    probs = exp_shifted / sum_exp
    mask = probs > 0
    log_probs = np.log(probs[mask])
    return max(0.0, -float(np.sum(probs[mask] * log_probs)))


def _reference_varentropy(logits: np.ndarray) -> tuple[float, float]:
    """The historical drift-strategy H/VH computation."""
    shifted = logits - np.max(logits)
    exp_shifted = np.exp(shifted)
    sum_exp = float(np.sum(exp_shifted))
    log_probs = shifted - np.log(sum_exp)
    probs = exp_shifted / sum_exp
    h = max(0.0, float(-np.sum(probs * log_probs)))
    vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))
    return h, vh


class TestShannonEntropyFastPath:
    """compute_shannon_entropy matches the masked reference."""

    @pytest.mark.parametrize("size", [2, 17, 1000, 50_000])
    def test_random_logits_match_reference(self, size: int) -> None:
        """Random finite logits: fast path equals the reference."""
        rng = np.random.default_rng(size)
        logits = rng.normal(0.0, 3.0, size=size)
        assert compute_shannon_entropy(logits) == pytest.approx(
            _reference_entropy(logits), rel=1e-9, abs=1e-12
        )

    def test_float32_matches_reference(self) -> None:
        """float32 logits (the vLLM shape) stay within float32 tolerance."""
        rng = np.random.default_rng(42)
        logits = rng.normal(0.0, 4.0, size=20_000).astype(np.float32)
        assert compute_shannon_entropy(logits) == pytest.approx(
            _reference_entropy(logits.astype(np.float64)), rel=1e-4
        )

    def test_uniform_distribution(self) -> None:
        """Flat logits give H = ln(n)."""
        logits = np.zeros(1024)
        assert compute_shannon_entropy(logits) == pytest.approx(np.log(1024), rel=1e-12)

    def test_peaked_distribution_near_zero(self) -> None:
        """A single dominant logit drives H toward 0."""
        logits = np.zeros(64)
        logits[7] = 60.0
        assert compute_shannon_entropy(logits) == pytest.approx(0.0, abs=1e-12)

    def test_masked_logits_take_exact_fallback(self) -> None:
        """-inf entries trigger the masked path with identical results."""
        logits = np.array([2.0, -np.inf, 1.0, -np.inf, 0.5])
        assert compute_shannon_entropy(logits) == pytest.approx(
            _reference_entropy(logits), rel=1e-12
        )

    def test_single_finite_logit(self) -> None:
        """One finite logit among -inf: entropy is exactly 0."""
        logits = np.full(16, -np.inf)
        logits[3] = 2.0
        assert compute_shannon_entropy(logits) == 0.0

    def test_all_neg_inf_returns_zero(self) -> None:
        """All -inf (fully masked) returns the degenerate 0.0."""
        assert compute_shannon_entropy(np.full(8, -np.inf)) == 0.0

    def test_nan_returns_zero(self) -> None:
        """NaN contamination returns the historical degenerate 0.0."""
        logits = np.array([1.0, np.nan, 2.0])
        assert compute_shannon_entropy(logits) == 0.0

    def test_underflow_only_survivor(self) -> None:
        """float32 underflow zeros match the reference's masked handling."""
        logits = np.zeros(32, dtype=np.float32)
        logits[0] = 200.0  # exp(-200) underflows in float32
        assert compute_shannon_entropy(logits) == pytest.approx(
            _reference_entropy(logits.astype(np.float64)), abs=1e-6
        )


class TestEntropyVarentropyFastPath:
    """compute_entropy_varentropy matches the historical drift math."""

    @pytest.mark.parametrize("size", [4, 300, 10_000])
    def test_random_logits_match_reference(self, size: int) -> None:
        """Random finite logits: H and VH match the reference."""
        rng = np.random.default_rng(size + 1)
        logits = rng.normal(0.0, 2.5, size=size)
        h, vh = compute_entropy_varentropy(logits)
        ref_h, ref_vh = _reference_varentropy(logits)
        assert h == pytest.approx(ref_h, rel=1e-9, abs=1e-12)
        assert vh == pytest.approx(ref_vh, rel=1e-7, abs=1e-10)

    def test_peaked_distribution(self) -> None:
        """A dominant token: H ~ 0 and VH ~ 0."""
        logits = np.zeros(128)
        logits[0] = 50.0
        h, vh = compute_entropy_varentropy(logits)
        assert h == pytest.approx(0.0, abs=1e-12)
        assert vh == pytest.approx(0.0, abs=1e-12)

    def test_uniform_distribution(self) -> None:
        """Flat logits: H = ln(n), VH = 0 (constant surprisal)."""
        logits = np.zeros(256)
        h, vh = compute_entropy_varentropy(logits)
        assert h == pytest.approx(np.log(256), rel=1e-12)
        assert vh == pytest.approx(0.0, abs=1e-12)

    def test_masked_logits_degenerate(self) -> None:
        """-inf entries reproduce the historical (0.0, 0.0) outputs."""
        logits = np.array([2.0, -np.inf, 1.0])
        assert compute_entropy_varentropy(logits) == (0.0, 0.0)

    def test_all_neg_inf_degenerate(self) -> None:
        """All -inf returns (0.0, 0.0)."""
        assert compute_entropy_varentropy(np.full(8, -np.inf)) == (0.0, 0.0)

    def test_nan_degenerate(self) -> None:
        """NaN contamination returns (0.0, 0.0)."""
        assert compute_entropy_varentropy(np.array([1.0, np.nan])) == (0.0, 0.0)


class TestStrategiesUseEquivalentMath:
    """Drift strategies produce the same outputs as before the fast path."""

    def test_hvh_drift_matches_reference_formulas(self) -> None:
        """hvh_drift's H/VH diagnostics equal the reference computation."""
        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.temperature.hvh_drift import HVHDriftStrategy

        rng = np.random.default_rng(5)
        config = QRSamplerConfig()
        strategy = HVHDriftStrategy(vocab_size=400)
        for _ in range(3):
            logits = rng.normal(0.0, 2.0, size=400)
            result = strategy.compute_temperature(logits, config)
            ref_h, ref_vh = _reference_varentropy(logits)
            assert result.shannon_entropy == pytest.approx(ref_h, rel=1e-9)
            assert result.diagnostics["varentropy"] == pytest.approx(ref_vh, rel=1e-7)

    def test_evdt_tt_matches_reference_formulas(self) -> None:
        """evdt_tt's H/VH match the reference computation."""
        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.temperature.evdt_tt import EVDTTTStrategy

        rng = np.random.default_rng(6)
        config = QRSamplerConfig()
        strategy = EVDTTTStrategy(vocab_size=300)
        logits = rng.normal(0.0, 2.0, size=300)
        result = strategy.compute_temperature(logits, config)
        ref_h, ref_vh = _reference_varentropy(logits)
        assert result.shannon_entropy == pytest.approx(ref_h, rel=1e-9)
        assert result.diagnostics["varentropy"] == pytest.approx(ref_vh, rel=1e-7)

    def test_tt_exchange_kept_measurement_matches_reference(self) -> None:
        """tt_exchange's h_kept/n_kept equal the historical probs-space math."""
        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.temperature.tt_exchange import TTExchangeStrategy

        rng = np.random.default_rng(8)
        config = QRSamplerConfig()
        strategy = TTExchangeStrategy(vocab_size=500)
        logits = rng.normal(0.0, 2.0, size=500)
        result = strategy.compute_temperature(logits, config)

        # Reference: probs-space min-p measurement (historical formula).
        shifted = logits - np.max(logits)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / exp_shifted.sum()
        h = _reference_entropy(logits)
        min_p = float(np.clip(config.tt_min_p_base + config.tt_min_p_scale * h, 0.0, 0.15))
        assert result.diagnostics["min_p"] == pytest.approx(min_p, rel=1e-9)
        if min_p > 0.0:
            mask = probs >= min_p * probs.max()
            kept = probs[mask]
            kept = kept / kept.sum()
            ref_h_kept = max(0.0, -float(np.sum(kept * np.log(kept))))
            assert result.diagnostics["n_kept"] == int(mask.sum())
            assert result.diagnostics["h_kept"] == pytest.approx(ref_h_kept, rel=1e-9)

    def test_tt_exchange_degenerate_all_masked(self) -> None:
        """All -inf logits reproduce the historical degenerate outputs."""
        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.temperature.tt_exchange import TTExchangeStrategy

        config = QRSamplerConfig()
        strategy = TTExchangeStrategy(vocab_size=8)
        result = strategy.compute_temperature(np.full(8, -np.inf), config)
        assert result.shannon_entropy == 0.0
        assert result.diagnostics["h_kept"] == 0.0
        assert result.diagnostics["n_kept"] == 0
        assert result.temperature == pytest.approx(config.tt_t_base)

    def test_tt_exchange_partial_mask_measures_kept_support(self) -> None:
        """Some -inf logits: kept measurement still runs on the finite set."""
        from qr_sampler.config import QRSamplerConfig
        from qr_sampler.temperature.tt_exchange import TTExchangeStrategy

        config = QRSamplerConfig()
        strategy = TTExchangeStrategy(vocab_size=6)
        logits = np.array([3.0, -np.inf, 2.0, 1.0, -np.inf, 0.0])
        result = strategy.compute_temperature(logits, config)
        # Historical behavior: H folds to 0.0 (0 * -inf = NaN guard), the
        # kept measurement runs on the finite support.
        assert result.shannon_entropy == 0.0
        assert result.diagnostics["n_kept"] >= 1
        assert result.diagnostics["h_kept"] >= 0.0
