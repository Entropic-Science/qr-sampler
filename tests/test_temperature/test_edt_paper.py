"""Tests for the EDTPaperStrategy (faithful arXiv:2403.14541 form)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.edt_paper import _TEMP_CLAMP, EDTPaperStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


def _make_config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


def _entropy(logits: np.ndarray) -> float:
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    return max(0.0, float(-np.sum(probs * log_probs)))


class TestEDTPaperFormula:
    def test_matches_paper_reference_math(self) -> None:
        """T = T0 * N^(theta/H) at the paper's baseline hyperparameters."""
        strategy = EDTPaperStrategy()
        config = _make_config(edtp_t0=0.6, edtp_theta=0.1, edtp_n=0.8)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
        h = _entropy(logits)
        expected = 0.6 * 0.8 ** (0.1 / h)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(float(np.clip(expected, *_TEMP_CLAMP)))
        assert result.shannon_entropy == pytest.approx(h)

    def test_temperature_rises_with_entropy_toward_t0(self) -> None:
        """More entropy -> higher T, approaching T0 from below (never above)."""
        strategy = EDTPaperStrategy()
        config = _make_config(edtp_t0=2.0, edtp_theta=1.0, edtp_n=0.8)
        peaked = np.array([8.0, 0.0, 0.0, 0.0])
        mid = np.array([1.0, 0.5, 0.0, -0.5])
        flat = np.zeros(4)
        t_peaked = strategy.compute_temperature(peaked, config).temperature
        t_mid = strategy.compute_temperature(mid, config).temperature
        t_flat = strategy.compute_temperature(flat, config).temperature
        assert t_peaked < t_mid < t_flat
        assert t_flat < 2.0

    def test_confident_token_clamps_to_box_floor(self) -> None:
        """H -> 0 sends the formula's T -> 0; the guardrail floor holds at 0.3."""
        strategy = EDTPaperStrategy()
        config = _make_config(edtp_t0=0.6, edtp_theta=0.1, edtp_n=0.8)
        logits = np.full(100, -60.0)
        logits[0] = 60.0
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[0])
        assert result.diagnostics["pre_clamp_temp"] < _TEMP_CLAMP[0]

    def test_delta_distribution_does_not_divide_by_zero(self) -> None:
        strategy = EDTPaperStrategy()
        config = _make_config(edtp_t0=0.6)
        logits = np.array([1e9, -1e9, -1e9])
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[0])

    def test_no_min_p_diagnostic_emitted(self) -> None:
        """The paper pairs no truncation: selector-level min_p_base applies."""
        strategy = EDTPaperStrategy()
        result = strategy.compute_temperature(np.zeros(6), _make_config())
        assert "min_p" not in result.diagnostics
        assert result.diagnostics["strategy"] == "edt_paper"


class TestEDTPaperStaticClone:
    def test_theta_zero_is_exact_static_clone(self) -> None:
        """FR-8.5: theta=0 => N^0 = 1 => T = t0 exactly, on any distribution."""
        strategy = EDTPaperStrategy()
        config = _make_config(edtp_t0=1.15, edtp_theta=0.0)
        for logits in (np.zeros(5), np.array([4.0, 1.0, -3.0]), np.array([9.0, -9.0])):
            result = strategy.compute_temperature(logits, config)
            assert result.temperature == pytest.approx(1.15)


class TestEDTPaperConfigBounds:
    def test_paper_defaults(self) -> None:
        config = _make_config()
        assert config.edtp_t0 == pytest.approx(0.6)
        assert config.edtp_theta == pytest.approx(0.1)
        assert config.edtp_n == pytest.approx(0.8)

    def test_t0_capped_at_guardrail_ceiling(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(edtp_t0=2.3)

    def test_n_must_be_in_open_unit_interval(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(edtp_n=1.0)
        with pytest.raises(ValidationError):
            _make_config(edtp_n=0.0)

    def test_theta_must_be_nonnegative(self) -> None:
        with pytest.raises(ValidationError):
            _make_config(edtp_theta=-0.1)


class TestEDTPaperRegistry:
    def test_registered_and_buildable(self) -> None:
        config = _make_config(temperature_strategy="edt_paper")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=32)
        assert isinstance(strategy, EDTPaperStrategy)
