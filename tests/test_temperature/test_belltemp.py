"""Tests for the BellTempStrategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.belltemp import _MIN_P_CLAMP, _TEMP_CLAMP, BellTempStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


def _make_config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


def _entropy_varentropy(logits: np.ndarray) -> tuple[float, float]:
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    h = max(0.0, float(-np.sum(probs * log_probs)))
    vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))
    return h, vh


class TestBellTempFormulas:
    def test_matches_v5_reference_math(self) -> None:
        strategy = BellTempStrategy(vocab_size=8)
        config = _make_config(belltemp_vh_weight=0.3)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, vh = _entropy_varentropy(logits)
        h_norm = min(1.0, max(0.0, h / math.log(8)))
        vh_norm = 1.0 - math.exp(-vh / config.belltemp_lambda_vh)
        bell = config.belltemp_t_peak * math.exp(
            -((h_norm - config.belltemp_mu) ** 2) / (2 * config.belltemp_sigma**2)
        )
        expected = float(np.clip(config.belltemp_t_base + bell + 0.3 * vh_norm, *_TEMP_CLAMP))
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(expected)

    def test_non_monotonic_bell_shape(self) -> None:
        """Mid-entropy tokens run hotter than both extremes."""
        strategy = BellTempStrategy(vocab_size=1000)
        config = _make_config(belltemp_t_base=0.7, belltemp_t_peak=0.6, belltemp_mu=0.5)
        peaked = np.full(1000, -40.0)
        peaked[0] = 40.0
        uniform = np.zeros(1000)
        # Mid-entropy: ~sqrt(1000) tokens active -> H_norm ~ 0.5.
        mid = np.full(1000, -40.0)
        mid[:32] = 0.0
        t_peaked = strategy.compute_temperature(peaked, config).temperature
        t_mid = strategy.compute_temperature(mid, config).temperature
        t_uniform = strategy.compute_temperature(uniform, config).temperature
        assert t_mid > t_peaked
        assert t_mid > t_uniform

    def test_adaptive_min_p_couples_to_temperature(self) -> None:
        config = _make_config(
            belltemp_min_p_base=0.01, belltemp_min_p_scale=0.05, belltemp_t_base=1.0
        )
        strategy = BellTempStrategy(vocab_size=8)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        result = strategy.compute_temperature(logits, config)
        t_frac = (result.temperature - _TEMP_CLAMP[0]) / (_TEMP_CLAMP[1] - _TEMP_CLAMP[0])
        expected = float(np.clip(0.01 + 0.05 * t_frac, *_MIN_P_CLAMP))
        assert result.diagnostics["min_p"] == pytest.approx(expected)

    def test_t_peak_hard_cap_enforced_by_config(self) -> None:
        with pytest.raises(Exception, match=r"less than or equal to 1\.5"):
            _make_config(belltemp_t_peak=1.6)


class TestBellTempStaticClone:
    def test_clone_reduces_to_fixed_t_and_min_p(self) -> None:
        """t_peak=0 + vh_weight=0 + min_p_scale=0: constant outputs."""
        config = _make_config(
            belltemp_t_peak=0.0,
            belltemp_vh_weight=0.0,
            belltemp_min_p_scale=0.0,
            belltemp_t_base=1.1,
            belltemp_min_p_base=0.02,
        )
        strategy = BellTempStrategy(vocab_size=6)
        for logits in (
            np.zeros(6),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 1.9, 1.8, -3.0, -4.0, -5.0]),
        ):
            result = strategy.compute_temperature(logits, config)
            assert result.temperature == pytest.approx(1.1)
            assert result.diagnostics["min_p"] == pytest.approx(0.02)
            assert "transformed_logits" not in result.diagnostics


class TestBellTempRegistry:
    def test_registered_under_belltemp(self) -> None:
        assert TemperatureStrategyRegistry.get("belltemp") is BellTempStrategy

    def test_built_instance_computes(self) -> None:
        config = _make_config(temperature_strategy="belltemp")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=50)
        result = strategy.compute_temperature(np.zeros(50), config)
        assert result.temperature >= _TEMP_CLAMP[0]

    def test_rejects_degenerate_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            BellTempStrategy(vocab_size=1)
