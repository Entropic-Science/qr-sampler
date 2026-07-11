"""Tests for the MixTemperaturesStrategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.mix_temperatures import MixTemperaturesStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


def _make_config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


def _softmax(logits: np.ndarray, t: float) -> np.ndarray:
    scaled = logits / t
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    result: np.ndarray = exp / exp.sum()
    return result


def _entropy_varentropy(logits: np.ndarray) -> tuple[float, float]:
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    h = max(0.0, float(-np.sum(probs * log_probs)))
    vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))
    return h, vh


class TestMixFormulas:
    def test_transformed_logits_encode_the_mixture(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=8)
        config = _make_config(mix_t_cool=0.7, mix_t_hot=1.5)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, vh = _entropy_varentropy(logits)
        gate = config.mix_gate_a * (h - config.mix_gate_b) + config.mix_gate_c * (
            vh - config.mix_gate_d
        )
        alpha = 1.0 / (1.0 + math.exp(-gate))
        expected_mix = alpha * _softmax(logits, 1.5) + (1 - alpha) * _softmax(logits, 0.7)

        result = strategy.compute_temperature(logits, config)
        transformed = result.diagnostics["transformed_logits"]
        recovered = np.exp(transformed) / np.exp(transformed).sum()
        np.testing.assert_allclose(recovered, expected_mix, rtol=1e-10)
        assert result.diagnostics["alpha"] == pytest.approx(alpha)

    def test_selector_temperature_is_one(self) -> None:
        """The mixture already encodes both arms; the selector must not rescale."""
        strategy = MixTemperaturesStrategy(vocab_size=5)
        result = strategy.compute_temperature(np.zeros(5), _make_config())
        assert result.temperature == 1.0

    def test_t_mix_is_convex_combination(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=5)
        config = _make_config(mix_t_cool=0.7, mix_t_hot=1.5)
        result = strategy.compute_temperature(np.array([3.0, 1.0, 0.0, -1.0, -3.0]), config)
        alpha = result.diagnostics["alpha"]
        assert result.diagnostics["t_mix"] == pytest.approx(alpha * 1.5 + (1 - alpha) * 0.7)

    def test_gate_routes_hot_on_high_entropy(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=1000)
        config = _make_config()
        peaked = np.full(1000, -40.0)
        peaked[0] = 40.0
        alpha_peaked = strategy.compute_temperature(peaked, config).diagnostics["alpha"]
        alpha_uniform = strategy.compute_temperature(np.zeros(1000), config).diagnostics["alpha"]
        assert alpha_uniform > alpha_peaked

    def test_masked_tokens_stay_masked(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=4)
        logits = np.array([1.0, 0.0, -np.inf, -1.0])
        result = strategy.compute_temperature(logits, _make_config())
        assert result.diagnostics["transformed_logits"][2] == -np.inf

    def test_shannon_entropy_describes_raw_distribution(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=5)
        logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        h, _ = _entropy_varentropy(logits)
        result = strategy.compute_temperature(logits, _make_config())
        assert result.shannon_entropy == pytest.approx(h)

    def test_extreme_gate_inputs_do_not_overflow(self) -> None:
        strategy = MixTemperaturesStrategy(vocab_size=5)
        config = _make_config(mix_gate_a=1e6, mix_gate_b=-1e6)
        result = strategy.compute_temperature(np.zeros(5), config)
        assert result.diagnostics["alpha"] == pytest.approx(1.0)


class TestMixBounds:
    def test_t_hot_reaches_guardrail_ceiling(self) -> None:
        """Re-widened bounds (assessment §8.2 item 6): 2.2 is legal."""
        config = _make_config(mix_t_hot=2.2)
        assert config.mix_t_hot == 2.2

    def test_t_hot_beyond_ceiling_rejected(self) -> None:
        with pytest.raises(Exception, match=r"less than or equal to 2\.2"):
            _make_config(mix_t_hot=2.3)

    def test_t_cool_below_floor_rejected(self) -> None:
        with pytest.raises(Exception, match=r"greater than or equal to 0\.3"):
            _make_config(mix_t_cool=0.2)


class TestMixStaticClone:
    def test_equal_arms_reduce_to_fixed_temperature(self) -> None:
        """mix_t_cool == mix_t_hot == T: p_mix == softmax(l/T) exactly."""
        config = _make_config(mix_t_cool=1.3, mix_t_hot=1.3, mix_min_p=0.01)
        strategy = MixTemperaturesStrategy(vocab_size=8)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        result = strategy.compute_temperature(logits, config)
        transformed = result.diagnostics["transformed_logits"]
        recovered = np.exp(transformed) / np.exp(transformed).sum()
        np.testing.assert_allclose(recovered, _softmax(logits, 1.3), rtol=1e-12)
        assert result.diagnostics["min_p"] == pytest.approx(0.01)
        assert result.diagnostics["t_mix"] == pytest.approx(1.3)


class TestMixRegistry:
    def test_registered_under_mix_temperatures(self) -> None:
        assert TemperatureStrategyRegistry.get("mix_temperatures") is MixTemperaturesStrategy

    def test_built_instance_computes(self) -> None:
        config = _make_config(temperature_strategy="mix_temperatures")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=50)
        result = strategy.compute_temperature(np.zeros(50), config)
        assert "transformed_logits" in result.diagnostics
