"""Tests for the GDTStrategy (Gaussian Dynamic Temperature)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.gdt import _MIN_P_CLAMP, _TEMP_CLAMP, GDTStrategy
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


def _reference(logits: np.ndarray, vocab_size: int, config: QRSamplerConfig) -> tuple[float, float]:
    """Legacy GDTTempProcessor math (createmp-evalsuite V5) + guardrail box."""
    h, vh = _entropy_varentropy(logits)
    h_norm = min(1.0, max(0.0, h / math.log(vocab_size)))
    vh_norm = 1.0 - math.exp(-vh / config.gdt_lambda_vh)
    bell = config.gdt_t_peak * math.exp(
        -((h_norm - config.gdt_mu) ** 2) / (2 * config.gdt_sigma**2)
    )
    vh_boost = config.gdt_alpha * vh_norm * (1.0 - h_norm)
    raw_t = config.gdt_t_base + bell + vh_boost
    excess = max(0.0, raw_t - config.gdt_t_base)
    if config.gdt_t_peak > 0:
        raw_mp = config.gdt_min_p_base + config.gdt_min_p_scale * excess / config.gdt_t_peak
    else:
        raw_mp = config.gdt_min_p_base
    return float(np.clip(raw_t, *_TEMP_CLAMP)), float(np.clip(raw_mp, *_MIN_P_CLAMP))


class TestGDTFormulas:
    def test_matches_v5_reference_math(self) -> None:
        strategy = GDTStrategy(vocab_size=8)
        config = _make_config()
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        expected_t, expected_mp = _reference(logits, 8, config)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(expected_t)
        assert result.diagnostics["min_p"] == pytest.approx(expected_mp)

    def test_shannon_entropy_returned(self) -> None:
        strategy = GDTStrategy(vocab_size=5)
        logits = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        h, _ = _entropy_varentropy(logits)
        result = strategy.compute_temperature(logits, _make_config())
        assert result.shannon_entropy == pytest.approx(h)

    def test_min_p_is_base_at_no_boost(self) -> None:
        """A peaked distribution far from mu gets (almost) no boost."""
        config = _make_config(gdt_t_base=0.8, gdt_t_peak=0.0, gdt_alpha=0.0, gdt_min_p_base=0.02)
        strategy = GDTStrategy(vocab_size=100)
        logits = np.full(100, -10.0)
        logits[0] = 30.0
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(0.02)

    def test_varentropy_boost_tapers_with_entropy(self) -> None:
        """The vh_boost term carries the (1 - H_norm) taper."""
        strategy = GDTStrategy(vocab_size=4)
        config = _make_config(gdt_alpha=1.0)
        uniform = np.zeros(4)  # H_norm == 1.0 exactly -> taper kills boost
        result = strategy.compute_temperature(uniform, config)
        assert result.diagnostics["vh_boost"] == pytest.approx(0.0)


class TestGDTClamping:
    def test_temperature_clamped_to_box(self) -> None:
        config = _make_config(gdt_t_base=0.05, gdt_t_peak=0.0, gdt_alpha=0.0)
        strategy = GDTStrategy(vocab_size=5)
        result = strategy.compute_temperature(np.array([9.0, 0.0, 0.0, 0.0, 0.0]), config)
        assert result.temperature == _TEMP_CLAMP[0]

    def test_min_p_clamped_to_box(self) -> None:
        config = _make_config(gdt_min_p_base=0.5)
        strategy = GDTStrategy(vocab_size=5)
        result = strategy.compute_temperature(np.zeros(5), config)
        assert result.diagnostics["min_p"] == _MIN_P_CLAMP[1]

    def test_t_peak_hard_cap_enforced_by_config(self) -> None:
        with pytest.raises(Exception, match=r"less than or equal to 1\.5"):
            _make_config(gdt_t_peak=1.6)


class TestGDTStaticClone:
    def test_clone_reduces_to_fixed_t_and_min_p(self) -> None:
        """gdt_t_peak=0 + gdt_alpha=0: constant T/min_p on every shape."""
        config = _make_config(gdt_t_peak=0.0, gdt_alpha=0.0, gdt_t_base=1.2, gdt_min_p_base=0.012)
        strategy = GDTStrategy(vocab_size=6)
        for logits in (
            np.zeros(6),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 1.9, 1.8, -3.0, -4.0, -5.0]),
        ):
            result = strategy.compute_temperature(logits, config)
            assert result.temperature == pytest.approx(1.2)
            assert result.diagnostics["min_p"] == pytest.approx(0.012)
            assert "transformed_logits" not in result.diagnostics


class TestGDTRegistry:
    def test_registered_under_gdt(self) -> None:
        assert TemperatureStrategyRegistry.get("gdt") is GDTStrategy

    def test_vocab_size_injected_by_registry(self) -> None:
        config = _make_config(temperature_strategy="gdt")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=50)
        assert isinstance(strategy, GDTStrategy)

    def test_rejects_degenerate_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            GDTStrategy(vocab_size=1)
