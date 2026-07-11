"""Tests for the DynaTempStrategy."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.dynatemp import _TEMP_CLAMP, DynaTempStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


def _make_config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


def _entropy(logits: np.ndarray) -> float:
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    return max(0.0, float(-np.sum(probs * log_probs)))


class TestDynaTempFormulas:
    def test_matches_llamacpp_reference_math(self) -> None:
        strategy = DynaTempStrategy(vocab_size=8)
        config = _make_config(dynatemp_t_center=1.2, dynatemp_t_range=0.4, dynatemp_exponent=1.3)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h_norm = min(1.0, max(0.0, _entropy(logits) / math.log(8)))
        expected = 1.2 - 0.4 + 2 * 0.4 * h_norm**1.3
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(float(np.clip(expected, *_TEMP_CLAMP)))

    def test_uniform_hits_top_of_range(self) -> None:
        strategy = DynaTempStrategy(vocab_size=4)
        config = _make_config(dynatemp_t_center=1.0, dynatemp_t_range=0.5)
        result = strategy.compute_temperature(np.zeros(4), config)
        assert result.temperature == pytest.approx(1.5)

    def test_peaked_hits_bottom_of_range(self) -> None:
        strategy = DynaTempStrategy(vocab_size=100)
        config = _make_config(dynatemp_t_center=1.0, dynatemp_t_range=0.5)
        logits = np.full(100, -60.0)
        logits[0] = 60.0
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(0.5, abs=1e-6)

    def test_constant_min_p_published(self) -> None:
        strategy = DynaTempStrategy(vocab_size=5)
        config = _make_config(dynatemp_min_p=0.12)
        result = strategy.compute_temperature(np.zeros(5), config)
        assert result.diagnostics["min_p"] == pytest.approx(0.12)

    def test_abl_dyn_08_recipe_is_reachable(self) -> None:
        """Hot + hard truncation (assessment §3.3) validates and computes.

        The recipe's parameters are accepted; at maximum entropy the raw
        top of the range (2.675) exceeds the repo guardrail ceiling and
        clamps to 2.2 — the hard box wins over any recipe by design.
        """
        config = _make_config(dynatemp_t_center=1.875, dynatemp_t_range=0.80, dynatemp_min_p=0.12)
        strategy = DynaTempStrategy(vocab_size=4)
        result = strategy.compute_temperature(np.zeros(4), config)  # H_norm = 1
        assert result.diagnostics["pre_clamp_temp"] == pytest.approx(1.875 + 0.80)
        assert result.temperature == _TEMP_CLAMP[1]
        assert result.diagnostics["min_p"] == pytest.approx(0.12)
        # At center entropy (H_norm^exp = 0.5) the recipe runs unclamped.
        mid = np.full(4, -40.0)
        mid[:2] = 0.0  # 2 of 4 tokens active -> H_norm = ln2/ln4 = 0.5
        result_mid = strategy.compute_temperature(mid, config)
        assert result_mid.temperature == pytest.approx(1.875, abs=1e-6)


class TestDynaTempClamping:
    def test_temperature_upper_clamp(self) -> None:
        config = _make_config(dynatemp_t_center=2.0, dynatemp_t_range=0.5)
        strategy = DynaTempStrategy(vocab_size=4)
        result = strategy.compute_temperature(np.zeros(4), config)
        assert result.temperature == _TEMP_CLAMP[1]

    def test_min_p_field_cap(self) -> None:
        with pytest.raises(Exception, match=r"less than or equal to 0\.15"):
            _make_config(dynatemp_min_p=0.2)


class TestDynaTempStaticClone:
    def test_clone_reduces_to_fixed_t_and_min_p(self) -> None:
        """dynatemp_t_range=0: constant T/min_p on every shape."""
        config = _make_config(dynatemp_t_range=0.0, dynatemp_t_center=1.31, dynatemp_min_p=0.026)
        strategy = DynaTempStrategy(vocab_size=6)
        for logits in (
            np.zeros(6),
            np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([2.0, 1.9, 1.8, -3.0, -4.0, -5.0]),
        ):
            result = strategy.compute_temperature(logits, config)
            assert result.temperature == pytest.approx(1.31)
            assert result.diagnostics["min_p"] == pytest.approx(0.026)
            assert "transformed_logits" not in result.diagnostics


class TestDynaTempRegistry:
    def test_registered_under_dynatemp(self) -> None:
        assert TemperatureStrategyRegistry.get("dynatemp") is DynaTempStrategy

    def test_built_instance_computes(self) -> None:
        config = _make_config(temperature_strategy="dynatemp")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=50)
        result = strategy.compute_temperature(np.zeros(50), config)
        assert result.temperature > 0

    def test_rejects_degenerate_vocab(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            DynaTempStrategy(vocab_size=1)
