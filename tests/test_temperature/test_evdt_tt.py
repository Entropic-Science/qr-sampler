"""Tests for the EVDTTTStrategy temperature strategy (V6 §7.1).

Also carries the FR-18 guardrail pin: no ``invert_vh`` field exists
anywhere in the config model — stronger than a runtime guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.config.model import ALL_FIELDS
from qr_sampler.temperature.evdt_tt import (
    _MIN_P_CLAMP,
    _TEMP_CLAMP,
    EVDTTTStrategy,
)
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config carrying the V6 §7.1 predicted defaults."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def strategy() -> EVDTTTStrategy:
    """Fresh EVDTTTStrategy instance with vocab_size=100."""
    return EVDTTTStrategy(vocab_size=100)


def _entropy_varentropy(logits: np.ndarray) -> tuple[float, float]:
    """Reference H/VH computation matching the strategy implementation."""
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    h = max(0.0, float(-np.sum(probs * log_probs)))
    vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))
    return h, vh


class TestInvertVhGuardrail:
    """FR-18: the invert_vh footgun does not exist as a field at all."""

    def test_no_invert_vh_field_exists(self) -> None:
        """V3/V4 finding: double-inversion collapses coherence. The field
        is absent by construction, not guarded at runtime."""
        assert "invert_vh" not in ALL_FIELDS
        assert not any("invert_vh" in f for f in ALL_FIELDS)


class TestEVDTTTFormulas:
    """Formula correctness against the V6 §7.1 reference math."""

    def test_temperature_formula_matches_v6_reference(
        self, strategy: EVDTTTStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, vh = _entropy_varentropy(logits)
        expected_raw = config.evdt_t_base + config.evdt_alpha * h + config.evdt_beta * vh
        expected_temp = float(np.clip(expected_raw, *_TEMP_CLAMP))
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(expected_temp)

    def test_min_p_formula_matches_v6_reference(
        self, strategy: EVDTTTStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, vh = _entropy_varentropy(logits)
        expected_raw = (
            config.evdt_min_p_base + config.evdt_min_p_scale * h + config.evdt_min_p_vh * vh
        )
        expected_min_p = float(np.clip(expected_raw, *_MIN_P_CLAMP))
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(expected_min_p)


class TestEVDTTTMonotonicityAndLimits:
    """Monotonicity/limit pins mirroring the frozen-gate style."""

    def test_temperature_monotone_in_entropy_with_positive_alpha(self) -> None:
        """With beta = 0 and alpha > 0, higher-H distributions get higher T."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_beta=0.0,
        )
        strategy = EVDTTTStrategy(vocab_size=20)
        temps = []
        # Sweep from peaked (low H) to uniform (high H).
        for scale in (8.0, 4.0, 2.0, 1.0, 0.5, 0.0):
            logits = np.linspace(scale, 0.0, 20)
            result = strategy.compute_temperature(logits, config)
            temps.append(result.diagnostics["pre_clamp_temp"])
        assert temps == sorted(temps)

    def test_zero_coefficients_pin_bases(self) -> None:
        """alpha = beta = scale = vh = 0 pins T_base and min_p_base exactly."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_alpha=0.0,
            evdt_beta=0.0,
            evdt_min_p_scale=0.0,
            evdt_min_p_vh=0.0,
        )
        strategy = EVDTTTStrategy(vocab_size=10)
        logits = np.linspace(0.0, 3.0, 30)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(config.evdt_t_base)
        assert result.diagnostics["min_p"] == pytest.approx(config.evdt_min_p_base)

    def test_one_hot_distribution_yields_base_values(self, config: QRSamplerConfig) -> None:
        """H = VH = 0 for a (near) one-hot distribution => T = T_base."""
        logits = np.full(50, -60.0)
        logits[0] = 60.0
        strategy = EVDTTTStrategy(vocab_size=50)
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy == pytest.approx(0.0, abs=1e-9)
        assert result.temperature == pytest.approx(config.evdt_t_base, abs=1e-6)
        assert result.diagnostics["min_p"] == pytest.approx(config.evdt_min_p_base, abs=1e-6)


class TestEVDTTTClamping:
    """Guardrail clamping to the V6 box."""

    def test_clamp_temperature_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_t_base=50.0,
        )
        strategy = EVDTTTStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[1])
        assert result.diagnostics["pre_clamp_temp"] > _TEMP_CLAMP[1]

    def test_clamp_temperature_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_t_base=-50.0,
        )
        strategy = EVDTTTStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[0])

    def test_clamp_min_p_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_min_p_base=5.0,
        )
        strategy = EVDTTTStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[1])

    def test_clamp_min_p_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            evdt_min_p_base=-5.0,
            evdt_min_p_scale=0.0,
            evdt_min_p_vh=0.0,
        )
        strategy = EVDTTTStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[0])


class TestEVDTTTStatelessness:
    """The strategy is stateless per token (no cross-token leakage)."""

    def test_same_logits_same_result_regardless_of_history(self, config: QRSamplerConfig) -> None:
        a = EVDTTTStrategy(vocab_size=10)
        b = EVDTTTStrategy(vocab_size=10)
        rng = np.random.default_rng(seed=42)
        for _ in range(10):
            a.compute_temperature(rng.normal(size=10), config)
        probe = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        result_a = a.compute_temperature(probe, config)
        result_b = b.compute_temperature(probe, config)
        assert result_a.temperature == result_b.temperature
        assert result_a.diagnostics["min_p"] == result_b.diagnostics["min_p"]


class TestEVDTTTDiagnostics:
    """Diagnostics surface for the logging subsystem."""

    def test_diagnostics_include_required_keys(
        self, strategy: EVDTTTStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([1.0, 0.0, -1.0])
        result = strategy.compute_temperature(logits, config)
        for key in ("min_p", "varentropy", "pre_clamp_temp", "pre_clamp_min_p"):
            assert key in result.diagnostics, f"missing diagnostic: {key}"
        assert result.diagnostics["strategy"] == "evdt_tt"

    def test_shannon_entropy_returned(
        self, strategy: EVDTTTStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.0, -1.0])
        h, _ = _entropy_varentropy(logits)
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy == pytest.approx(h)

    def test_result_is_frozen(self, strategy: EVDTTTStrategy, config: QRSamplerConfig) -> None:
        logits = np.array([1.0, 0.0])
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]


class TestEVDTTTRegistry:
    """Registry integration."""

    def test_registered_under_evdt_tt(self) -> None:
        klass = TemperatureStrategyRegistry.get("evdt_tt")
        assert klass is EVDTTTStrategy

    def test_built_instance_computes_temperature(self, config: QRSamplerConfig) -> None:
        cfg = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="evdt_tt",
        )
        strategy = TemperatureStrategyRegistry.build(cfg, vocab_size=10)
        assert isinstance(strategy, EVDTTTStrategy)
        logits = np.arange(10, dtype=np.float64)
        result = strategy.compute_temperature(logits, config)
        assert _TEMP_CLAMP[0] <= result.temperature <= _TEMP_CLAMP[1]
        assert _MIN_P_CLAMP[0] <= result.diagnostics["min_p"] <= _MIN_P_CLAMP[1]
