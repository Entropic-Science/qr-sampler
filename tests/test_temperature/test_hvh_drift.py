"""Tests for the HVHDriftStrategy temperature strategy."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.hvh_drift import (
    _MIN_P_CLAMP,
    _TEMP_CLAMP,
    HVHDriftStrategy,
)
from qr_sampler.temperature.registry import TemperatureStrategyRegistry


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config carrying V6_HVD_R01_01 hyperparameters."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def strategy() -> HVHDriftStrategy:
    """Fresh HVHDriftStrategy instance with vocab_size=100."""
    return HVHDriftStrategy(vocab_size=100)


def _entropy_varentropy(logits: np.ndarray) -> tuple[float, float]:
    """Reference H/VH computation matching the strategy implementation."""
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    h = max(0.0, float(-np.sum(probs * log_probs)))
    vh = max(0.0, float(np.sum(probs * (-log_probs - h) ** 2)))
    return h, vh


class TestHVHDriftFirstCall:
    """First-call initialization: EMAs seeded with current values."""

    def test_first_call_initializes_emas_to_current_values(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        h, vh = _entropy_varentropy(logits)
        strategy.compute_temperature(logits, config)
        assert strategy.H_ema == pytest.approx(h)
        assert strategy.VH_ema == pytest.approx(vh)

    def test_first_token_drift_is_zero(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["d_h"] == 0.0
        assert result.diagnostics["d_vh"] == 0.0


class TestHVHDriftEMAUpdate:
    """EMA decay across multiple tokens."""

    def test_emas_decay_with_lambda(self, config: QRSamplerConfig) -> None:
        strategy = HVHDriftStrategy(vocab_size=10)
        logits_a = np.array([3.0, 2.0, 1.0, 0.0, -1.0])
        logits_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        h_a, vh_a = _entropy_varentropy(logits_a)
        h_b, vh_b = _entropy_varentropy(logits_b)

        strategy.compute_temperature(logits_a, config)
        # After first call, EMAs equal h_a / vh_a.
        assert strategy.H_ema == pytest.approx(h_a)
        assert strategy.VH_ema == pytest.approx(vh_a)

        strategy.compute_temperature(logits_b, config)
        lam = config.hvh_lambda_ema
        expected_h_ema = (1.0 - lam) * h_a + lam * h_b
        expected_vh_ema = (1.0 - lam) * vh_a + lam * vh_b
        assert strategy.H_ema == pytest.approx(expected_h_ema)
        assert strategy.VH_ema == pytest.approx(expected_vh_ema)


class TestHVHDriftFormulas:
    """Formula correctness against V6 reference math."""

    def test_temperature_formula_matches_v6_reference(
        self, config: QRSamplerConfig
    ) -> None:
        # First call: dH = dVH = 0, so the drift terms drop out.
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, vh = _entropy_varentropy(logits)
        expected_raw = (
            config.hvh_t_base
            + config.hvh_alpha_h * h
            + config.hvh_alpha_vh * vh
        )
        expected_temp = float(np.clip(expected_raw, *_TEMP_CLAMP))

        result = strategy.compute_temperature(logits, config)
        assert abs(result.temperature - expected_temp) < 1e-8

    def test_min_p_formula_matches_v6_reference(self, config: QRSamplerConfig) -> None:
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h, _ = _entropy_varentropy(logits)
        # First-call: dH = 0, so the drift term drops.
        expected_raw_min_p = config.hvh_min_p_base + config.hvh_kappa_h * h
        expected_min_p = float(np.clip(expected_raw_min_p, *_MIN_P_CLAMP))

        result = strategy.compute_temperature(logits, config)
        assert abs(result.diagnostics["min_p"] - expected_min_p) < 1e-8

    def test_temperature_uses_drift_after_first_call(
        self, config: QRSamplerConfig
    ) -> None:
        strategy = HVHDriftStrategy(vocab_size=10)
        logits_a = np.array([5.0, 0.0, 0.0, 0.0, 0.0])  # peaked
        logits_b = np.zeros(5)  # uniform — high H

        strategy.compute_temperature(logits_a, config)
        h_b, vh_b = _entropy_varentropy(logits_b)
        # After step a, EMAs hold h_a / vh_a.
        lam = config.hvh_lambda_ema
        h_a, vh_a = _entropy_varentropy(logits_a)
        h_ema_post = (1.0 - lam) * h_a + lam * h_b
        vh_ema_post = (1.0 - lam) * vh_a + lam * vh_b
        d_h = h_b - h_ema_post
        d_vh = vh_b - vh_ema_post

        expected_raw = (
            config.hvh_t_base
            + config.hvh_alpha_h * h_b
            + config.hvh_alpha_vh * vh_b
            + config.hvh_gamma_dh * d_h
            + config.hvh_delta_dvh * d_vh
        )
        expected_temp = float(np.clip(expected_raw, *_TEMP_CLAMP))
        result = strategy.compute_temperature(logits_b, config)
        assert abs(result.temperature - expected_temp) < 1e-8


class TestHVHDriftClamping:
    """Guardrail clamping to the V6 box."""

    def test_clamp_temperature_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            hvh_t_base=-50.0,  # force well below 0.3 lower clamp
        )
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[0])
        assert result.diagnostics["pre_clamp_temp"] < _TEMP_CLAMP[0]

    def test_clamp_temperature_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            hvh_t_base=50.0,  # force well above 2.2 upper clamp
        )
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[1])
        assert result.diagnostics["pre_clamp_temp"] > _TEMP_CLAMP[1]

    def test_clamp_min_p_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            hvh_min_p_base=-5.0,  # force below 0.0
            hvh_kappa_h=0.0,
            hvh_nu_dh=0.0,
        )
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[0])

    def test_clamp_min_p_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            hvh_min_p_base=5.0,  # force above 0.15
        )
        strategy = HVHDriftStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[1])


class TestHVHDriftDiagnostics:
    """Diagnostics surface for the logging subsystem."""

    def test_diagnostics_include_required_keys_first_call(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([1.0, 0.0, -1.0])
        result = strategy.compute_temperature(logits, config)
        for key in ("min_p", "varentropy", "h_ema", "vh_ema", "d_h", "d_vh"):
            assert key in result.diagnostics, f"missing diagnostic: {key}"

    def test_diagnostics_include_required_keys_subsequent_call(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits_a = np.array([1.0, 0.0, -1.0])
        logits_b = np.array([0.5, 0.5, 0.5])
        strategy.compute_temperature(logits_a, config)
        result = strategy.compute_temperature(logits_b, config)
        for key in ("min_p", "varentropy", "h_ema", "vh_ema", "d_h", "d_vh"):
            assert key in result.diagnostics

    def test_shannon_entropy_returned(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.0, -1.0])
        h, _ = _entropy_varentropy(logits)
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy == pytest.approx(h)


class TestHVHDriftRegistry:
    """Registry integration."""

    def test_registered_under_hvh_drift(self) -> None:
        klass = TemperatureStrategyRegistry.get("hvh_drift")
        assert klass is HVHDriftStrategy

    def test_vocab_size_injected_by_registry(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="hvh_drift",
        )
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=4096)
        assert isinstance(strategy, HVHDriftStrategy)
        assert strategy._vocab_size == 4096

    def test_built_instance_computes_temperature(
        self, config: QRSamplerConfig
    ) -> None:
        cfg = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="hvh_drift",
        )
        strategy = TemperatureStrategyRegistry.build(cfg, vocab_size=10)
        logits = np.arange(10, dtype=np.float64)
        result = strategy.compute_temperature(logits, config)
        assert _TEMP_CLAMP[0] <= result.temperature <= _TEMP_CLAMP[1]
        assert _MIN_P_CLAMP[0] <= result.diagnostics["min_p"] <= _MIN_P_CLAMP[1]


class TestHVHDriftIsolation:
    """Per-request state isolation (precondition for adapter lifecycle)."""

    def test_two_instances_have_independent_state(
        self, config: QRSamplerConfig
    ) -> None:
        a = HVHDriftStrategy(vocab_size=10)
        b = HVHDriftStrategy(vocab_size=10)
        # Drive instance A with 10 different distributions.
        rng = np.random.default_rng(seed=42)
        for _ in range(10):
            logits = rng.normal(size=10).astype(np.float64)
            a.compute_temperature(logits, config)
        # Instance B sees one fresh token.
        logits_b = np.zeros(10)
        b.compute_temperature(logits_b, config)
        # A has accumulated history, B is at first-call seed.
        assert a.H_ema != b.H_ema
        assert a.VH_ema != b.VH_ema
        assert a._first_call is False
        assert b._first_call is False  # b called once

    def test_result_is_frozen(
        self, strategy: HVHDriftStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([1.0, 0.0])
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]
