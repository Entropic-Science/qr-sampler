"""Tests for the TTExchangeStrategy temperature strategy (V6 §7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.registry import TemperatureStrategyRegistry
from qr_sampler.temperature.tt_exchange import (
    _MIN_P_CLAMP,
    _TEMP_CLAMP,
    TTExchangeStrategy,
)


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config carrying the V6 §7.3 predicted defaults."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def strategy() -> TTExchangeStrategy:
    """Fresh TTExchangeStrategy instance with vocab_size=100."""
    return TTExchangeStrategy(vocab_size=100)


def _entropy(logits: np.ndarray) -> float:
    """Reference H computation matching the strategy implementation."""
    shifted = logits - np.max(logits)
    log_sum_exp = float(np.log(np.sum(np.exp(shifted))))
    log_probs = shifted - log_sum_exp
    probs = np.exp(log_probs)
    return max(0.0, float(-np.sum(probs * log_probs)))


def _kept_entropy(logits: np.ndarray, min_p: float) -> float:
    """Reference H_kept: entropy of the min-p-truncated, renormalised probs."""
    shifted = logits - np.max(logits)
    probs = np.exp(shifted) / np.sum(np.exp(shifted))
    mask = probs >= min_p * probs.max()
    kept = probs[mask]
    kept = kept / kept.sum()
    return max(0.0, float(-np.sum(kept * np.log(kept))))


class TestTTExchangeFormulas:
    """Formula correctness against the V6 §7.3 reference math."""

    def test_min_p_formula_matches_v6_reference(
        self, strategy: TTExchangeStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h = _entropy(logits)
        expected_raw = config.tt_min_p_base + config.tt_min_p_scale * h
        expected_min_p = float(np.clip(expected_raw, *_MIN_P_CLAMP))
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(expected_min_p)

    def test_temperature_formula_matches_v6_reference(
        self, strategy: TTExchangeStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        h = _entropy(logits)
        min_p = float(np.clip(config.tt_min_p_base + config.tt_min_p_scale * h, *_MIN_P_CLAMP))
        h_kept = _kept_entropy(logits, min_p)
        expected_raw = config.tt_t_base + config.tt_gamma * max(0.0, h - h_kept)
        expected_temp = float(np.clip(expected_raw, *_TEMP_CLAMP))

        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(expected_temp)
        assert result.diagnostics["h_kept"] == pytest.approx(h_kept)

    def test_zero_min_p_means_no_entropy_removed(self) -> None:
        """With min_p forced to 0, H_kept == H and T holds exactly T_base."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_min_p_base=0.0,
            tt_min_p_scale=0.0,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["entropy_removed"] == 0.0
        assert result.temperature == pytest.approx(config.tt_t_base)

    def test_entropy_removed_is_never_negative(self, config: QRSamplerConfig) -> None:
        """max(0, H - H_kept) floor: negative gamma can never invert the sign."""
        strategy = TTExchangeStrategy(vocab_size=10)
        rng = np.random.default_rng(seed=7)
        for _ in range(20):
            logits = rng.normal(scale=3.0, size=50)
            result = strategy.compute_temperature(logits, config)
            assert result.diagnostics["entropy_removed"] >= 0.0


class TestTTExchangeMonotonicity:
    """Monotonicity/limit pins mirroring the frozen-gate style."""

    def test_heavier_truncation_raises_temperature(self) -> None:
        """Larger min_p removes more entropy => T is monotone non-decreasing.

        Uses a fixed flat-ish distribution and sweeps tt_min_p_base upward
        (positive gamma): the entropy removed grows with the threshold, so
        the pre-clamp temperature must be non-decreasing.
        """
        logits = np.linspace(0.0, 2.0, 40)
        strategy = TTExchangeStrategy(vocab_size=40)
        temps = []
        for base in (0.0, 0.01, 0.03, 0.06, 0.10, 0.15):
            config = QRSamplerConfig(
                _env_file=None,  # type: ignore[call-arg]
                tt_min_p_base=base,
                tt_min_p_scale=0.0,
            )
            result = strategy.compute_temperature(logits, config)
            temps.append(result.diagnostics["pre_clamp_temp"])
        assert temps == sorted(temps)

    def test_gamma_zero_pins_t_base(self) -> None:
        """gamma = 0 disconnects truncation from temperature entirely."""
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_gamma=0.0,
            tt_min_p_base=0.05,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.linspace(0.0, 1.0, 30)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(config.tt_t_base)

    def test_peaked_distribution_stays_near_t_base(self, config: QRSamplerConfig) -> None:
        """A near-one-hot distribution loses ~no entropy to truncation."""
        logits = np.full(50, -20.0)
        logits[0] = 20.0
        strategy = TTExchangeStrategy(vocab_size=50)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(config.tt_t_base, abs=1e-6)


class TestTTExchangeClamping:
    """Guardrail clamping to the V6 box."""

    def test_clamp_temperature_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_t_base=50.0,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[1])
        assert result.diagnostics["pre_clamp_temp"] > _TEMP_CLAMP[1]

    def test_clamp_temperature_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_t_base=-50.0,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.temperature == pytest.approx(_TEMP_CLAMP[0])

    def test_clamp_min_p_to_upper_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_min_p_base=5.0,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[1])

    def test_clamp_min_p_to_lower_bound(self) -> None:
        config = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            tt_min_p_base=-5.0,
            tt_min_p_scale=0.0,
        )
        strategy = TTExchangeStrategy(vocab_size=10)
        logits = np.zeros(10)
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["min_p"] == pytest.approx(_MIN_P_CLAMP[0])


class TestTTExchangeStatelessness:
    """The strategy is stateless per token (no cross-token leakage)."""

    def test_same_logits_same_result_regardless_of_history(self, config: QRSamplerConfig) -> None:
        a = TTExchangeStrategy(vocab_size=10)
        b = TTExchangeStrategy(vocab_size=10)
        rng = np.random.default_rng(seed=42)
        # Drive instance A with 10 unrelated distributions first.
        for _ in range(10):
            a.compute_temperature(rng.normal(size=10), config)
        probe = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        result_a = a.compute_temperature(probe, config)
        result_b = b.compute_temperature(probe, config)
        assert result_a.temperature == result_b.temperature
        assert result_a.diagnostics["min_p"] == result_b.diagnostics["min_p"]


class TestTTExchangeDiagnostics:
    """Diagnostics surface for the logging subsystem."""

    def test_diagnostics_include_required_keys(
        self, strategy: TTExchangeStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([1.0, 0.0, -1.0])
        result = strategy.compute_temperature(logits, config)
        for key in ("min_p", "h_kept", "entropy_removed", "n_kept", "pre_clamp_temp"):
            assert key in result.diagnostics, f"missing diagnostic: {key}"
        assert result.diagnostics["strategy"] == "tt_exchange"

    def test_shannon_entropy_returned(
        self, strategy: TTExchangeStrategy, config: QRSamplerConfig
    ) -> None:
        logits = np.array([2.0, 1.0, 0.0, -1.0])
        result = strategy.compute_temperature(logits, config)
        assert result.shannon_entropy == pytest.approx(_entropy(logits))

    def test_result_is_frozen(self, strategy: TTExchangeStrategy, config: QRSamplerConfig) -> None:
        logits = np.array([1.0, 0.0])
        result = strategy.compute_temperature(logits, config)
        with pytest.raises(AttributeError):
            result.temperature = 99.0  # type: ignore[misc]


class TestTTExchangeRegistry:
    """Registry integration."""

    def test_registered_under_tt_exchange(self) -> None:
        klass = TemperatureStrategyRegistry.get("tt_exchange")
        assert klass is TTExchangeStrategy

    def test_built_instance_computes_temperature(self, config: QRSamplerConfig) -> None:
        cfg = QRSamplerConfig(
            _env_file=None,  # type: ignore[call-arg]
            temperature_strategy="tt_exchange",
        )
        strategy = TemperatureStrategyRegistry.build(cfg, vocab_size=10)
        assert isinstance(strategy, TTExchangeStrategy)
        logits = np.arange(10, dtype=np.float64)
        result = strategy.compute_temperature(logits, config)
        assert _TEMP_CLAMP[0] <= result.temperature <= _TEMP_CLAMP[1]
        assert _MIN_P_CLAMP[0] <= result.diagnostics["min_p"] <= _MIN_P_CLAMP[1]
