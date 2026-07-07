"""Tests for the ZScoreMeanAmplifier and amplification registry."""

from __future__ import annotations

import math

import pytest

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import SignalAmplificationError


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config for amplification tests."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def amplifier(config: QRSamplerConfig) -> ZScoreMeanAmplifier:
    """Default ZScoreMeanAmplifier."""
    return ZScoreMeanAmplifier(config)


class TestZScoreMeanAmplifier:
    """Tests for ZScoreMeanAmplifier."""

    def test_known_value_unbiased(self, amplifier: ZScoreMeanAmplifier) -> None:
        """Bytes with mean exactly 127.5 should produce u ≈ 0.5."""
        # Alternate 0 and 255 to get exact mean of 127.5.
        raw = bytes([0, 255] * 1000)
        result = amplifier.amplify(raw)
        assert abs(result.u - 0.5) < 0.01

    def test_known_value_high_bias(self, amplifier: ZScoreMeanAmplifier) -> None:
        """All-255 bytes should produce u close to 1.0."""
        raw = bytes([255] * 1000)
        result = amplifier.amplify(raw)
        assert result.u > 0.99

    def test_known_value_low_bias(self, amplifier: ZScoreMeanAmplifier) -> None:
        """All-0 bytes should produce u close to 0.0."""
        raw = bytes([0] * 1000)
        result = amplifier.amplify(raw)
        assert result.u < 0.01

    def test_u_is_clamped(self, amplifier: ZScoreMeanAmplifier) -> None:
        """u should never be exactly 0.0 or 1.0 due to epsilon clamping."""
        raw = bytes([255] * 100000)
        result = amplifier.amplify(raw)
        assert result.u < 1.0
        assert result.u > 0.0

    def test_clamping_lower_bound(self, amplifier: ZScoreMeanAmplifier) -> None:
        """u should never go below epsilon."""
        raw = bytes([0] * 100000)
        result = amplifier.amplify(raw)
        assert result.u >= 1e-10

    def test_sem_is_derived(self, amplifier: ZScoreMeanAmplifier) -> None:
        """SEM should equal population_std / sqrt(N)."""
        raw = bytes([100] * 256)
        result = amplifier.amplify(raw)
        expected_sem = 73.61215932167728 / math.sqrt(256)
        assert abs(result.diagnostics["sem"] - expected_sem) < 1e-10

    def test_diagnostics_keys(self, amplifier: ZScoreMeanAmplifier) -> None:
        """Diagnostics should contain expected keys."""
        raw = bytes([128] * 100)
        result = amplifier.amplify(raw)
        assert "sample_mean" in result.diagnostics
        assert "z_score" in result.diagnostics
        assert "sem" in result.diagnostics
        assert "sample_count" in result.diagnostics

    def test_diagnostics_sample_count(self, amplifier: ZScoreMeanAmplifier) -> None:
        """sample_count should match the byte count."""
        raw = bytes([42] * 500)
        result = amplifier.amplify(raw)
        assert result.diagnostics["sample_count"] == 500

    def test_diagnostics_sample_mean(self, amplifier: ZScoreMeanAmplifier) -> None:
        """sample_mean should match numpy mean of bytes."""
        raw = bytes([10, 20, 30])
        result = amplifier.amplify(raw)
        assert abs(result.diagnostics["sample_mean"] - 20.0) < 1e-10

    def test_empty_bytes_raises(self, amplifier: ZScoreMeanAmplifier) -> None:
        """Empty input should raise SignalAmplificationError."""
        with pytest.raises(SignalAmplificationError, match="empty"):
            amplifier.amplify(b"")

    def test_single_byte(self, amplifier: ZScoreMeanAmplifier) -> None:
        """Single byte should work (extreme z-score, clamped u)."""
        raw = bytes([200])
        result = amplifier.amplify(raw)
        assert 0.0 < result.u < 1.0
        assert result.diagnostics["sample_count"] == 1

    def test_result_is_frozen(self, amplifier: ZScoreMeanAmplifier) -> None:
        """AmplificationResult should be immutable."""
        result = amplifier.amplify(bytes([128] * 100))
        with pytest.raises(AttributeError):
            result.u = 0.42  # type: ignore[misc]

    def test_is_subclass_of_abc(self) -> None:
        """ZScoreMeanAmplifier should be a SignalAmplifier subclass."""
        assert issubclass(ZScoreMeanAmplifier, SignalAmplifier)


class TestZScoreCalibration:
    """Tests for the gated device-baseline calibration (zscore_calibration_samples)."""

    def test_default_zero_is_noop(self) -> None:
        """With the default 0 samples, calibrate() must not touch the source or baseline."""
        config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        amplifier = ZScoreMeanAmplifier(config)

        class _ExplodingSource(MockUniformSource):
            def get_random_bytes(self, n: int) -> bytes:
                raise AssertionError("calibrate() must not fetch when disabled")

        amplifier.calibrate(_ExplodingSource(), config)
        raw = bytes([0, 255] * 1000)  # mean exactly 127.5 vs the ideal baseline
        assert abs(amplifier.amplify(raw).u - 0.5) < 0.01

    def test_biased_device_pins_u_without_calibration(self) -> None:
        """The 'acorn' failure: a static device offset saturates every u to the clamp."""
        config = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None, sample_count=10000
        )
        amplifier = ZScoreMeanAmplifier(config)
        source = MockUniformSource(mean=122.0, seed=7)  # -5.5 byte offset
        us = [amplifier.amplify(source.get_random_bytes(10000)).u for _ in range(20)]
        assert all(u < 1e-6 for u in us)  # every draw pinned low → choose(k) == 0 forever

    def test_calibration_unpins_biased_device(self) -> None:
        """Calibrated against the biased device, u spreads back over (0, 1)."""
        config = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None, sample_count=10000, zscore_calibration_samples=100
        )
        amplifier = ZScoreMeanAmplifier(config)
        source = MockUniformSource(mean=122.0, seed=7)
        amplifier.calibrate(source, config)
        us = [amplifier.amplify(source.get_random_bytes(10000)).u for _ in range(50)]
        assert min(us) < 0.25
        assert max(us) > 0.75
        low, high = sum(u < 0.5 for u in us), sum(u >= 0.5 for u in us)
        assert low >= 10 and high >= 10  # spread across both halves, not pinned

    def test_calibrated_baseline_reflects_device(self) -> None:
        """The learned baseline lands on the device's true mean, not the ideal 127.5."""
        config = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None, sample_count=10000, zscore_calibration_samples=100
        )
        amplifier = ZScoreMeanAmplifier(config)
        amplifier.calibrate(MockUniformSource(mean=122.0, seed=3), config)
        assert abs(amplifier._population_mean - 122.0) < 0.5

    def test_stuck_source_raises(self) -> None:
        """Zero block-mean variance (a stuck source) cannot define a baseline."""
        config = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None, sample_count=100, zscore_calibration_samples=10
        )
        amplifier = ZScoreMeanAmplifier(config)

        class _StuckSource(MockUniformSource):
            def get_random_bytes(self, n: int) -> bytes:
                return bytes([42]) * n

        with pytest.raises(SignalAmplificationError, match="stuck"):
            amplifier.calibrate(_StuckSource(), config)

    def test_calibration_cached_per_source_instance(self) -> None:
        """A second amplifier calibrating against the same source must not refetch."""
        config = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None, sample_count=1000, zscore_calibration_samples=20
        )

        class _CountingSource(MockUniformSource):
            def __init__(self) -> None:
                super().__init__(mean=120.0, seed=11)
                self.calls = 0

            def get_random_bytes(self, n: int) -> bytes:
                self.calls += 1
                return super().get_random_bytes(n)

        source = _CountingSource()
        first = ZScoreMeanAmplifier(config)
        first.calibrate(source, config)
        assert source.calls == 20
        second = ZScoreMeanAmplifier(config)
        second.calibrate(source, config)
        assert source.calls == 20  # cache hit — no new fetches
        assert second._population_mean == first._population_mean
        assert second._population_std == first._population_std


class TestAmplifierRegistry:
    """Tests for the AmplifierRegistry."""

    def test_zscore_mean_is_registered(self) -> None:
        """The zscore_mean amplifier should be registered at import time."""
        klass = AmplifierRegistry.get("zscore_mean")
        assert klass is ZScoreMeanAmplifier

    def test_unknown_name_raises(self) -> None:
        """Looking up an unregistered name should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown signal amplifier"):
            AmplifierRegistry.get("nonexistent_amplifier")

    def test_build_from_config(self, config: QRSamplerConfig) -> None:
        """build() should return a working ZScoreMeanAmplifier."""
        amplifier = AmplifierRegistry.build(config)
        assert isinstance(amplifier, ZScoreMeanAmplifier)
        result = amplifier.amplify(bytes([128] * 100))
        assert 0.0 < result.u < 1.0

    def test_list_registered(self) -> None:
        """list_registered() should include zscore_mean."""
        names = AmplifierRegistry.list_registered()
        assert "zscore_mean" in names

    def test_duplicate_registration_raises(self) -> None:
        """Registering the same name twice should raise ValueError."""
        with pytest.raises(ValueError, match="already registered"):

            @AmplifierRegistry.register("zscore_mean")
            class DuplicateAmplifier(SignalAmplifier):
                def amplify(self, raw_bytes: bytes) -> AmplificationResult:
                    return AmplificationResult(u=0.5, diagnostics={})


class TestAmplificationResultImmutability:
    """Tests for AmplificationResult frozen dataclass."""

    def test_frozen(self) -> None:
        """AmplificationResult should reject attribute mutation."""
        result = AmplificationResult(u=0.5, diagnostics={"key": "value"})
        with pytest.raises(AttributeError):
            result.u = 0.7  # type: ignore[misc]

    def test_slots(self) -> None:
        """AmplificationResult should use __slots__."""
        result = AmplificationResult(u=0.5, diagnostics={})
        assert hasattr(result, "__slots__")
