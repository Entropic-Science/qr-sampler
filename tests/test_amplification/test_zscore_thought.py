"""Tests for the ZScoreThoughtAmplifier.

Mirrors ``test_zscore.py`` to prove the per-token path is byte-identical to
``ZScoreMeanAmplifier``, and adds coverage for the optional, duck-typed
thought-level protocol (``begin_thought`` / ``thought_aggregate``).
"""

from __future__ import annotations

import math

import pytest

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.amplification.ecdf import ECDFAmplifier
from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.zscore import ZScoreMeanAmplifier
from qr_sampler.amplification.zscore_thought import ZScoreThoughtAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.exceptions import SignalAmplificationError

_DIAGNOSTIC_KEYS = {"sample_mean", "z_score", "sem", "sample_count"}
_AGGREGATE_KEYS = {"sample_mean", "z_score", "sem", "sample_count", "bias", "u"}


@pytest.fixture()
def config() -> QRSamplerConfig:
    """Default config for amplification tests."""
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def amplifier(config: QRSamplerConfig) -> ZScoreThoughtAmplifier:
    """Default ZScoreThoughtAmplifier."""
    return ZScoreThoughtAmplifier(config)


class TestZScoreThoughtPerTokenBehavior:
    """The per-token path must behave exactly like ZScoreMeanAmplifier."""

    def test_known_value_unbiased(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """Bytes with mean exactly 127.5 should produce u ≈ 0.5."""
        raw = bytes([0, 255] * 1000)
        result = amplifier.amplify(raw)
        assert abs(result.u - 0.5) < 0.01

    def test_known_value_high_bias(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """All-255 bytes should produce u close to 1.0."""
        raw = bytes([255] * 1000)
        result = amplifier.amplify(raw)
        assert result.u > 0.99

    def test_known_value_low_bias(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """All-0 bytes should produce u close to 0.0."""
        raw = bytes([0] * 1000)
        result = amplifier.amplify(raw)
        assert result.u < 0.01

    def test_u_is_clamped(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """u should never be exactly 0.0 or 1.0 due to epsilon clamping."""
        raw = bytes([255] * 100000)
        result = amplifier.amplify(raw)
        assert result.u < 1.0
        assert result.u > 0.0

    def test_clamping_lower_bound(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """u should never go below epsilon."""
        raw = bytes([0] * 100000)
        result = amplifier.amplify(raw)
        assert result.u >= 1e-10

    def test_sem_is_derived(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """SEM should equal population_std / sqrt(N)."""
        raw = bytes([100] * 256)
        result = amplifier.amplify(raw)
        expected_sem = 73.61215932167728 / math.sqrt(256)
        assert abs(result.diagnostics["sem"] - expected_sem) < 1e-10

    def test_diagnostics_key_set_is_exact(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """Diagnostics should contain exactly the zscore_mean key set."""
        raw = bytes([128] * 100)
        result = amplifier.amplify(raw)
        assert set(result.diagnostics) == _DIAGNOSTIC_KEYS

    def test_diagnostics_sample_count(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """sample_count should match the byte count."""
        raw = bytes([42] * 500)
        result = amplifier.amplify(raw)
        assert result.diagnostics["sample_count"] == 500

    def test_diagnostics_sample_mean(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """sample_mean should match numpy mean of bytes."""
        raw = bytes([10, 20, 30])
        result = amplifier.amplify(raw)
        assert abs(result.diagnostics["sample_mean"] - 20.0) < 1e-10

    def test_empty_bytes_raises(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """Empty input should raise SignalAmplificationError."""
        with pytest.raises(SignalAmplificationError, match="empty"):
            amplifier.amplify(b"")

    def test_single_byte(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """Single byte should work (extreme z-score, clamped u)."""
        raw = bytes([200])
        result = amplifier.amplify(raw)
        assert 0.0 < result.u < 1.0
        assert result.diagnostics["sample_count"] == 1

    def test_result_is_frozen(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """AmplificationResult should be immutable."""
        result = amplifier.amplify(bytes([128] * 100))
        with pytest.raises(AttributeError):
            result.u = 0.42  # type: ignore[misc]

    def test_is_subclass_of_abc(self) -> None:
        """ZScoreThoughtAmplifier should be a SignalAmplifier subclass."""
        assert issubclass(ZScoreThoughtAmplifier, SignalAmplifier)

    def test_is_subclass_of_zscore_mean(self) -> None:
        """It should subclass ZScoreMeanAmplifier to inherit the exact math."""
        assert issubclass(ZScoreThoughtAmplifier, ZScoreMeanAmplifier)


class TestByteIdenticalToZScoreMean:
    """Every amplify() output must be byte-identical to ZScoreMeanAmplifier."""

    @pytest.mark.parametrize(
        "raw",
        [
            bytes([0, 255] * 1000),
            bytes([255] * 1000),
            bytes([0] * 1000),
            bytes([128] * 777),
            bytes([200]),
            bytes(range(256)) * 4,
            bytes([7, 11, 13, 250, 3, 99]),
        ],
    )
    def test_u_and_diagnostics_match(self, config: QRSamplerConfig, raw: bytes) -> None:
        """u and the full diagnostics dict must match the mean amplifier."""
        mean_amp = ZScoreMeanAmplifier(config)
        thought_amp = ZScoreThoughtAmplifier(config)

        mean_result = mean_amp.amplify(raw)
        thought_result = thought_amp.amplify(raw)

        assert thought_result.u == mean_result.u
        assert thought_result.diagnostics == mean_result.diagnostics

    def test_per_call_output_is_stateless(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """u for a buffer is independent of accumulator state (invariant 5)."""
        raw = bytes([90] * 300)
        first = amplifier.amplify(raw).u
        # Fold a strongly biased buffer in between; the next identical call must
        # still produce the same u — the accumulator is a pure side-channel.
        amplifier.amplify(bytes([255] * 5000))
        second = amplifier.amplify(raw).u
        assert first == second


class TestAmplifierRegistry:
    """The amplifier must register itself under 'zscore_thought'."""

    def test_zscore_thought_is_registered(self) -> None:
        """get() should return the ZScoreThoughtAmplifier class."""
        assert AmplifierRegistry.get("zscore_thought") is ZScoreThoughtAmplifier

    def test_build_from_config(self) -> None:
        """build() should construct a working ZScoreThoughtAmplifier."""
        config = QRSamplerConfig(_env_file=None, signal_amplifier_type="zscore_thought")  # type: ignore[call-arg]
        amplifier = AmplifierRegistry.build(config)
        assert isinstance(amplifier, ZScoreThoughtAmplifier)
        result = amplifier.amplify(bytes([128] * 100))
        assert 0.0 < result.u < 1.0

    def test_list_registered_includes_zscore_thought(self) -> None:
        """list_registered() should include zscore_thought alongside the others."""
        names = AmplifierRegistry.list_registered()
        assert "zscore_thought" in names
        assert "zscore_mean" in names
        assert "ecdf" in names


class TestThoughtProtocolIsOptional:
    """The thought protocol must stay duck-typed, not on the ABC or siblings."""

    def test_zscore_mean_has_no_begin_thought(self) -> None:
        """ZScoreMeanAmplifier must not expose the thought protocol."""
        assert not hasattr(ZScoreMeanAmplifier, "begin_thought")
        assert not hasattr(ZScoreMeanAmplifier, "thought_aggregate")

    def test_ecdf_has_no_begin_thought(self) -> None:
        """ECDFAmplifier must not expose the thought protocol."""
        assert not hasattr(ECDFAmplifier, "begin_thought")
        assert not hasattr(ECDFAmplifier, "thought_aggregate")

    def test_abc_has_no_begin_thought(self) -> None:
        """The SignalAmplifier ABC must not declare the thought protocol."""
        assert not hasattr(SignalAmplifier, "begin_thought")
        assert not hasattr(SignalAmplifier, "thought_aggregate")

    def test_thought_amplifier_exposes_protocol(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """ZScoreThoughtAmplifier must expose the protocol for feature-detection."""
        assert hasattr(amplifier, "begin_thought")
        assert hasattr(amplifier, "thought_aggregate")


class TestThoughtAggregate:
    """begin_thought() -> folded amplify() calls -> thought_aggregate()."""

    def test_aggregate_shape(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """thought_aggregate() should return the documented key set."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([128] * 100))
        amplifier.amplify(bytes([130] * 100))
        aggregate = amplifier.thought_aggregate()
        assert set(aggregate) == _AGGREGATE_KEYS

    def test_empty_thought_is_neutral(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """With nothing folded the aggregate should be neutral (z=0, u=0.5)."""
        amplifier.begin_thought()
        aggregate = amplifier.thought_aggregate()
        assert aggregate["sample_count"] == 0
        assert aggregate["z_score"] == 0.0
        assert aggregate["u"] == 0.5
        assert aggregate["bias"] == 0.0

    def test_aggregate_counts_all_folded_bytes(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """sample_count should sum the byte counts of every folded call."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([128] * 40))
        amplifier.amplify(bytes([128] * 60))
        assert amplifier.thought_aggregate()["sample_count"] == 100

    def test_aggregate_mean_is_pooled(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """The thought mean should pool all folded bytes, not average means."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([100] * 100))
        amplifier.amplify(bytes([200] * 300))
        # Pooled mean = (100*100 + 200*300) / 400 = 175.0
        assert abs(amplifier.thought_aggregate()["sample_mean"] - 175.0) < 1e-9

    def test_high_bias_thought_positive_z(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """A high-mean thought should report a positive z-score and bias."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([200] * 5000))
        aggregate = amplifier.thought_aggregate()
        assert aggregate["z_score"] > 0.0
        assert aggregate["bias"] > 0.0
        assert aggregate["u"] > 0.5

    def test_low_bias_thought_negative_z(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """A low-mean thought should report a negative z-score and bias."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([50] * 5000))
        aggregate = amplifier.thought_aggregate()
        assert aggregate["z_score"] < 0.0
        assert aggregate["bias"] < 0.0
        assert aggregate["u"] < 0.5

    def test_unbiased_thought_near_zero_z(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """A balanced thought (mean 127.5) should report ~zero z-score."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([0, 255] * 2500))
        aggregate = amplifier.thought_aggregate()
        assert abs(aggregate["z_score"]) < 1e-6
        assert abs(aggregate["u"] - 0.5) < 1e-6

    def test_aggregate_sem_is_derived_over_pool(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """SEM should be population_std / sqrt(total folded count)."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([128] * 128))
        amplifier.amplify(bytes([128] * 128))
        expected_sem = 73.61215932167728 / math.sqrt(256)
        assert abs(amplifier.thought_aggregate()["sem"] - expected_sem) < 1e-10

    def test_begin_thought_resets_accumulator(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """begin_thought() should discard any previously folded bytes."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([255] * 1000))
        assert amplifier.thought_aggregate()["sample_count"] == 1000
        amplifier.begin_thought()
        reset = amplifier.thought_aggregate()
        assert reset["sample_count"] == 0
        assert reset["z_score"] == 0.0

    def test_aggregate_u_is_clamped(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """The thought-level u should also be clamped into (0, 1)."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([255] * 100000))
        aggregate = amplifier.thought_aggregate()
        assert 0.0 < aggregate["u"] < 1.0

    def test_empty_amplify_does_not_fold(self, amplifier: ZScoreThoughtAmplifier) -> None:
        """A raising empty amplify() must not fold anything into the thought."""
        amplifier.begin_thought()
        amplifier.amplify(bytes([128] * 100))
        with pytest.raises(SignalAmplificationError):
            amplifier.amplify(b"")
        # The failed call contributed nothing; only the first 100 bytes count.
        assert amplifier.thought_aggregate()["sample_count"] == 100


class TestReturnsAmplificationResult:
    """amplify() must still return a proper AmplificationResult."""

    def test_returns_amplification_result(self, amplifier: ZScoreThoughtAmplifier) -> None:
        result = amplifier.amplify(bytes([128] * 100))
        assert isinstance(result, AmplificationResult)
