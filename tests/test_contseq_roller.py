"""Tests for the contseq roller (spec §3.3).

Covers: code range under ``mock_uniform``, byte parity with the
amplifier math on a fixed buffer, and fallback-flag propagation when
the primary raises ``EntropyUnavailableError``.
"""

from __future__ import annotations

import pytest

from qr_sampler.amplification.zscore import ZScoreMeanAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.contseq import ContseqRoller, RollResult
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.presets import BUILTIN_PRESETS


class _FixedSource(EntropySource):
    """Returns a deterministic ramp buffer for parity checks."""

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return bytes(i % 256 for i in range(n))

    def close(self) -> None:
        pass


class _FailingSource(EntropySource):
    """Primary that is always unavailable — drives the fallback leg."""

    @property
    def name(self) -> str:
        return "failing_primary"

    @property
    def is_available(self) -> bool:
        return False

    def get_random_bytes(self, n: int) -> bytes:
        raise EntropyUnavailableError("primary down (test)")

    def close(self) -> None:
        pass


@pytest.fixture()
def mock_config() -> QRSamplerConfig:
    """Config running the roller against the mock source, small fetches."""
    return QRSamplerConfig(
        entropy_source_type="mock_uniform",
        sample_count=1024,
    )


def test_roll_code_in_byte_range(mock_config: QRSamplerConfig) -> None:
    """Every roll lands in [0, 255] with u in (0, 1)."""
    roller = ContseqRoller(mock_config)
    try:
        for _ in range(50):
            result = roller.roll()
            assert isinstance(result, RollResult)
            assert 0 <= result.code <= 255
            assert 0.0 < result.u < 1.0
            assert result.source == "mock_uniform"
            assert result.is_fallback is False
            assert result.latency_ms >= 0.0
    finally:
        roller.close()


def test_roll_byte_parity_with_amplifier(mock_config: QRSamplerConfig) -> None:
    """The roller's code matches the amplifier math on the same buffer.

    Swaps in a deterministic source so the exact bytes are known, then
    recomputes ``min(int(u * 256), 255)`` from a separately constructed
    ``ZScoreMeanAmplifier`` — catching any drift in the reduction.
    """
    roller = ContseqRoller(mock_config)
    roller._source = _FixedSource()
    try:
        result = roller.roll()

        buffer = _FixedSource().get_random_bytes(mock_config.sample_count)
        expected_u = ZScoreMeanAmplifier(mock_config).amplify(buffer).u
        assert result.u == pytest.approx(expected_u)
        assert result.code == min(int(expected_u * 256), 255)
    finally:
        roller.close()


def test_roll_fallback_flag_propagates(mock_config: QRSamplerConfig) -> None:
    """A primary EntropyUnavailableError surfaces as is_fallback=True."""
    roller = ContseqRoller(mock_config)
    roller._source = FallbackEntropySource(_FailingSource(), _FixedSource())
    try:
        result = roller.roll()
        assert result.is_fallback is True
        assert result.source == "fixed"
        assert 0 <= result.code <= 255

        status = roller.status()
        assert status["currently_degraded"] is True
        assert status["fallback_count"] == 1
    finally:
        roller.close()


def test_status_without_fallback_wrapper(mock_config: QRSamplerConfig) -> None:
    """fallback_mode='error' builds a bare source; status reports clean."""
    config = mock_config.model_copy(update={"fallback_mode": "error"})
    roller = ContseqRoller(config)
    try:
        status = roller.status()
        assert status["currently_degraded"] is False
        assert status["fallback_count"] == 0
    finally:
        roller.close()


def test_contseq_preset_registered() -> None:
    """The contseq preset pins the quantum source + zscore amplifier."""
    assert BUILTIN_PRESETS["contseq"] == {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
    }


def test_ensure_source_importable_registers_quantum() -> None:
    """quantum_grpc resolves even when nothing else imported quantum.py.

    Regression for the OWUI-container 500 (iter-57): the registry is
    populated by import side effects, and the OWUI process never imports
    ``qr_sampler.entropy.quantum`` on its own.
    """
    from qr_sampler.contseq import _ensure_source_importable
    from qr_sampler.entropy.registry import EntropySourceRegistry

    config = QRSamplerConfig(entropy_source_type="quantum_grpc")
    result = _ensure_source_importable(config)

    assert result.entropy_source_type == "quantum_grpc"
    assert EntropySourceRegistry.get("quantum_grpc") is not None


def test_ensure_source_importable_degrades_on_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """grpcio genuinely absent → roller degrades to fallback_mode, no crash."""
    import sys

    from qr_sampler.contseq import _ensure_source_importable

    # None in sys.modules makes `import qr_sampler.entropy.quantum`
    # raise ImportError without touching the real module machinery.
    monkeypatch.setitem(sys.modules, "qr_sampler.entropy.quantum", None)

    config = QRSamplerConfig(entropy_source_type="quantum_grpc", fallback_mode="system")
    result = _ensure_source_importable(config)
    assert result.entropy_source_type == "system"

    # fallback_mode="error" still must not crash — degrade to system.
    config = QRSamplerConfig(entropy_source_type="quantum_grpc", fallback_mode="error")
    result = _ensure_source_importable(config)
    assert result.entropy_source_type == "system"
