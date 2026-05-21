"""Tests for QuantumGrpcSource (mocked gRPC, protocol-agnostic)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qr_sampler.entropy.quantum import (
    _decode_bytes_field1,
    _encode_varint,
    _encode_varint_request,
)
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError


@pytest.fixture(autouse=True)
def _disable_grpc_preprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the live ``socket.create_connection`` pre-probe.

    Phase 2 lazy init defers gRPC channel creation to first fetch.
    The pre-probe still runs ahead of that fetch and would call
    ``socket.create_connection(("localhost", 50051), ...)`` against
    whatever the test host has bound there. Disabling it keeps the
    tests pure unit-level against the mocked ``grpc.aio.insecure_channel``.
    """
    monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "0")


def _make_config(**overrides: Any) -> Any:
    """Create a mock config object with gRPC defaults."""
    from qr_sampler.config import QRSamplerConfig

    defaults = {
        "grpc_server_address": "localhost:50051",
        "grpc_timeout_ms": 5000.0,
        "grpc_retry_count": 2,
        "grpc_mode": "unary",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


def _encode_mock_response(data: bytes) -> bytes:
    """Encode a mock protobuf response with field 1 = length-delimited bytes.

    This produces valid protobuf wire format that _decode_bytes_field1() can parse.
    """
    # Tag: field 1, wire type 2 (length-delimited) = (1 << 3) | 2 = 0x0a
    return b"\x0a" + _encode_varint(len(data)) + data


# ---------------------------------------------------------------------------
# Wire format helper tests
# ---------------------------------------------------------------------------


class TestWireFormatHelpers:
    """Tests for the generic protobuf wire-format helpers."""

    def test_encode_varint_request_zero(self) -> None:
        """Zero byte count produces empty bytes (proto3 default omission)."""
        assert _encode_varint_request(0) == b""

    def test_encode_varint_request_small(self) -> None:
        """Small values encode as tag + single varint byte."""
        result = _encode_varint_request(100)
        # tag 0x08 (field 1, varint), value 100 = 0x64
        assert result == b"\x08\x64"

    def test_encode_varint_request_large(self) -> None:
        """Large values use multi-byte varint encoding."""
        result = _encode_varint_request(20480)
        # 20480 = 0x5000 -> LEB128: 0x80 0xa0 0x01
        assert result == b"\x08\x80\xa0\x01"

    def test_decode_bytes_field1_simple(self) -> None:
        """Extract field 1 bytes from a simple response."""
        payload = b"\xde\xad\xbe\xef"
        encoded = _encode_mock_response(payload)
        assert _decode_bytes_field1(encoded) == payload

    def test_decode_bytes_field1_with_extra_fields(self) -> None:
        """Field 1 extraction works even when other fields are present."""
        # Field 2 (varint): tag=0x10, value=42
        # Field 1 (bytes): tag=0x0a, length=3, data=b"abc"
        wire = b"\x10\x2a" + b"\x0a\x03abc"
        assert _decode_bytes_field1(wire) == b"abc"

    def test_decode_bytes_field1_not_found(self) -> None:
        """Should raise when field 1 bytes is missing."""
        # Only a varint field 2
        wire = b"\x10\x2a"
        with pytest.raises(EntropyUnavailableError, match="field 1"):
            _decode_bytes_field1(wire)

    def test_decode_bytes_field1_empty_input(self) -> None:
        """Should raise on empty input."""
        with pytest.raises(EntropyUnavailableError, match="field 1"):
            _decode_bytes_field1(b"")

    def test_roundtrip_request_response(self) -> None:
        """Encoded request should be decodable by standard protobuf parsing."""
        from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

        # Encode with generic helper
        wire = _encode_varint_request(256)
        # Decode with the full message parser
        req = EntropyRequest.FromString(wire)
        assert req.bytes_needed == 256

    def test_encode_matches_message_class(self) -> None:
        """Generic encoder output should match EntropyRequest.SerializeToString()."""
        from qr_sampler.proto.entropy_service_pb2 import EntropyRequest

        for n in (1, 100, 1024, 20480, 65535):
            generic = _encode_varint_request(n)
            # EntropyRequest uses field 1 for bytes_needed, same as generic
            msg = EntropyRequest(bytes_needed=n, sequence_id=0)
            # Both should produce identical field 1 encoding
            # (EntropyRequest may also have field 2 if non-zero, but with
            # sequence_id=0 it's omitted in proto3)
            assert generic == msg.SerializeToString()


# ---------------------------------------------------------------------------
# Source import tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceImport:
    """Tests for import-time checks."""

    def test_requires_grpcio(self) -> None:
        """Should raise ImportError if grpcio is not available."""
        with (
            patch.dict("sys.modules", {"grpc": None, "grpc.aio": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from qr_sampler.entropy.quantum import QuantumGrpcSource

            config = _make_config()
            QuantumGrpcSource(config)


# ---------------------------------------------------------------------------
# Unary mode tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceUnary:
    """Tests for unary transport mode with mocked gRPC."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource with fully mocked gRPC channel."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            # Mock channel.unary_unary() -> returns a callable (method handle).
            mock_unary_handle = AsyncMock(return_value=_encode_mock_response(b"\x42" * 100))
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)

            # Mock channel.stream_stream() -> not used in unary mode but still called.
            mock_stream_handle = MagicMock()
            mock_channel.stream_stream = MagicMock(return_value=mock_stream_handle)

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            # Phase 2 lazy init: bring up the channel so `call_args`
            # assertions remain valid for tests that don't fetch.
            source._ensure_channel()
            source._mock_channel = mock_channel  # type: ignore[attr-defined]
            source._mock_unary_handle = mock_unary_handle  # type: ignore[attr-defined]
            yield source
            source.close()

    def test_name(self, source: Any) -> None:
        assert source.name == "quantum_grpc"

    def test_is_available(self, source: Any) -> None:
        assert source.is_available is True

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        data = source.get_random_bytes(100)
        assert len(data) == 100
        assert data == b"\x42" * 100

    def test_health_check(self, source: Any) -> None:
        health = source.health_check()
        assert health["source"] == "quantum_grpc"
        assert health["mode"] == "unary"
        assert "p99_ms" in health
        assert "method_path" in health
        assert "authenticated" in health

    def test_health_check_no_api_key_leak(self, source: Any) -> None:
        """health_check() must never contain the raw API key."""
        health = source.health_check()
        assert "api_key" not in str(health).lower().replace("authenticated", "")

    def test_close_sets_unavailable(self, source: Any) -> None:
        source.close()
        assert source.is_available is False

    def test_unary_uses_configured_method_path(self, source: Any) -> None:
        """channel.unary_unary() should be called with the configured method path."""
        source._mock_channel.unary_unary.assert_called_once()
        call_args = source._mock_channel.unary_unary.call_args
        assert call_args[0][0] == "/qr_entropy.EntropyService/GetEntropy"


# ---------------------------------------------------------------------------
# Circuit breaker tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource with a method handle that always fails."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_unary_handle = AsyncMock(side_effect=Exception("connection refused"))
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            yield source
            source.close()

    def test_circuit_opens_after_consecutive_failures(self, source: Any) -> None:
        """Circuit should open after cb_max_consecutive_failures."""
        for _ in range(source._cb_max_consecutive_failures):
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)

        assert source._circuit_open is True
        assert source.is_available is False

    def test_circuit_open_raises_immediately(self, source: Any) -> None:
        """When circuit is open, should raise without trying gRPC."""
        source._circuit_open = True
        source._circuit_open_until = time.monotonic() + 100.0

        with pytest.raises(EntropyUnavailableError, match="Circuit breaker open"):
            source.get_random_bytes(10)


# ---------------------------------------------------------------------------
# Address parsing tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceAddressParsing:
    """Tests for TCP vs Unix socket address handling."""

    def test_tcp_address(self) -> None:
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_server_address="myhost:9090")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            mock_channel_fn.assert_called_once()
            call_args = mock_channel_fn.call_args
            assert call_args[0][0] == "myhost:9090"
            source.close()

    def test_unix_socket_address(self) -> None:
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_server_address="unix:///var/run/qrng.sock")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            call_args = mock_channel_fn.call_args
            assert call_args[0][0] == "unix:///var/run/qrng.sock"
            source.close()


# ---------------------------------------------------------------------------
# Latency tracking tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceLatencyTracking:
    """Tests for adaptive timeout computation."""

    def test_update_latency_and_timeout(self) -> None:
        """P99 and timeout should update from latency window."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)

            # Feed in 20 latency samples.
            for i in range(20):
                source._update_latency(float(i))

            # P99 should be near the max of the window.
            assert source._p99_ms >= 15.0

            # Adaptive timeout: max(5ms, P99 * 1.5), capped at config.
            timeout = source._get_timeout()
            assert timeout >= 5.0
            assert timeout <= config.grpc_timeout_ms

            source.close()


# ---------------------------------------------------------------------------
# Half-open circuit breaker tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerHalfOpen:
    """Tests for circuit breaker half-open state and recovery."""

    def test_half_open_allows_one_request(self) -> None:
        """After recovery window expires, one request should be attempted."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,  # Immediate recovery for testing.
        )

        success_response = _encode_mock_response(b"\xaa" * 10)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            # First calls fail, then succeed.
            mock_unary_handle = AsyncMock(
                side_effect=[
                    Exception("fail"),
                    Exception("fail"),
                    Exception("fail"),
                    success_response,
                ]
            )
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                # Trigger circuit breaker open (3 consecutive failures).
                for _ in range(3):
                    with pytest.raises(EntropyUnavailableError):
                        source.get_random_bytes(10)

                assert source._circuit_open is True

                # Recovery window is 0.0s, so half-open should trigger
                # immediately. The next call should succeed.
                data = source.get_random_bytes(10)
                assert data == b"\xaa" * 10
                assert source._circuit_open is False
                assert source._consecutive_failures == 0
            finally:
                source.close()

    def test_half_open_failure_reopens_circuit(self) -> None:
        """If the half-open test request fails, circuit should reopen."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            # All calls fail — half-open test will also fail.
            mock_unary_handle = AsyncMock(side_effect=Exception("still broken"))
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                # Open the circuit.
                for _ in range(3):
                    with pytest.raises(EntropyUnavailableError):
                        source.get_random_bytes(10)

                assert source._circuit_open is True

                # Half-open attempt should fail and reopen circuit.
                with pytest.raises(EntropyUnavailableError):
                    source.get_random_bytes(10)

                # Circuit should be open again (consecutive failures incremented).
                assert source._circuit_open is True
            finally:
                source.close()


# ---------------------------------------------------------------------------
# Server streaming tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceServerStreaming:
    """Tests for server_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in server_streaming mode."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="server_streaming")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            # Mock channel.unary_unary() (always created).
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())

            # Mock channel.stream_stream() -> returns a callable that produces
            # a stream call object with .read() and .cancel().
            mock_stream_call = AsyncMock()
            mock_stream_call.read = AsyncMock(return_value=_encode_mock_response(b"\x55" * 50))
            mock_stream_call.cancel = MagicMock()
            mock_stream_handle = MagicMock(return_value=mock_stream_call)
            mock_channel.stream_stream = MagicMock(return_value=mock_stream_handle)

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            source._mock_stream_handle = mock_stream_handle  # type: ignore[attr-defined]
            yield source
            source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Server streaming should return data from the stream."""
        data = source.get_random_bytes(50)
        assert len(data) == 50
        assert data == b"\x55" * 50

    def test_stream_end_raises(self) -> None:
        """If the stream ends unexpectedly (read returns None), should raise."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="server_streaming", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())

            mock_stream_call = AsyncMock()
            mock_stream_call.read = AsyncMock(return_value=None)
            mock_stream_call.cancel = MagicMock()
            mock_stream_handle = MagicMock(return_value=mock_stream_call)
            mock_channel.stream_stream = MagicMock(return_value=mock_stream_handle)

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                with pytest.raises(EntropyUnavailableError):
                    source.get_random_bytes(10)
            finally:
                source.close()


# ---------------------------------------------------------------------------
# Bidi streaming tests
# ---------------------------------------------------------------------------


class TestQuantumGrpcSourceBidiStreaming:
    """Tests for bidi_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in bidi_streaming mode."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="bidi_streaming")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())

            # Mock channel.stream_stream() -> returns a callable that produces
            # a bidi call object with .write() and .read().
            mock_bidi_call = AsyncMock()
            mock_bidi_call.write = AsyncMock()
            mock_bidi_call.read = AsyncMock(return_value=_encode_mock_response(b"\xcc" * 64))
            mock_stream_handle = MagicMock(return_value=mock_bidi_call)
            mock_channel.stream_stream = MagicMock(return_value=mock_stream_handle)

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            source._mock_stream_handle = mock_stream_handle  # type: ignore[attr-defined]
            yield source
            source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Bidi streaming should return data from the persistent stream."""
        data = source.get_random_bytes(64)
        assert len(data) == 64
        assert data == b"\xcc" * 64

    def test_stream_reuses_call(self, source: Any) -> None:
        """Bidi streaming should reuse the same call object."""
        source.get_random_bytes(64)
        source.get_random_bytes(64)
        # The stream_method (from channel.stream_stream) should only be called
        # once (the bidi call is reused).
        assert source._mock_stream_handle.call_count == 1

    def test_bidi_stream_end_resets(self) -> None:
        """If bidi stream ends (read returns None), call should reset."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="bidi_streaming", grpc_retry_count=0)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())

            mock_bidi_call = AsyncMock()
            mock_bidi_call.write = AsyncMock()
            mock_bidi_call.read = AsyncMock(return_value=None)
            mock_stream_handle = MagicMock(return_value=mock_bidi_call)
            mock_channel.stream_stream = MagicMock(return_value=mock_stream_handle)

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                with pytest.raises(EntropyUnavailableError):
                    source.get_random_bytes(10)
                # The bidi call should have been reset to None.
                assert source._bidi_call is None
            finally:
                source.close()


# ---------------------------------------------------------------------------
# API key metadata injection tests
# ---------------------------------------------------------------------------


class TestApiKeyMetadataInjection:
    """Tests for API key metadata injection on gRPC calls."""

    def test_metadata_passed_on_unary(self) -> None:
        """Unary calls should include API key metadata when configured."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_api_key="test-secret-key",
            grpc_api_key_header="x-api-key",
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_unary_handle = AsyncMock(return_value=_encode_mock_response(b"\x01" * 10))
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                source.get_random_bytes(10)

                # Verify the unary method handle was called with metadata.
                mock_unary_handle.assert_called_once()
                call_kwargs = mock_unary_handle.call_args
                metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
                assert metadata is not None
                assert ("x-api-key", "test-secret-key") in metadata
            finally:
                source.close()

    def test_no_metadata_without_api_key(self) -> None:
        """When no API key is configured, metadata should be None."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_mode="unary", grpc_api_key="")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel

            mock_unary_handle = AsyncMock(return_value=_encode_mock_response(b"\x01" * 10))
            mock_channel.unary_unary = MagicMock(return_value=mock_unary_handle)
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                source.get_random_bytes(10)

                mock_unary_handle.assert_called_once()
                call_kwargs = mock_unary_handle.call_args
                metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
                assert metadata is None
            finally:
                source.close()


# ---------------------------------------------------------------------------
# Streaming validation tests
# ---------------------------------------------------------------------------


class TestStreamingModeValidation:
    """Tests for streaming mode validation when stream path is empty."""

    def test_server_streaming_requires_stream_path(self) -> None:
        """server_streaming mode with empty stream path should raise."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="server_streaming",
            grpc_stream_method_path="",
        )

        with pytest.raises(ConfigValidationError, match="grpc_stream_method_path"):
            from qr_sampler.entropy.quantum import QuantumGrpcSource

            QuantumGrpcSource(config)

    def test_bidi_streaming_requires_stream_path(self) -> None:
        """bidi_streaming mode with empty stream path should raise."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="bidi_streaming",
            grpc_stream_method_path="",
        )

        with pytest.raises(ConfigValidationError, match="grpc_stream_method_path"):
            from qr_sampler.entropy.quantum import QuantumGrpcSource

            QuantumGrpcSource(config)

    def test_unary_mode_allows_empty_stream_path(self) -> None:
        """Unary mode should work fine with empty stream path."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="unary",
            grpc_stream_method_path="",
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            # Should not have created a stream method handle.
            assert source._stream_method is None
            source.close()


# ---------------------------------------------------------------------------
# API key redaction tests
# ---------------------------------------------------------------------------


class TestApiKeyRedaction:
    """Tests for API key redaction in health_check()."""

    def test_health_check_shows_authenticated_flag(self) -> None:
        """health_check() should show authenticated=True when key is set."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_api_key="super-secret-key-12345")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                health = source.health_check()
                assert health["authenticated"] is True
                # The actual key must NOT appear anywhere in the health dict.
                health_str = str(health)
                assert "super-secret-key-12345" not in health_str
            finally:
                source.close()

    def test_health_check_shows_unauthenticated(self) -> None:
        """health_check() should show authenticated=False when no key."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(grpc_api_key="")

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            try:
                health = source.health_check()
                assert health["authenticated"] is False
            finally:
                source.close()


# ---------------------------------------------------------------------------
# Custom method path tests
# ---------------------------------------------------------------------------


class TestCustomMethodPath:
    """Tests for configurable gRPC method paths."""

    def test_custom_unary_method_path(self) -> None:
        """channel.unary_unary() should use the configured method path."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_method_path="/qrng.QuantumRNG/GetRandomBytes",
            grpc_stream_method_path="",
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            try:
                # Verify the method path passed to channel.unary_unary.
                call_args = mock_channel.unary_unary.call_args
                assert call_args[0][0] == "/qrng.QuantumRNG/GetRandomBytes"
            finally:
                source.close()

    def test_custom_stream_method_path(self) -> None:
        """channel.stream_stream() should use the configured stream method path."""
        try:
            import grpc.aio  # noqa: F401
        except ImportError:
            pytest.skip("grpcio not installed")

        config = _make_config(
            grpc_mode="server_streaming",
            grpc_stream_method_path="/custom.Service/StreamData",
        )

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.quantum import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._ensure_channel()
            try:
                call_args = mock_channel.stream_stream.call_args
                assert call_args[0][0] == "/custom.Service/StreamData"
            finally:
                source.close()
