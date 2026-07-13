"""Tests for the QuantumGrpcSource facade (mocked gRPC).

Covers construction/validation, the serial fetch path with its circuit
breaker orchestration, the pipelined prefetch/redeem path, pre-probe
healthy suppression, API-key handling, and quota classification. The
pure breaker/preprobe state machines have their own unit-test modules;
transport-mode dispatch is covered in ``test_qgrpc_transport.py``.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qr_sampler.entropy.qgrpc.transport import _FetchReply
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError
from tests.test_entropy.qgrpc_util import (
    FakeRpcError,
    encode_mock_response,
    encode_mock_response_with_echo,
    extract_request_sequence,
    make_config,
    make_mocked_source,
)

pytest.importorskip("grpc.aio", reason="grpcio not installed")


class TestQuantumGrpcSourceImport:
    """Tests for import-time checks."""

    def test_requires_grpcio(self) -> None:
        """Should raise ImportError if grpcio is not available."""
        with (
            patch.dict("sys.modules", {"grpc": None, "grpc.aio": None}),
            pytest.raises(ImportError, match="grpcio"),
        ):
            from qr_sampler.entropy.qgrpc import QuantumGrpcSource

            config = make_config()
            QuantumGrpcSource(config)


class TestQuantumGrpcSourceUnary:
    """Tests for the serial fetch path in unary mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource with fully mocked gRPC channel."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x42" * 100))
        source, mock_channel = make_mocked_source(unary_handle=handle, grpc_mode="unary")
        source._mock_channel = mock_channel  # type: ignore[attr-defined]
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
        """channel.unary_unary() should be called with the configured method path.

        Two unary handles exist since the draw path landed: the entropy
        handle (first) and the PurityService draw handle (second).
        """
        calls = source._mock_channel.unary_unary.call_args_list
        assert [c[0][0] for c in calls] == [
            "/qr_entropy.EntropyService/GetEntropy",
            "/qr_purity.PurityService/GetDraw",
        ]


class TestCircuitBreakerOrchestration:
    """Circuit breaker behavior as driven by the serial fetch path."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource whose method handle always fails."""
        handle = AsyncMock(side_effect=Exception("connection refused"))
        source, _ = make_mocked_source(unary_handle=handle, grpc_mode="unary", grpc_retry_count=0)
        yield source
        source.close()

    def test_circuit_opens_after_consecutive_failures(self, source: Any) -> None:
        """Circuit should open after cb_max_consecutive_failures."""
        for _ in range(source._breaker.max_consecutive_failures):
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)

        assert source._breaker.circuit_open is True
        assert source.is_available is False

    def test_circuit_open_raises_immediately(self, source: Any) -> None:
        """When circuit is open, should raise without trying gRPC."""
        source._breaker.circuit_open = True
        source._breaker.circuit_open_until = time.monotonic() + 100.0

        with pytest.raises(EntropyUnavailableError, match="Circuit breaker open"):
            source.get_random_bytes(10)

    def test_half_open_resets_channel_before_attempt(self, source: Any) -> None:
        """iter-53: the half-open attempt must run on a freshly-reset channel.

        The dominant open-circuit cause is a stale channel; testing
        recovery on the suspect channel wastes the whole half-open cycle.
        """
        source._breaker.circuit_open = True
        source._breaker.circuit_open_until = time.monotonic() - 1.0  # window elapsed

        events: list[str] = []
        source._reset_channel = lambda: events.append("reset")

        def fake_fetch(n: int) -> bytes:
            events.append("fetch")
            raise EntropyUnavailableError("still down")

        source._fetch_sync = fake_fetch
        with pytest.raises(EntropyUnavailableError):
            source.get_random_bytes(10)
        assert events[0] == "reset"
        assert events[1] == "fetch"

    def test_half_open_success_closes_circuit(self, source: Any) -> None:
        source._breaker.circuit_open = True
        source._breaker.circuit_open_until = time.monotonic() - 1.0

        source._reset_channel = lambda: None
        source._fetch_sync = lambda n: _FetchReply(b"\x00" * n, 0, 0, 1.0)
        assert source.get_random_bytes(10) == b"\x00" * 10
        assert source._breaker.circuit_open is False
        assert source._breaker.consecutive_failures == 0


class TestCircuitBreakerHalfOpenEndToEnd:
    """Half-open recovery through the real mocked-transport path."""

    def test_half_open_allows_one_request(self) -> None:
        """After recovery window expires, one request should be attempted."""
        success_response = encode_mock_response(b"\xaa" * 10)
        handle = AsyncMock(
            side_effect=[
                Exception("fail"),
                Exception("fail"),
                Exception("fail"),
                success_response,
            ]
        )
        source, _ = make_mocked_source(
            unary_handle=handle,
            ensure=False,
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,  # Immediate recovery for testing.
        )
        try:
            # Trigger circuit breaker open (3 consecutive failures).
            for _ in range(3):
                with pytest.raises(EntropyUnavailableError):
                    source.get_random_bytes(10)

            assert source._breaker.circuit_open is True

            # Recovery window is 0.0s, so half-open should trigger
            # immediately. The next call should succeed.
            data = source.get_random_bytes(10)
            assert data == b"\xaa" * 10
            assert source._breaker.circuit_open is False
            assert source._breaker.consecutive_failures == 0
        finally:
            source.close()

    def test_half_open_failure_reopens_circuit(self) -> None:
        """If the half-open test request fails, circuit should reopen."""
        handle = AsyncMock(side_effect=Exception("still broken"))
        source, _ = make_mocked_source(
            unary_handle=handle,
            ensure=False,
            grpc_mode="unary",
            grpc_retry_count=0,
            cb_recovery_window_s=0.0,
        )
        try:
            # Open the circuit.
            for _ in range(3):
                with pytest.raises(EntropyUnavailableError):
                    source.get_random_bytes(10)

            assert source._breaker.circuit_open is True

            # Half-open attempt should fail and reopen circuit.
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)

            # Circuit should be open again (consecutive failures incremented).
            assert source._breaker.circuit_open is True
        finally:
            source.close()


class TestQuantumGrpcSourceAddressParsing:
    """Tests for TCP vs Unix socket address handling."""

    @pytest.mark.parametrize(
        "address",
        ["myhost:9090", "unix:///var/run/qrng.sock"],
    )
    def test_address_reaches_channel_factory(self, address: str) -> None:
        config = make_config(grpc_server_address=address)

        with patch("grpc.aio.insecure_channel") as mock_channel_fn:
            mock_channel = MagicMock()
            mock_channel_fn.return_value = mock_channel
            mock_channel.unary_unary = MagicMock(return_value=MagicMock())
            mock_channel.stream_stream = MagicMock(return_value=MagicMock())

            from qr_sampler.entropy.qgrpc import QuantumGrpcSource

            source = QuantumGrpcSource(config)
            source._channel.ensure()
            mock_channel_fn.assert_called_once()
            assert mock_channel_fn.call_args[0][0] == address
            source.close()


class TestQuantumGrpcSourcePrefetch:
    """Tests for the commit-then-fetch prefetch/redeem path."""

    def _make_source(self, unary_handle: Any) -> Any:
        source, _ = make_mocked_source(
            unary_handle=unary_handle, grpc_mode="unary", grpc_retry_count=0
        )
        return source

    def test_prefetch_and_redeem_with_echo(self) -> None:
        """Happy path: ticket redeems to payload, echo verifies the nonce."""
        nonce = 987654321

        async def echoing_handle(request: bytes, **kwargs: Any) -> bytes:
            seq = extract_request_sequence(request)
            return encode_mock_response_with_echo(b"\x55" * 32, seq)

        source = self._make_source(echoing_handle)
        try:
            ticket = source.prefetch(32, nonce=nonce)
            assert ticket is not None
            data = source.get_random_bytes_with_ticket(32, ticket)
            assert data == b"\x55" * 32
            assert ticket.hit is True
            assert ticket.echo_verified is True
            assert source._prefetch_hits == 1
            health = source.health_check()
            assert health["prefetch_fired"] == 1
        finally:
            source.close()

    def test_prefetch_without_echo_is_unverified_but_served(self) -> None:
        """A server that doesn't echo still serves bytes; echo flag False."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x66" * 16))
        source = self._make_source(handle)
        try:
            ticket = source.prefetch(16, nonce=12345)
            data = source.get_random_bytes_with_ticket(16, ticket)
            assert data == b"\x66" * 16
            assert ticket.echo_verified is False
        finally:
            source.close()

    def test_failed_prefetch_falls_back_to_serial_fetch(self) -> None:
        """A broken in-flight fetch degrades to the synchronous path."""
        calls: list[str] = []

        async def flaky_handle(request: bytes, **kwargs: Any) -> bytes:
            if not calls:
                calls.append("prefetch")
                raise RuntimeError("stream reset mid-flight")
            calls.append("serial")
            return encode_mock_response(b"\x77" * 8)

        source = self._make_source(flaky_handle)
        try:
            ticket = source.prefetch(8, nonce=1)
            assert ticket is not None
            data = source.get_random_bytes_with_ticket(8, ticket)
            assert data == b"\x77" * 8
            assert ticket.hit is False
            assert calls == ["prefetch", "serial"]
            assert source._prefetch_misses == 1
        finally:
            source.close()

    def test_prefetch_returns_none_when_circuit_open(self) -> None:
        """No speculative dispatch while the breaker is open."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x00" * 8))
        source = self._make_source(handle)
        try:
            source._breaker.circuit_open = True
            source._breaker.circuit_open_until = time.monotonic() + 100.0
            assert source.prefetch(8, nonce=1) is None
        finally:
            source.close()

    def test_redeem_none_ticket_uses_serial_path(self) -> None:
        """A None ticket is exactly get_random_bytes()."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x11" * 4))
        source = self._make_source(handle)
        try:
            assert source.get_random_bytes_with_ticket(4, None) == b"\x11" * 4
        finally:
            source.close()


class TestQuantumGrpcSourceDraw:
    """Server-integrated draw path (get_draw / prefetch_draw)."""

    def _make_source(self, unary_handle: Any, **overrides: Any) -> Any:
        source, _ = make_mocked_source(
            unary_handle=unary_handle, grpc_mode="unary", grpc_retry_count=0, **overrides
        )
        return source

    @staticmethod
    def _echoing_draw_handle(u: float = 0.625) -> Any:
        from qr_sampler.proto.purity_service_pb2 import DrawRequest, DrawResponse

        async def handle(request: bytes, **kwargs: Any) -> bytes:
            req = DrawRequest.FromString(request)
            return DrawResponse(
                u=u,
                z=1.5,
                sequence_id=req.sequence_id,
                generation_timestamp_ns=777,
                source_id=req.source_id or "bound-source",
                coherence_z=3.9,
                coherence_valid=True,
                coherence_r=0.4,
                purity_label="quantum/intact/raw/qf:device",
                integrated_bytes=req.block_bytes or 2_097_152,
                integrator="bit_z",
            ).SerializeToString()

        return handle

    def test_supports_server_draw_classvar(self) -> None:
        from qr_sampler.entropy.qgrpc import QuantumGrpcSource

        assert QuantumGrpcSource.supports_server_draw is True

    def test_serial_get_draw_maps_response_to_meta(self) -> None:
        source = self._make_source(self._echoing_draw_handle())
        try:
            u, meta = source.get_draw(2_097_152, "qrng-a")
            assert u == 0.625
            assert meta.z == 1.5
            assert meta.coherence_z == 3.9
            assert meta.coherence_valid is True
            assert meta.coherence_r == 0.4
            assert meta.purity_label == "quantum/intact/raw/qf:device"
            assert meta.integrated_bytes == 2_097_152
            assert meta.integrator == "bit_z"
            assert meta.source_id == "qrng-a"
            assert meta.generation_timestamp_ns == 777
            # Serial path: no commitment nonce, no echo verdict.
            assert meta.echo_verified is None
        finally:
            source.close()

    def test_prefetch_draw_and_redeem_with_echo(self) -> None:
        """Happy path: draw ticket redeems; echo verifies the nonce exactly
        like the byte path (nonce rides DrawRequest.sequence_id)."""
        nonce = 123456789
        source = self._make_source(self._echoing_draw_handle())
        try:
            ticket = source.prefetch_draw(2_097_152, "qrng-a", nonce=nonce)
            assert ticket is not None
            assert ticket.nonce == nonce
            u, meta = source.get_draw(2_097_152, "qrng-a", ticket)
            assert u == 0.625
            assert ticket.hit is True
            assert ticket.echo_verified is True
            assert meta.echo_verified is True
            assert ticket.server_timestamp_ns == 777
            assert source._prefetch_hits == 1
        finally:
            source.close()

    def test_failed_draw_redeem_falls_back_to_serial_draw(self) -> None:
        calls: list[str] = []
        real_handle = self._echoing_draw_handle()

        async def flaky_handle(request: bytes, **kwargs: Any) -> bytes:
            if not calls:
                calls.append("prefetch")
                raise RuntimeError("stream reset mid-flight")
            calls.append("serial")
            return await real_handle(request)

        source = self._make_source(flaky_handle)
        try:
            ticket = source.prefetch_draw(0, "", nonce=1)
            assert ticket is not None
            u, meta = source.get_draw(0, "", ticket)
            assert u == 0.625
            assert ticket.hit is False
            assert calls == ["prefetch", "serial"]
            assert source._prefetch_misses == 1
            assert meta.echo_verified is None  # served serially, nonce-less
        finally:
            source.close()

    def test_get_draw_absent_u_raises_entropy_unavailable(self) -> None:
        """A server that never serves u (e.g. EntropyService-only backend)
        surfaces as EntropyUnavailableError — the pipeline's degradation
        trigger."""
        from qr_sampler.proto.purity_service_pb2 import DrawResponse

        handle = AsyncMock(return_value=DrawResponse(u=0.0, z=1.0).SerializeToString())
        source = self._make_source(handle)
        try:
            with pytest.raises(EntropyUnavailableError):
                source.get_draw(0, "")
        finally:
            source.close()

    def test_prefetch_draw_returns_none_when_circuit_open(self) -> None:
        source = self._make_source(self._echoing_draw_handle())
        try:
            source._breaker.circuit_open = True
            source._breaker.circuit_open_until = time.monotonic() + 100.0
            assert source.prefetch_draw(0, "", nonce=1) is None
        finally:
            source.close()

    def test_get_draw_when_closed_raises(self) -> None:
        source = self._make_source(self._echoing_draw_handle())
        source.close()
        with pytest.raises(EntropyUnavailableError, match="closed"):
            source.get_draw(0, "")

    def test_get_draw_failure_counts_against_breaker(self) -> None:
        """Draw failures feed the same breaker as byte failures."""
        handle = AsyncMock(side_effect=Exception("connection refused"))
        source = self._make_source(handle)
        try:
            for _ in range(source._breaker.max_consecutive_failures):
                with pytest.raises(EntropyUnavailableError):
                    source.get_draw(0, "")
            assert source._breaker.circuit_open is True
        finally:
            source.close()


class TestPreprobeHealthySuppression:
    """A recent successful fetch suppresses the per-token TCP pre-probe."""

    def test_preprobe_skipped_within_healthy_window(self, monkeypatch: Any) -> None:
        # Re-enable the pre-probe (package autouse fixture disables it).
        monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "1")

        handle = AsyncMock(return_value=encode_mock_response(b"\x22" * 4))
        source, _ = make_mocked_source(unary_handle=handle, grpc_mode="unary")
        try:
            probes: list[None] = []

            def counting_connect(*args: Any, **kwargs: Any) -> Any:
                probes.append(None)
                return MagicMock()

            monkeypatch.setattr("socket.create_connection", counting_connect)
            source.get_random_bytes(4)  # first fetch: probe runs
            assert len(probes) == 1
            source.get_random_bytes(4)  # healthy: probe suppressed
            source.get_random_bytes(4)
            assert len(probes) == 1
        finally:
            source.close()


class TestApiKeyMetadataInjection:
    """Tests for API key metadata injection on gRPC calls."""

    def test_metadata_passed_on_unary(self) -> None:
        """Unary calls should include API key metadata when configured."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x01" * 10))
        source, _ = make_mocked_source(
            unary_handle=handle,
            ensure=False,
            grpc_mode="unary",
            grpc_api_key="test-secret-key",
            grpc_api_key_header="x-api-key",
        )
        try:
            source.get_random_bytes(10)

            # Verify the unary method handle was called with metadata.
            handle.assert_called_once()
            call_kwargs = handle.call_args
            metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
            assert metadata is not None
            assert ("x-api-key", "test-secret-key") in metadata
        finally:
            source.close()

    def test_no_metadata_without_api_key(self) -> None:
        """When no API key is configured, metadata should be None."""
        handle = AsyncMock(return_value=encode_mock_response(b"\x01" * 10))
        source, _ = make_mocked_source(
            unary_handle=handle, ensure=False, grpc_mode="unary", grpc_api_key=""
        )
        try:
            source.get_random_bytes(10)

            handle.assert_called_once()
            call_kwargs = handle.call_args
            metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
            assert metadata is None
        finally:
            source.close()


class TestApiKeyRedaction:
    """Tests for API key redaction in health_check()."""

    def test_health_check_shows_authenticated_flag(self) -> None:
        """health_check() should show authenticated=True when key is set."""
        source, _ = make_mocked_source(ensure=False, grpc_api_key="super-secret-key-12345")
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
        source, _ = make_mocked_source(ensure=False, grpc_api_key="")
        try:
            health = source.health_check()
            assert health["authenticated"] is False
        finally:
            source.close()


class TestStreamingModeValidation:
    """Tests for streaming mode validation when stream path is empty."""

    @pytest.mark.parametrize("mode", ["server_streaming", "bidi_streaming"])
    def test_streaming_modes_require_stream_path(self, mode: str) -> None:
        config = make_config(grpc_mode=mode, grpc_stream_method_path="")

        with pytest.raises(ConfigValidationError, match="grpc_stream_method_path"):
            from qr_sampler.entropy.qgrpc import QuantumGrpcSource

            QuantumGrpcSource(config)

    def test_unary_mode_allows_empty_stream_path(self) -> None:
        """Unary mode should work fine with empty stream path."""
        source, _ = make_mocked_source(grpc_mode="unary", grpc_stream_method_path="")
        # Should not have created a stream method handle.
        assert source._channel.stream_method is None
        source.close()


class TestCustomMethodPath:
    """Tests for configurable gRPC method paths."""

    def test_custom_unary_method_path(self) -> None:
        """channel.unary_unary() should use the configured method path."""
        source, mock_channel = make_mocked_source(
            grpc_method_path="/qrng.QuantumRNG/GetRandomBytes",
            grpc_stream_method_path="",
        )
        try:
            # First unary handle = the entropy path (the draw handle follows).
            call_args = mock_channel.unary_unary.call_args_list[0]
            assert call_args[0][0] == "/qrng.QuantumRNG/GetRandomBytes"
        finally:
            source.close()

    def test_custom_stream_method_path(self) -> None:
        """channel.stream_stream() should use the configured stream method path."""
        source, mock_channel = make_mocked_source(
            grpc_mode="server_streaming",
            grpc_stream_method_path="/custom.Service/StreamData",
        )
        try:
            # First stream handle = the entropy path (the draw stream follows).
            call_args = mock_channel.stream_stream.call_args_list[0]
            assert call_args[0][0] == "/custom.Service/StreamData"
        finally:
            source.close()


class TestQuotaExhaustedClassification:
    """RESOURCE_EXHAUSTED is a quota verdict, not a connectivity one."""

    def test_classifier_direct_and_cause_chain(self) -> None:
        from qr_sampler.entropy.qgrpc.source import _is_quota_exhausted

        assert _is_quota_exhausted(FakeRpcError("RESOURCE_EXHAUSTED")) is True
        wrapper = RuntimeError("wrapped")
        wrapper.__cause__ = FakeRpcError("RESOURCE_EXHAUSTED")
        assert _is_quota_exhausted(wrapper) is True
        assert _is_quota_exhausted(FakeRpcError("UNAVAILABLE")) is False
        assert _is_quota_exhausted(RuntimeError("no code attr")) is False
        assert _is_quota_exhausted(None) is False

    def test_quota_error_surfaces_distinct_message_and_event(self, caplog: Any) -> None:
        """A RESOURCE_EXHAUSTED unary failure must say 'quota', not 'tunnel'."""
        handle = AsyncMock(side_effect=FakeRpcError("RESOURCE_EXHAUSTED"))
        source, _ = make_mocked_source(
            unary_handle=handle, ensure=False, grpc_mode="unary", grpc_retry_count=0
        )
        try:
            with (
                caplog.at_level("ERROR", logger="qr_sampler"),
                pytest.raises(EntropyUnavailableError) as exc_info,
            ):
                source.get_random_bytes(100)
            assert "quota exhausted" in str(exc_info.value)
            assert "connectivity is fine" in str(exc_info.value)
            events = [r.__dict__.get("event") for r in caplog.records]
            assert "qrng.quota_exhausted" in events
        finally:
            source.close()

    def test_sample_count_above_request_cap_warns_at_init(self, caplog: Any) -> None:
        cap = make_config().qrng_max_bytes_per_request
        with caplog.at_level("WARNING", logger="qr_sampler"):
            source, _ = make_mocked_source(ensure=False, sample_count=cap + 1)
        source.close()
        events = [r.__dict__.get("event") for r in caplog.records]
        assert "qrng.sample_count_exceeds_request_cap" in events

    def test_sample_count_at_cap_is_silent(self, caplog: Any) -> None:
        cap = make_config().qrng_max_bytes_per_request
        with caplog.at_level("WARNING", logger="qr_sampler"):
            source, _ = make_mocked_source(ensure=False, sample_count=cap)
        source.close()
        events = [r.__dict__.get("event") for r in caplog.records]
        assert "qrng.sample_count_exceeds_request_cap" not in events


class TestServerRejectionChannelReset:
    """2026-07: a server-side rejection (FAILED_PRECONDITION etc.) leaves the
    channel healthy, so the per-token teardown/rebuild is skipped; a transport
    failure still resets the (likely stale) channel."""

    def test_classifier_direct_and_cause_chain(self) -> None:
        from qr_sampler.entropy.qgrpc.source import _is_server_side_rejection

        assert _is_server_side_rejection(FakeRpcError("FAILED_PRECONDITION")) is True
        assert _is_server_side_rejection(FakeRpcError("INVALID_ARGUMENT")) is True
        wrapper = RuntimeError("wrapped")
        wrapper.__cause__ = FakeRpcError("FAILED_PRECONDITION")
        assert _is_server_side_rejection(wrapper) is True
        # transport-level + non-gRPC errors are NOT server rejections
        assert _is_server_side_rejection(FakeRpcError("UNAVAILABLE")) is False
        assert _is_server_side_rejection(FakeRpcError("DEADLINE_EXCEEDED")) is False
        assert _is_server_side_rejection(RuntimeError("no code attr")) is False
        assert _is_server_side_rejection(None) is False

    def _source_failing_with(self, exc: Exception) -> Any:
        handle = AsyncMock(side_effect=exc)
        return make_mocked_source(unary_handle=handle, grpc_mode="unary", grpc_retry_count=0)[0]

    def test_server_rejection_does_not_reset_channel(self) -> None:
        exc = FakeRpcError("FAILED_PRECONDITION")
        source = self._source_failing_with(exc)
        resets: list[int] = []
        source._reset_channel = lambda: resets.append(1)

        def _raise(n: int) -> bytes:
            raise exc

        source._fetch_sync = _raise
        try:
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)
            assert resets == [], "a healthy-channel server rejection must not rebuild the channel"
        finally:
            source.close()

    def test_transport_failure_still_resets_channel(self) -> None:
        exc = FakeRpcError("UNAVAILABLE")
        source = self._source_failing_with(exc)
        resets: list[int] = []
        source._reset_channel = lambda: resets.append(1)

        def _raise(n: int) -> bytes:
            raise exc

        source._fetch_sync = _raise
        try:
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)
            assert resets == [1], "a transport failure should still reset the likely-stale channel"
        finally:
            source.close()
