"""Tests for the server-integrated draw transport (``fetch_draw``).

Covers the draw request/reply codec seam (``encode_draw_request`` /
``decode_draw_reply``) and the two draw dispatch modes (unary and bidi
streaming) over fully mocked gRPC channels, including decode failure
paths and commitment-nonce echo.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from qr_sampler.entropy.qgrpc.transport import decode_draw_reply, encode_draw_request
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.proto.purity_service_pb2 import DrawRequest, DrawResponse
from tests.test_entropy.qgrpc_util import make_mocked_source

pytest.importorskip("grpc.aio", reason="grpcio not installed")


def _draw_response_wire(**fields: Any) -> bytes:
    """Wire bytes for a DrawResponse built from keyword fields."""
    return DrawResponse(**fields).SerializeToString()


def _fetch_draw_sync(
    source: Any, source_id: str, block_bytes: int, nonce: int = 0, timeout: float = 5.0
) -> Any:
    """Dispatch ``fetch_draw`` onto the source's background loop and block."""
    source._channel.ensure()
    future = source._channel.submit(source._transport.fetch_draw(source_id, block_bytes, nonce))
    return future.result(timeout=timeout)


class FakeDrawBidiCall:
    """Write-driven fake bidi call speaking the draw wire format.

    Each ``write()`` parses the DrawRequest and enqueues one DrawResponse
    echoing its ``sequence_id``; ``read()`` blocks until a response is
    available (mirrors ``qgrpc_util.FakeBidiCall`` for the byte path).
    """

    def __init__(self, u: float | None = 0.5, echo_sequence: bool = True) -> None:
        self._u = u
        self._echo_sequence = echo_sequence
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.write_count = 0
        self.requests: list[DrawRequest] = []
        self.cancelled = False

    async def write(self, request: bytes) -> None:
        self.write_count += 1
        if self._u is None:
            await self._queue.put(None)  # stream end
            return
        req = DrawRequest.FromString(request)
        self.requests.append(req)
        response = DrawResponse(
            u=self._u,
            z=0.25,
            sequence_id=req.sequence_id if self._echo_sequence else 0,
            source_id=req.source_id or "bound-source",
            integrator="bit_z",
        )
        await self._queue.put(response.SerializeToString())

    async def read(self) -> bytes | None:
        return await self._queue.get()

    def cancel(self) -> None:
        self.cancelled = True


# ---------------------------------------------------------------------------
# Codec seam
# ---------------------------------------------------------------------------


class TestEncodeDrawRequest:
    """``encode_draw_request`` — the single draw request encoder."""

    def test_all_defaults_is_empty(self) -> None:
        """Server-default draw (key binding, default block) is empty wire."""
        assert encode_draw_request("", 0) == b""
        assert encode_draw_request("", 0, 0) == b""

    def test_nonce_rides_sequence_id(self) -> None:
        msg = DrawRequest.FromString(encode_draw_request("dev", 1024, 77))
        assert msg.sequence_id == 77
        assert msg.source_id == "dev"
        assert msg.block_bytes == 1024

    def test_zero_nonce_is_byte_identical_to_nonce_less(self) -> None:
        assert encode_draw_request("dev", 64) == encode_draw_request("dev", 64, 0)


class TestDecodeDrawReply:
    """``decode_draw_reply`` — the single draw response decoder."""

    def test_decode_extracts_all_fields(self) -> None:
        wire = _draw_response_wire(
            u=0.734,
            z=0.625,
            sequence_id=42,
            generation_timestamp_ns=123456789,
            source_id="dragonfly-0",
            coherence_z=4.5,
            coherence_valid=True,
            purity_label="quantum/intact/raw/qf:99+",
            integrated_bytes=2_097_152,
            integrator="bit_z",
            coherence_r=0.25,
        )
        msg = decode_draw_reply(wire)
        assert msg.u == 0.734
        assert msg.z == 0.625
        assert msg.sequence_id == 42
        assert msg.generation_timestamp_ns == 123456789
        assert msg.source_id == "dragonfly-0"
        assert msg.coherence_z == 4.5
        assert msg.coherence_valid is True
        assert msg.purity_label == "quantum/intact/raw/qf:99+"
        assert msg.integrated_bytes == 2_097_152
        assert msg.integrator == "bit_z"
        assert msg.coherence_r == 0.25

    def test_absent_u_raises(self) -> None:
        """A response without field 1 (u) is 'no draw served'."""
        with pytest.raises(EntropyUnavailableError, match=r"field 1 \(u\)"):
            decode_draw_reply(_draw_response_wire(z=1.5, sequence_id=7))
        with pytest.raises(EntropyUnavailableError, match=r"field 1 \(u\)"):
            decode_draw_reply(b"")

    def test_u_zero_raises(self) -> None:
        """Explicit u=0.0 is indistinguishable from absent — refused.

        The server clamps every served u to (1e-10, 1-1e-10), so 0.0 can
        never be a legitimate draw.
        """
        # Hand-craft field 1 fixed64 = 0.0 (SerializeToString would omit it).
        wire = b"\x09" + b"\x00" * 8
        with pytest.raises(EntropyUnavailableError, match=r"field 1 \(u\)"):
            decode_draw_reply(wire)

    def test_clamp_boundaries_accepted(self) -> None:
        for u in (1e-10, 1.0 - 1e-10):
            assert decode_draw_reply(_draw_response_wire(u=u)).u == u


# ---------------------------------------------------------------------------
# Unary dispatch
# ---------------------------------------------------------------------------


class TestFetchDrawUnary:
    """``fetch_draw`` in unary mode against a mocked channel."""

    def _make_echoing_source(self, u: float = 0.5) -> tuple[Any, AsyncMock]:
        """Source whose draw method echoes the request's sequence_id."""

        async def _serve(request_bytes: bytes, **_kwargs: Any) -> bytes:
            req = DrawRequest.FromString(request_bytes)
            return _draw_response_wire(
                u=u,
                z=1.25,
                sequence_id=req.sequence_id,
                source_id=req.source_id or "bound-source",
                integrated_bytes=req.block_bytes or 2_097_152,
                integrator="bit_z",
            )

        handle = AsyncMock(side_effect=_serve)
        source, _ = make_mocked_source(unary_handle=handle)
        return source, handle

    def test_fetch_draw_returns_decoded_response(self) -> None:
        source, handle = self._make_echoing_source(u=0.734)
        try:
            reply = _fetch_draw_sync(source, "dragonfly-0", 2_097_152)
            assert reply.response.u == 0.734
            assert reply.response.z == 1.25
            assert reply.response.source_id == "dragonfly-0"
            assert reply.response.integrator == "bit_z"
            assert reply.elapsed_ms >= 0.0
            # The wire request carried the source id + block size.
            sent = DrawRequest.FromString(handle.call_args[0][0])
            assert sent.source_id == "dragonfly-0"
            assert sent.block_bytes == 2_097_152
            assert sent.sequence_id == 0
        finally:
            source.close()

    def test_nonce_echo(self) -> None:
        """The commitment nonce rides sequence_id and is echoed verbatim."""
        source, handle = self._make_echoing_source()
        try:
            nonce = 0x7FEDCBA987654321
            reply = _fetch_draw_sync(source, "dev", 1024, nonce=nonce)
            assert reply.response.sequence_id == nonce
            sent = DrawRequest.FromString(handle.call_args[0][0])
            assert sent.sequence_id == nonce
        finally:
            source.close()

    def test_decode_failure_raises(self) -> None:
        """A reply without u surfaces as EntropyUnavailableError."""
        handle = AsyncMock(return_value=_draw_response_wire(z=2.0))
        source, _ = make_mocked_source(unary_handle=handle)
        try:
            with pytest.raises(EntropyUnavailableError, match=r"field 1 \(u\)"):
                _fetch_draw_sync(source, "dev", 64)
        finally:
            source.close()

    def test_rpc_failure_propagates(self) -> None:
        handle = AsyncMock(side_effect=RuntimeError("boom"))
        source, _ = make_mocked_source(unary_handle=handle)
        try:
            with pytest.raises(RuntimeError, match="boom"):
                _fetch_draw_sync(source, "dev", 64)
        finally:
            source.close()

    def test_empty_draw_method_path_raises(self) -> None:
        """grpc_draw_method_path='' disables the draw handle."""
        source, _ = make_mocked_source(grpc_draw_method_path="")
        try:
            with pytest.raises(EntropyUnavailableError, match="Draw method not initialized"):
                _fetch_draw_sync(source, "dev", 64)
        finally:
            source.close()

    def test_server_streaming_mode_refused(self) -> None:
        """Draws define no server-streaming shape."""
        source, _ = make_mocked_source(stream_handle=MagicMock(), grpc_mode="server_streaming")
        try:
            with pytest.raises(EntropyUnavailableError, match="does not support draws"):
                _fetch_draw_sync(source, "dev", 64)
        finally:
            source.close()


# ---------------------------------------------------------------------------
# Bidi dispatch
# ---------------------------------------------------------------------------


class TestFetchDrawBidi:
    """``fetch_draw`` in bidi_streaming mode over a fake draw stream."""

    def _make_bidi_source(self, fake_call: FakeDrawBidiCall) -> tuple[Any, MagicMock]:
        stream_handle = MagicMock(return_value=fake_call)
        source, _ = make_mocked_source(stream_handle=stream_handle, grpc_mode="bidi_streaming")
        return source, stream_handle

    def test_fetch_draw_returns_decoded_response(self) -> None:
        fake_call = FakeDrawBidiCall(u=0.618)
        source, _ = self._make_bidi_source(fake_call)
        try:
            reply = _fetch_draw_sync(source, "dragonfly-0", 1024)
            assert reply.response.u == 0.618
            assert reply.response.source_id == "dragonfly-0"
            assert fake_call.requests[0].block_bytes == 1024
        finally:
            source.close()

    def test_session_is_reused(self) -> None:
        fake_call = FakeDrawBidiCall()
        source, stream_handle = self._make_bidi_source(fake_call)
        try:
            _fetch_draw_sync(source, "dev", 64)
            _fetch_draw_sync(source, "dev", 64)
            # One stream established, two requests written over it.
            assert stream_handle.call_count == 1
            assert fake_call.write_count == 2
        finally:
            source.close()

    def test_concurrent_draws_correlate_by_nonce(self) -> None:
        """Two in-flight draws must each get their own echoed response."""
        fake_call = FakeDrawBidiCall()
        source, _ = self._make_bidi_source(fake_call)
        try:
            source._channel.ensure()
            fut1 = source._channel.submit(source._transport.fetch_draw("dev", 64, 1111))
            fut2 = source._channel.submit(source._transport.fetch_draw("dev", 64, 2222))
            # Redeem out of order: correlation must not mix them up.
            reply2 = fut2.result(timeout=5.0)
            reply1 = fut1.result(timeout=5.0)
            assert reply1.response.sequence_id == 1111
            assert reply2.response.sequence_id == 2222
        finally:
            source.close()

    def test_stream_end_raises_and_resets_session(self) -> None:
        fake_call = FakeDrawBidiCall(u=None)  # stream ends on first write
        source, _ = self._make_bidi_source(fake_call)
        try:
            with pytest.raises(EntropyUnavailableError):
                _fetch_draw_sync(source, "dev", 64)
            assert source._transport._draw_bidi_session is None
        finally:
            source.close()

    def test_draw_and_byte_sessions_are_independent(self) -> None:
        """The draw stream rides its own session, not the byte stream's."""
        from tests.test_entropy.qgrpc_util import FakeBidiCall

        byte_call = FakeBidiCall(b"\xcc" * 64)
        draw_call = FakeDrawBidiCall()
        # First stream established = byte session, second = draw session.
        stream_handle = MagicMock(side_effect=[byte_call, draw_call])
        source, _ = make_mocked_source(stream_handle=stream_handle, grpc_mode="bidi_streaming")
        try:
            assert source.get_random_bytes(64) == b"\xcc" * 64
            reply = _fetch_draw_sync(source, "dev", 64)
            assert reply.response.u == 0.5
            transport = source._transport
            assert transport._bidi_session is not None
            assert transport._draw_bidi_session is not None
            assert transport._bidi_session is not transport._draw_bidi_session
        finally:
            source.close()

    def test_empty_draw_stream_path_raises(self) -> None:
        source, _ = make_mocked_source(
            stream_handle=MagicMock(return_value=FakeDrawBidiCall()),
            grpc_mode="bidi_streaming",
            grpc_draw_stream_method_path="",
        )
        try:
            with pytest.raises(EntropyUnavailableError, match="Draw stream method"):
                _fetch_draw_sync(source, "dev", 64)
        finally:
            source.close()
