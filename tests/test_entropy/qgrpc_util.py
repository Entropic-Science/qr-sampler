"""Shared helpers for the qgrpc entropy-source test modules."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from qr_sampler.proto.wire import encode_varint


def make_config(**overrides: Any) -> Any:
    """Create a real QRSamplerConfig with gRPC defaults."""
    from qr_sampler.config import QRSamplerConfig

    defaults = {
        "grpc_server_address": "localhost:50051",
        "grpc_timeout_ms": 5000.0,
        "grpc_retry_count": 2,
        "grpc_mode": "unary",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


def encode_mock_response(data: bytes) -> bytes:
    """Encode a mock protobuf response with field 1 = length-delimited bytes."""
    # Tag: field 1, wire type 2 (length-delimited) = (1 << 3) | 2 = 0x0a
    return b"\x0a" + encode_varint(len(data)) + data


def encode_mock_response_with_echo(data: bytes, sequence_id: int) -> bytes:
    """Mock response with field 1 payload + field 2 sequence echo."""
    return encode_mock_response(data) + b"\x10" + encode_varint(sequence_id)


def extract_request_sequence(request: bytes) -> int:
    """Parse field 2 (varint) from a wire-encoded entropy request."""
    offset = 0
    while offset < len(request):
        tag = request[offset]
        offset += 1
        field_number = tag >> 3
        value = 0
        shift = 0
        while True:
            b = request[offset]
            value |= (b & 0x7F) << shift
            offset += 1
            if not (b & 0x80):
                break
            shift += 7
        if field_number == 2:
            return value
    return 0


class FakeBidiCall:
    """Write-driven fake bidi call.

    Each ``write()`` enqueues one response; ``read()`` blocks until a
    response is available. This models real stream semantics (the
    ``_BidiSession`` reader task must block between responses, not spin)
    in a way ``AsyncMock(return_value=...)`` cannot.
    """

    def __init__(self, payload: bytes | None, echo_sequence: bool = True) -> None:
        import asyncio

        self._payload = payload
        self._echo_sequence = echo_sequence
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.write_count = 0
        self.cancelled = False

    async def write(self, request: bytes) -> None:
        self.write_count += 1
        if self._payload is None:
            await self._queue.put(None)  # stream end
            return
        response = encode_mock_response(self._payload)
        if self._echo_sequence:
            # Echo the request's field-2 sequence_id, if present.
            seq = extract_request_sequence(request)
            if seq:
                response += b"\x10" + encode_varint(seq)
        await self._queue.put(response)

    async def read(self) -> bytes | None:
        return await self._queue.get()

    def cancel(self) -> None:
        self.cancelled = True


def make_mocked_source(
    *,
    unary_handle: Any | None = None,
    stream_handle: Any | None = None,
    ensure: bool = True,
    **config_overrides: Any,
) -> tuple[Any, MagicMock]:
    """Build a QuantumGrpcSource against a fully mocked gRPC channel.

    Returns:
        ``(source, mock_channel)`` — the mock channel exposes
        ``unary_unary`` / ``stream_stream`` MagicMocks for assertions.

    The ``grpc.aio.insecure_channel`` patch stays active until
    ``source.close()`` runs, so lazily-initialized sources
    (``ensure=False``) still land on the mock at first-fetch time.
    """
    import contextlib

    config = make_config(**config_overrides)

    patcher = patch("grpc.aio.insecure_channel")
    mock_channel_fn = patcher.start()
    mock_channel = MagicMock()
    mock_channel_fn.return_value = mock_channel
    mock_channel.unary_unary = MagicMock(return_value=unary_handle or MagicMock())
    mock_channel.stream_stream = MagicMock(return_value=stream_handle or MagicMock())
    # Expose the channel factory mock for address assertions.
    mock_channel.insecure_channel_fn = mock_channel_fn

    from qr_sampler.entropy.qgrpc import QuantumGrpcSource

    source = QuantumGrpcSource(config)

    original_close = source.close

    def close_and_unpatch() -> None:
        original_close()
        with contextlib.suppress(Exception):
            patcher.stop()

    source.close = close_and_unpatch  # type: ignore[method-assign]
    if ensure:
        source._channel.ensure()
    return source, mock_channel


def make_bidi_source(payload: bytes | None, **config_overrides: Any) -> tuple[Any, Any]:
    """Build a bidi-mode source whose stream is a ``FakeBidiCall``."""
    fake_call = FakeBidiCall(payload)
    stream_handle = MagicMock(return_value=fake_call)
    source, _ = make_mocked_source(
        stream_handle=stream_handle,
        grpc_mode="bidi_streaming",
        **config_overrides,
    )
    return source, stream_handle


class FakeStatusCode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeRpcError(Exception):
    """Duck-types grpc.aio.AioRpcError's .code() without importing grpc."""

    def __init__(self, code_name: str) -> None:
        super().__init__(f"fake rpc error ({code_name})")
        self._code = FakeStatusCode(code_name)

    def code(self) -> FakeStatusCode:
        return self._code
