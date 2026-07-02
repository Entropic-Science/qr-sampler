"""Wire transport for the gRPC entropy source: encode, dispatch, decode.

One request/reply codec (the :mod:`qr_sampler.proto` message stubs — the
package's single wire format) plus the three transport modes:

- **Unary**: simple request-response. One HTTP/2 stream per call.
- **Server streaming**: one config request, one streamed response per call.
- **Bidirectional streaming**: persistent stream with response correlation.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.proto.entropy_service_pb2 import EntropyRequest, EntropyResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.entropy.qgrpc.channel import GrpcChannel


def encode_request(n: int, nonce: int = 0) -> bytes:
    """Encode one entropy request: field 1 = byte count, field 2 = nonce.

    Produces standard protobuf wire bytes via
    :class:`~qr_sampler.proto.entropy_service_pb2.EntropyRequest` — valid
    for any server whose request proto carries the byte count as field 1
    (varint), e.g. both ``EntropyRequest(bytes_needed=n)`` and
    ``RandomRequest(num_bytes=n)``. The pipelined fetch path uses the
    ``sequence_id`` field as a commitment nonce derived from the previously
    selected token; per proto3 semantics a zero value is omitted from the
    wire, so serial fetches produce byte-identical requests to a
    nonce-less encoder.
    """
    return EntropyRequest(bytes_needed=n, sequence_id=nonce).SerializeToString()


def decode_reply(raw: bytes) -> tuple[bytes, int, int]:
    """Decode one entropy response into (payload, sequence_id, gen_ts).

    Decodes via :class:`~qr_sampler.proto.entropy_service_pb2
    .EntropyResponse` — proto3 semantics, so the LAST occurrence of a
    repeated field 1 wins (recorded behavior change #6; byte-identical for
    every real server, which sends field 1 exactly once) and absent varint
    fields decode as 0 (servers that do not echo ``sequence_id`` simply
    yield ``(payload, 0, 0)``).

    Field-number collision with the production qbert server (its
    ``qrng.proto``, 2026-06-10): ``RandomResponse`` defines field 2 as
    ``uint64 timestamp`` (epoch MICROSECONDS) and field 3 as ``string
    device_id``. Against that server, the value this decoder returns in
    the ``sequence_id`` slot is actually the server timestamp — it can
    never equal a commitment nonce, so ``echo_verified`` stays False by
    construction, and the bidi pool's unknown-echo branch FIFO-matches
    (unary responses correlate by HTTP/2 stream regardless). ``device_id``
    is wire-type 2 at field 3 and is skipped cleanly, so the third slot
    decodes as 0. All of this is benign — do NOT "fix" echo verification
    by trusting field 2 against this server; the field is occupied.

    Args:
        raw: Raw protobuf wire-format bytes.

    Returns:
        Tuple of (payload_bytes, sequence_id, generation_timestamp_ns).

    Raises:
        EntropyUnavailableError: If the payload (field 1) is missing or
            empty.
    """
    msg = EntropyResponse.FromString(raw)
    if not msg.data:
        raise EntropyUnavailableError("Failed to decode gRPC response: field 1 (bytes) not found")
    return msg.data, msg.sequence_id, msg.generation_timestamp_ns


class _FetchReply:
    """Decoded result of one entropy fetch, with true call latency.

    ``elapsed_ms`` is measured inside the background-loop coroutine —
    request write to response decode — so it reflects actual network +
    server time regardless of how long the engine later blocks waiting
    for it (which, on the pipelined path, is ideally ~0).
    """

    __slots__ = ("elapsed_ms", "generation_timestamp_ns", "payload", "sequence_id")

    def __init__(
        self,
        payload: bytes,
        sequence_id: int,
        generation_timestamp_ns: int,
        elapsed_ms: float,
    ) -> None:
        self.payload = payload
        self.sequence_id = sequence_id
        self.generation_timestamp_ns = generation_timestamp_ns
        self.elapsed_ms = elapsed_ms


class _BidiSession:
    """One persistent bidirectional stream with response correlation.

    All methods run on the source's background asyncio loop (single
    thread), so no locking is needed beyond the write serializer.

    A dedicated reader task drains the stream and resolves pending
    futures. Responses are matched by ``sequence_id`` echo when the
    server provides one; servers that do not echo (``sequence_id == 0``)
    fall back to FIFO matching, which is sound because HTTP/2 preserves
    per-stream ordering. This replaces the previous write-then-read
    pattern, which interleaved incorrectly when more than one fetch was
    in flight (a state the pipelined prefetch path makes routine).
    """

    def __init__(self, call: Any, loop: asyncio.AbstractEventLoop) -> None:
        self._call = call
        self._loop = loop
        self._write_lock = asyncio.Lock()
        # Insertion-ordered: doubles as the FIFO queue for no-echo servers.
        self._pending: dict[int, asyncio.Future[tuple[bytes, int, int]]] = {}
        self._fifo_counter = 0  # synthetic keys for nonce-less requests
        self.dead = False
        self._reader_task = loop.create_task(self._read_loop())

    async def request(self, n: int, nonce: int) -> tuple[bytes, int, int]:
        """Send one entropy request and await its correlated response."""
        if self.dead:
            raise EntropyUnavailableError("Bidi stream is closed")
        key = nonce
        if key == 0:
            # Negative synthetic keys can never collide with real nonces
            # (which are positive 63-bit values).
            self._fifo_counter -= 1
            key = self._fifo_counter
        fut: asyncio.Future[tuple[bytes, int, int]] = self._loop.create_future()
        self._pending[key] = fut
        try:
            async with self._write_lock:
                await self._call.write(encode_request(n, nonce))
        except Exception:
            self._pending.pop(key, None)
            raise
        return await fut

    async def _read_loop(self) -> None:
        error: Exception
        try:
            while True:
                raw = await self._call.read()
                if raw is None:
                    error = EntropyUnavailableError("Bidi stream ended unexpectedly")
                    break
                payload, seq, gen_ts = decode_reply(raw)
                fut = self._pending.pop(seq, None) if seq else None
                if fut is None and self._pending:
                    # No echo (or unknown echo): FIFO-match oldest pending.
                    oldest_key = next(iter(self._pending))
                    fut = self._pending.pop(oldest_key)
                if fut is not None and not fut.done():
                    fut.set_result((payload, seq, gen_ts))
                # else: response for a cancelled/unknown request — drop.
        except Exception as exc:  # reader died — fail everything pending
            error = exc
        self.dead = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(EntropyUnavailableError(f"Bidi stream failed: {error}"))
        self._pending.clear()

    def close(self) -> None:
        """Cancel the reader task and the underlying call. Loop-thread only."""
        self.dead = True
        with contextlib.suppress(Exception):
            self._reader_task.cancel()
        with contextlib.suppress(Exception):
            self._call.cancel()


class GrpcTransport:
    """Mode dispatch (unary / server-streaming / bidi) over a GrpcChannel.

    Args:
        channel: The lazily-initialized channel + loop owner.
        mode: One of ``'unary'``, ``'server_streaming'``, ``'bidi_streaming'``.
        metadata: gRPC call metadata (API-key auth), or empty tuple.
        timeout_ms_provider: Callable yielding the current adaptive
            timeout in milliseconds (the circuit breaker's).
    """

    def __init__(
        self,
        channel: GrpcChannel,
        *,
        mode: str,
        metadata: tuple[tuple[str, str], ...],
        timeout_ms_provider: Callable[[], float],
    ) -> None:
        self._channel = channel
        self._mode = mode
        self._metadata = metadata
        self._timeout_ms = timeout_ms_provider
        # Streaming state (lazily initialized on the loop thread).
        self._bidi_session: _BidiSession | None = None

    async def fetch(self, n: int, nonce: int = 0) -> _FetchReply:
        """Route to the appropriate transport mode; measure true call time.

        Elapsed time is captured here, on the background loop, so it
        reflects network + server latency for the call itself — not how
        long a (possibly much later) redeemer blocked waiting for it.
        """
        t0 = time.perf_counter()
        if self._mode == "unary":
            payload, seq, gen_ts = await self._fetch_unary(n, nonce)
        elif self._mode == "server_streaming":
            payload, seq, gen_ts = await self._fetch_server_streaming(n, nonce)
        elif self._mode == "bidi_streaming":
            payload, seq, gen_ts = await self._fetch_bidi_streaming(n, nonce)
        else:
            raise EntropyUnavailableError(f"Unknown gRPC mode: {self._mode!r}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return _FetchReply(payload, seq, gen_ts, elapsed_ms)

    async def _fetch_unary(self, n: int, nonce: int = 0) -> tuple[bytes, int, int]:
        """Single request-response per call. Simplest. Higher overhead."""
        request_bytes = encode_request(n, nonce)
        timeout_s = self._timeout_ms() / 1000.0
        raw_response: bytes = await self._channel.unary_method(
            request_bytes,
            timeout=timeout_s,
            metadata=self._metadata or None,
        )
        return decode_reply(raw_response)

    async def _fetch_server_streaming(self, n: int, nonce: int = 0) -> tuple[bytes, int, int]:
        """Use the streaming RPC in a request/response style.

        Sends one request and reads one response from the stream.
        The stream is re-established on each call for server-streaming semantics.
        """
        request_bytes = encode_request(n, nonce)

        async def request_iterator() -> Any:
            yield request_bytes

        stream_method = self._channel.stream_method
        if stream_method is None:  # pragma: no cover — validated at source init
            raise EntropyUnavailableError("Stream method not initialized")
        call = stream_method(request_iterator(), metadata=self._metadata or None)
        raw_response: bytes | None = await call.read()
        if raw_response is None:
            raise EntropyUnavailableError("Server stream ended unexpectedly")
        call.cancel()
        return decode_reply(raw_response)

    async def _fetch_bidi_streaming(self, n: int, nonce: int = 0) -> tuple[bytes, int, int]:
        """Fetch over one persistent, correlation-safe bidirectional stream.

        The ``_BidiSession`` is lazily created on first call and reused;
        its reader task matches responses to requests by ``sequence_id``
        echo (FIFO for servers that don't echo), which makes concurrent
        in-flight fetches — routine on the pipelined prefetch path — safe.
        If the stream breaks, the session is discarded and re-established
        on the next call.
        """
        try:
            session = self._bidi_session
            if session is None or session.dead:
                stream_method = self._channel.stream_method
                if stream_method is None:  # pragma: no cover — validated at source init
                    raise EntropyUnavailableError("Stream method not initialized")
                call = stream_method(metadata=self._metadata or None)
                loop = self._channel.loop
                assert loop is not None  # running on the loop already
                session = _BidiSession(call, loop)
                self._bidi_session = session
            return await session.request(n, nonce)
        except EntropyUnavailableError:
            self._bidi_session = None
            raise
        except Exception:
            # Stream broken — reset for next call.
            self._bidi_session = None
            raise

    def clear_bidi(self) -> None:
        """Forget the bidi session (after a channel reset tore down its loop)."""
        self._bidi_session = None

    def close_bidi(self) -> None:
        """Shut down the bidi session, if any. Loop-thread only."""
        if self._bidi_session is not None:
            self._bidi_session.close()
            self._bidi_session = None
