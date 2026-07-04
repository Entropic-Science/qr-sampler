"""Background asyncio loop + gRPC channel lifecycle for the entropy source.

Owns the dedicated event-loop thread and the ``grpc.aio`` channel with its
generic method handles. Everything here is lazily initialized on first use
— constructing a :class:`GrpcChannel` opens no sockets and spawns no
threads, which keeps ``import qr_sampler`` (and adapter construction)
100% side-effect-free.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from concurrent.futures import Future

logger = logging.getLogger("qr_sampler")


def identity_serializer(request: bytes) -> bytes:
    """Pass-through serializer for pre-encoded request bytes.

    The transport encodes the request as raw protobuf bytes before calling
    the gRPC method handle, so the serializer is an identity function.
    """
    return request


def identity_deserializer(data: bytes) -> bytes:
    """Pass-through deserializer that returns raw response bytes.

    The transport decodes the raw wire-format bytes itself after receiving
    them (see :func:`qr_sampler.entropy.qgrpc.transport.decode_reply`).
    """
    return data


class GrpcChannel:
    """Lazily-initialized gRPC async channel on a background event loop.

    Args:
        address: gRPC server address (``host:port`` or ``unix:///path``).
        timeout_ms: Budget for the initial channel bring-up.
        method_path: gRPC method path for the unary RPC.
        stream_method_path: gRPC method path for the streaming RPC
            (empty string disables the streaming handle).
        draw_method_path: gRPC method path for the unary server-draw RPC
            (empty string disables the draw handle).
        draw_stream_method_path: gRPC method path for the bidi-streaming
            server-draw RPC (empty string disables the draw stream handle).
    """

    def __init__(
        self,
        address: str,
        *,
        timeout_ms: float,
        method_path: str,
        stream_method_path: str,
        draw_method_path: str = "",
        draw_stream_method_path: str = "",
    ) -> None:
        self._address = address
        self._timeout_ms = timeout_ms
        self._method_path = method_path
        self._stream_method_path = stream_method_path
        self._draw_method_path = draw_method_path
        self._draw_stream_method_path = draw_stream_method_path

        # Background event loop + gRPC channel are LAZILY initialized on
        # the first fetch. The prior eager init opened a thread + an
        # HTTP/2 channel object at construction time of any module that
        # built this source; deferring to first use means the channel is
        # created when the network is actually needed.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._grpc_channel: Any = None
        self._unary_method: Any = None
        self._stream_method: Any | None = None
        self._draw_unary_method: Any | None = None
        self._draw_stream_method: Any | None = None

    @property
    def initialized(self) -> bool:
        """Whether the loop + channel are currently up."""
        return self._initialized

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The background event loop (None until :meth:`ensure`)."""
        return self._loop

    @property
    def unary_method(self) -> Any:
        """The generic unary method handle (None until :meth:`ensure`)."""
        return self._unary_method

    @property
    def stream_method(self) -> Any | None:
        """The generic streaming method handle (None when path is empty)."""
        return self._stream_method

    @property
    def draw_unary_method(self) -> Any | None:
        """The generic unary draw method handle (None when path is empty)."""
        return self._draw_unary_method

    @property
    def draw_stream_method(self) -> Any | None:
        """The generic draw stream method handle (None when path is empty)."""
        return self._draw_stream_method

    def ensure(self) -> None:
        """Start the background loop + open the gRPC channel on first call.

        Idempotent. Thread-safe via a lock so concurrent first calls from
        multiple sampling threads don't double-spawn the loop.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="qr-sampler-grpc-loop",
            )
            self._thread.start()

            future = asyncio.run_coroutine_threadsafe(self._init_channel(), self._loop)
            future.result(timeout=self._timeout_ms / 1000.0)
            self._initialized = True

    def submit(self, coro: Coroutine[Any, Any, Any]) -> Future[Any]:
        """Dispatch a coroutine onto the background loop.

        Callers must have run :meth:`ensure` first.
        """
        assert self._loop is not None  # ensure() sets it
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_loop(self) -> None:
        """Run the asyncio event loop in the background thread."""
        assert self._loop is not None  # set in ensure()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _init_channel(self) -> None:
        """Create the gRPC async channel and generic method handles."""
        import grpc.aio

        # gRPC keepalive: tuned to respect the server's default
        # ``min_ping_interval_without_data_ms = 300_000`` (5 min) policy.
        # Earlier config (30 s + permit_without_calls=True +
        # max_pings_without_data=0) pinged so aggressively that the QRNG
        # server / tunnel front replied ``GOAWAY ENHANCE_YOUR_CALM
        # (too_many_pings)`` and dropped the channel mid-call (Errno 11
        # BlockingIOError surfaced from grpcio's PollerCompletionQueue
        # was the visible symptom). Per-token traffic keeps the channel
        # busy anyway so ``permit_without_calls=False`` is harmless on
        # the hot path; we still get a single keepalive on long idle
        # gaps because the timer fires after ``keepalive_time_ms``
        # elapses with no data.
        options = [
            ("grpc.keepalive_time_ms", 300_000),
            ("grpc.keepalive_timeout_ms", 20_000),
            ("grpc.keepalive_permit_without_calls", False),
        ]

        self._grpc_channel = grpc.aio.insecure_channel(self._address, options=options)

        # Generic unary method handle — works with any proto that uses
        # field 1 varint (request) and field 1 bytes (response).
        self._unary_method = self._grpc_channel.unary_unary(
            self._method_path,
            request_serializer=identity_serializer,
            response_deserializer=identity_deserializer,
        )

        # Generic streaming method handle — only created when path is non-empty.
        self._stream_method = None
        if self._stream_method_path:
            self._stream_method = self._grpc_channel.stream_stream(
                self._stream_method_path,
                request_serializer=identity_serializer,
                response_deserializer=identity_deserializer,
            )

        # Server-draw method handles (PurityService) — only created when the
        # respective path is non-empty. Same identity codecs: the transport
        # pre-encodes DrawRequest bytes and decodes DrawResponse bytes itself.
        self._draw_unary_method = None
        if self._draw_method_path:
            self._draw_unary_method = self._grpc_channel.unary_unary(
                self._draw_method_path,
                request_serializer=identity_serializer,
                response_deserializer=identity_deserializer,
            )
        self._draw_stream_method = None
        if self._draw_stream_method_path:
            self._draw_stream_method = self._grpc_channel.stream_stream(
                self._draw_stream_method_path,
                request_serializer=identity_serializer,
                response_deserializer=identity_deserializer,
            )

    def reset(self) -> None:
        """Tear down the channel + loop so the next call re-inits cleanly.

        Used when a verification fetch (or retry exhaustion) surfaces a
        stale channel. Best-effort: errors during teardown are swallowed
        since the goal is just to flip ``initialized`` back to False.
        """
        with self._init_lock:
            if not self._initialized:
                return
            loop = self._loop
            channel = self._grpc_channel
            if loop is not None and channel is not None:
                try:
                    fut = asyncio.run_coroutine_threadsafe(channel.close(), loop)
                    fut.result(timeout=2.0)
                except Exception:
                    pass
            if loop is not None:
                with contextlib.suppress(Exception):
                    loop.call_soon_threadsafe(loop.stop)
            self._initialized = False
            self._loop = None
            self._thread = None
            self._grpc_channel = None
            self._unary_method = None
            self._stream_method = None
            self._draw_unary_method = None
            self._draw_stream_method = None

    def close(self, pre_close: Callable[[], None] | None = None) -> None:
        """Release the channel, event loop, and background thread.

        No-op when the channel was never opened (the source was
        constructed but never used).

        Args:
            pre_close: Optional loop-thread callback executed before the
                channel closes (the transport uses it to shut down its
                bidi session).
        """
        if not self._initialized:
            return

        loop = self._loop
        thread = self._thread
        assert loop is not None and thread is not None  # ensure() set both

        async def _shutdown() -> None:
            if pre_close is not None:
                pre_close()
            await self._grpc_channel.close()

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=5.0)
        except Exception:
            logger.warning("Error during gRPC channel cleanup", exc_info=True)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)
