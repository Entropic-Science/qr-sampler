"""Protocol-agnostic gRPC entropy source with configurable transport modes.

This is the primary production entropy source. It fetches random bytes from
a remote entropy server over gRPC, supporting three transport modes:

- **Unary**: simple request-response. One HTTP/2 stream per call.
- **Server streaming**: client sends one config request, server streams responses.
- **Bidirectional streaming**: persistent stream with lowest latency.

The source is **protocol-agnostic**: it uses configurable gRPC method paths
and generic protobuf wire-format helpers rather than hard-coded stubs. This
allows it to connect to any gRPC entropy server (e.g. ``qr_entropy.EntropyService``,
``qrng.QuantumRNG``, or any custom proto) as long as the request encodes the
byte count as protobuf field 1 (varint) and the response returns data as
protobuf field 1 (length-delimited bytes).

All modes satisfy the just-in-time constraint: the gRPC request is sent
only when ``get_random_bytes()`` is called (i.e., after logits are available).

Includes an adaptive circuit breaker that tracks rolling P99 latency and
falls back to a secondary source when the server is slow or unreachable.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

logger = logging.getLogger("qr_sampler")

# TCP-connect pre-probe tunables. The pre-probe fronts every fetch with a
# bounded-time connect() to the gRPC server address; if the kernel does not
# return a listening socket within ``_PREPROBE_TIMEOUT_S``, we raise
# ``EntropyUnavailableError`` immediately and back off for
# ``_PREPROBE_BACKOFF_S`` before probing again. This converts a ~15 s
# (3 retries x ~5 s timeout) "QRNG unreachable" event into a ~500 ms one,
# which keeps the OWUI / httpx streaming client well inside its read budget
# and lets the FallbackEntropySource take over before the user-facing
# request times out.
#
# Operators can opt out by setting ``QR_GRPC_PREPROBE_ENABLED=0`` in the
# environment (default is enabled — failing fast is strictly better than
# the old behaviour for the in-tree deployment).
_PREPROBE_TIMEOUT_S = 0.5
_PREPROBE_BACKOFF_S = 5.0
_PREPROBE_ENABLED_ENV_VAR = "QR_GRPC_PREPROBE_ENABLED"


# ---------------------------------------------------------------------------
# Generic protobuf wire-format helpers
# ---------------------------------------------------------------------------


def _encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint (LEB128).

    Args:
        value: Non-negative integer to encode.

    Returns:
        LEB128-encoded bytes.
    """
    parts: list[int] = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a varint from bytes at the given offset.

    Args:
        data: Raw bytes.
        offset: Starting position.

    Returns:
        Tuple of (decoded_value, new_offset).
    """
    result = 0
    shift = 0
    while True:
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, offset


def _encode_varint_request(n: int) -> bytes:
    """Encode a generic protobuf request with field 1 = varint *n*.

    This produces valid protobuf wire bytes for any message where the
    byte count is field 1 (varint), e.g. both ``EntropyRequest(bytes_needed=n)``
    and ``RandomRequest(num_bytes=n)``.

    Args:
        n: Number of bytes to request (encoded as field 1, wire type 0).

    Returns:
        Serialized protobuf bytes.
    """
    if n == 0:
        return b""
    # Tag: field 1, wire type 0 (varint) = (1 << 3) | 0 = 0x08
    return b"\x08" + _encode_varint(n)


def _decode_bytes_field1(data: bytes) -> bytes:
    """Extract field 1 (length-delimited bytes) from a protobuf message.

    Scans protobuf wire-format bytes for the first occurrence of field 1
    with wire type 2 (length-delimited) and returns its raw bytes payload.
    All other fields are skipped. This works for any response proto where
    field 1 is the data payload (e.g. ``EntropyResponse.data``,
    ``RandomResponse.data``).

    Args:
        data: Raw protobuf wire-format bytes.

    Returns:
        The bytes payload from field 1.

    Raises:
        EntropyUnavailableError: If field 1 is not found or the wire
            format is invalid.
    """
    offset = 0
    while offset < len(data):
        tag, offset = _decode_varint(data, offset)
        field_number = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            # Varint — consume and skip.
            _, offset = _decode_varint(data, offset)
        elif wire_type == 2:
            # Length-delimited.
            length, offset = _decode_varint(data, offset)
            payload = data[offset : offset + length]
            offset += length
            if field_number == 1:
                return payload
        elif wire_type == 5:
            offset += 4  # 32-bit fixed
        elif wire_type == 1:
            offset += 8  # 64-bit fixed
        else:
            break
    raise EntropyUnavailableError("Failed to decode gRPC response: field 1 (bytes) not found")


def _generic_request_serializer(request: bytes) -> bytes:
    """Pass-through serializer for pre-encoded request bytes.

    The generic client encodes the request as raw protobuf bytes before
    calling the gRPC method handle, so the serializer is an identity function.
    """
    return request


def _generic_response_deserializer(data: bytes) -> bytes:
    """Pass-through deserializer that returns raw response bytes.

    The caller extracts field 1 via ``_decode_bytes_field1()`` after
    receiving the raw wire-format bytes.
    """
    return data


# ---------------------------------------------------------------------------
# Source implementation
# ---------------------------------------------------------------------------


@register_entropy_source("quantum_grpc")
class QuantumGrpcSource(EntropySource):
    """Protocol-agnostic gRPC entropy source with configurable transport mode.

    Connects to any gRPC entropy server using configurable method paths and
    generic protobuf wire-format encoding. All modes satisfy the just-in-time
    constraint: the gRPC request is only sent when ``get_random_bytes()`` is
    called (i.e., after logits are available). The transport mode affects
    connection management overhead, not entropy freshness.

    Args:
        config: Sampler configuration with gRPC settings.

    Raises:
        ImportError: If ``grpcio`` is not installed.
        ConfigValidationError: If streaming mode is requested but
            ``grpc_stream_method_path`` is empty.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        try:
            import grpc.aio  # noqa: F401 — availability check
        except ImportError as exc:
            raise ImportError(
                "grpcio is required for QuantumGrpcSource. Install it with: pip install qr-sampler"
            ) from exc

        self._address = config.grpc_server_address
        self._timeout_ms = config.grpc_timeout_ms
        self._retry_count = config.grpc_retry_count
        self._mode = config.grpc_mode
        self._method_path = config.grpc_method_path
        self._stream_method_path = config.grpc_stream_method_path
        self._api_key = config.grpc_api_key
        self._api_key_header = config.grpc_api_key_header
        self._closed = False

        # Validate streaming config upfront.
        if self._mode in ("server_streaming", "bidi_streaming") and not self._stream_method_path:
            raise ConfigValidationError(
                f"grpc_mode={self._mode!r} requires a non-empty grpc_stream_method_path"
            )

        # Build call metadata (empty tuple if no auth).
        self._metadata: tuple[tuple[str, str], ...] = ()
        if self._api_key:
            self._metadata = ((self._api_key_header, self._api_key),)

        # Circuit breaker config.
        self._cb_min_timeout_ms = config.cb_min_timeout_ms
        self._cb_timeout_multiplier = config.cb_timeout_multiplier
        self._cb_recovery_window_s = config.cb_recovery_window_s
        self._cb_max_consecutive_failures = config.cb_max_consecutive_failures

        # Circuit breaker state.
        self._latency_window: deque[float] = deque(maxlen=config.cb_window_size)
        self._p99_ms: float = self._timeout_ms
        self._consecutive_failures: int = 0
        self._circuit_open: bool = False
        self._circuit_open_until: float = 0.0

        # Background event loop + gRPC channel are LAZILY initialized
        # on the first ``get_random_bytes()`` call. Phase 2 (2026-05-21):
        # the prior eager init in __init__ opened a thread + an HTTP/2
        # channel object at import time of any module that constructed
        # this source (e.g. ``qr_sampler.engines.vllm.VLLMAdapter``).
        # Modal's snapshot lifecycle freezes that thread state, then
        # restores it post-snapshot into a container where the
        # cloudflared sidecar is freshly started at ``snap=False`` —
        # not yet reachable, but the captured channel still routes to
        # the stale internal state. Deferring to first use means the
        # channel is created in the post-restore wall-clock, after the
        # sidecar is up.
        #
        # See Phase K research (requirements.md Appendix A) §"qr_sampler
        # import-time sockets" for the full rationale.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._channel_initialized: bool = False
        self._channel_init_lock = threading.Lock()

        # Streaming state (lazily initialized).
        self._bidi_call: Any | None = None

        # TCP pre-probe state. Tracks the monotonic timestamp of the most
        # recent failed pre-probe so we can short-circuit subsequent calls
        # within the backoff window without re-touching the socket.
        self._last_preprobe_fail_monotonic: float = 0.0
        self._preprobe_enabled: bool = os.environ.get(_PREPROBE_ENABLED_ENV_VAR, "1") != "0"

    def _run_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        assert self._loop is not None  # set in _ensure_channel
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_channel(self) -> None:
        """Start the background loop + open the gRPC channel on first call.

        Idempotent. Thread-safe via a lock so concurrent first calls
        from multiple sampling threads don't double-spawn the loop.
        """
        if self._channel_initialized:
            return
        with self._channel_init_lock:
            if self._channel_initialized:
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
            self._channel_initialized = True

    def warmup(self) -> None:
        """Eagerly open the gRPC channel + verify the server is reachable.

        Called by the engine adapter once at startup (after the
        cloudflared sidecar has had a chance to come up) so that
        per-token ``get_random_bytes()`` calls land on an already-open,
        already-verified channel — no first-fetch connect cost.

        Soft-fail: if the channel can't open or the verification fetch
        fails, log a warning and return cleanly. ``FallbackEntropySource``
        will engage on subsequent fetches and the circuit breaker /
        TCP pre-probe will keep the system serving via system PRNG
        until the QRNG backend recovers.

        Idempotent and snapshot-safe: if a previous warmup left a
        captured-stale channel (e.g. across a Modal /sleep + /wake_up),
        the verification fetch surfaces the stale state and we reset
        the channel before declaring warmup complete.
        """
        try:
            self._ensure_channel()
        except Exception as exc:
            logger.warning(
                "QuantumGrpcSource.warmup: channel init failed (%s); "
                "falling back to lazy init on first fetch",
                exc,
                extra={"event": "qrng.warmup.channel_init_failed"},
            )
            return

        # Tiny verification fetch on the freshly-opened channel. Small
        # n=8 keeps the bandwidth cost negligible and the round-trip
        # short. We use a snug timeout (2x the adaptive minimum) so a
        # stale channel from a prior snapshot surfaces fast rather than
        # blocking startup.
        try:
            self.get_random_bytes(8)
            logger.info(
                "QuantumGrpcSource.warmup: channel ready (%s)",
                self._address,
                extra={"event": "qrng.warmup.ok"},
            )
        except EntropyUnavailableError as exc:
            # First fetch failed — could be a stale post-snapshot
            # channel, or genuinely unreachable backend. Reset the
            # channel state so the next fetch re-initializes cleanly,
            # then return. Fallback handles the remaining requests
            # until QRNG recovers.
            logger.warning(
                "QuantumGrpcSource.warmup: verification fetch failed (%s); "
                "resetting channel for clean lazy re-init",
                exc,
                extra={"event": "qrng.warmup.verify_failed"},
            )
            self._reset_channel()
        except Exception as exc:
            logger.warning(
                "QuantumGrpcSource.warmup: unexpected verification error (%s)",
                exc,
                extra={"event": "qrng.warmup.verify_unexpected"},
            )
            self._reset_channel()

    def _reset_channel(self) -> None:
        """Tear down the channel + loop so the next call re-inits cleanly.

        Used by ``warmup()`` when verification surfaces a stale
        post-snapshot channel. Best-effort: errors during teardown are
        swallowed since the goal is just to flip ``_channel_initialized``
        back to False.
        """
        with self._channel_init_lock:
            if not self._channel_initialized:
                return
            loop = self._loop
            channel = getattr(self, "_channel", None)
            if loop is not None and channel is not None:
                try:
                    fut = asyncio.run_coroutine_threadsafe(channel.close(), loop)
                    fut.result(timeout=2.0)
                except Exception:
                    pass
            if loop is not None:
                with contextlib.suppress(Exception):
                    loop.call_soon_threadsafe(loop.stop)
            self._channel_initialized = False
            self._loop = None
            self._thread = None

    async def _init_channel(self) -> None:
        """Create the gRPC async channel and generic method handles."""
        import grpc.aio

        # gRPC keepalive: tuned to respect the server's default
        # ``min_ping_interval_without_data_ms = 300_000`` (5 min) policy.
        # Earlier config (30 s + permit_without_calls=True +
        # max_pings_without_data=0) pinged so aggressively that the QRNG
        # server / cloudflared front replied ``GOAWAY ENHANCE_YOUR_CALM
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

        self._channel = grpc.aio.insecure_channel(self._address, options=options)

        # Generic unary method handle — works with any proto that uses
        # field 1 varint (request) and field 1 bytes (response).
        self._unary_method = self._channel.unary_unary(
            self._method_path,
            request_serializer=_generic_request_serializer,
            response_deserializer=_generic_response_deserializer,
        )

        # Generic streaming method handle — only created when path is non-empty.
        self._stream_method: Any | None = None
        if self._stream_method_path:
            self._stream_method = self._channel.stream_stream(
                self._stream_method_path,
                request_serializer=_generic_request_serializer,
                response_deserializer=_generic_response_deserializer,
            )

    @property
    def name(self) -> str:
        """Return ``'quantum_grpc'``."""
        return "quantum_grpc"

    @property
    def is_available(self) -> bool:
        """Whether the source can currently provide entropy.

        Returns ``False`` if the circuit breaker is open (too many failures).
        """
        if self._closed:
            return False
        return not (self._circuit_open and time.monotonic() < self._circuit_open_until)

    def get_random_bytes(self, n: int) -> bytes:
        """Fetch *n* random bytes from the gRPC entropy server.

        Synchronous wrapper around the async transport. Uses the background
        event loop thread to dispatch async calls.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes from the entropy server.

        Raises:
            EntropyUnavailableError: If the server is unreachable or the
                circuit breaker is open.
        """
        if self._closed:
            raise EntropyUnavailableError("QuantumGrpcSource is closed")

        # Circuit breaker check.
        if self._circuit_open:
            if time.monotonic() >= self._circuit_open_until:
                # Half-open: try one request.
                self._circuit_open = False
                logger.info("Circuit breaker half-open, attempting reconnection")
                # iter-53 (2026-06-09): reset the channel BEFORE the
                # half-open attempt. The dominant open-circuit cause in
                # the Modal deploy is a stale post-/sleep channel (the
                # cloudflared sidecar restarts on _wake underneath the
                # snapshot-frozen HTTP/2 state); testing recovery on the
                # suspect channel wastes the whole half-open cycle —
                # observed as a 2-cycle (~20 s, ~22 PRNG tokens) recovery
                # crawl on the first post-wake generation. A reset costs
                # one loopback reconnect (~ms) per half-open, negligible
                # against the recovery-window cadence.
                try:
                    self._reset_channel()
                except Exception as reset_exc:
                    logger.warning(
                        "half-open channel reset failed: %s", reset_exc
                    )
            else:
                raise EntropyUnavailableError(
                    "Circuit breaker open: too many consecutive gRPC failures"
                )

        last_error: Exception | None = None
        for attempt in range(1 + self._retry_count):
            try:
                t0 = time.perf_counter()
                data = self._fetch_sync(n)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self._update_latency(elapsed_ms)
                self._consecutive_failures = 0
                return data
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "gRPC entropy fetch attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self._retry_count,
                    exc,
                )

        # All retries exhausted.
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._cb_max_consecutive_failures:
            self._circuit_open = True
            self._circuit_open_until = time.monotonic() + self._cb_recovery_window_s
            logger.warning(
                "Circuit breaker opened after %d consecutive failures",
                self._consecutive_failures,
            )

        # iter-52d (2026-05-25): when every retry on an established
        # channel fails, the channel itself is likely stale — most
        # commonly because of a Modal /sleep + /wake_up snapshot cycle
        # that froze the gRPC channel state while the cloudflared
        # sidecar was restarted underneath us. Reset the channel so the
        # next call re-initialises cleanly against the freshly-spawned
        # sidecar. Best-effort: errors during teardown are swallowed.
        try:
            self._reset_channel()
        except Exception as reset_exc:
            logger.warning(
                "channel reset after retry-exhaust failed: %s", reset_exc
            )

        raise EntropyUnavailableError(
            f"gRPC entropy fetch failed after {1 + self._retry_count} attempts: {last_error}"
        ) from last_error

    def _fetch_sync(self, n: int) -> bytes:
        """Dispatch an async fetch to the background loop and block.

        Fronts the dispatch with a bounded-time TCP-connect pre-probe so
        an unreachable sidecar fails the request in ~500 ms rather than
        consuming the full multi-second gRPC retry budget. See the module-
        level pre-probe constants for tunables; ``QR_GRPC_PREPROBE_ENABLED=0``
        disables the pre-probe entirely if a downstream consumer needs the
        legacy long-retry behaviour.
        """
        self._tcp_preprobe()

        # Lazy channel + background-loop bring-up (Phase 2). Runs only
        # on the first call; subsequent calls short-circuit on the
        # _channel_initialized flag.
        self._ensure_channel()
        assert self._loop is not None  # _ensure_channel just set it

        timeout_s = self._get_timeout() / 1000.0
        coro = self._fetch_async(n)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        t0 = time.perf_counter()
        try:
            return future.result(timeout=timeout_s)
        except TimeoutError as exc:
            self._record_timeout_sample(time.perf_counter() - t0, timeout_s)
            raise EntropyUnavailableError(
                f"gRPC entropy fetch timed out after {timeout_s * 1000:.0f}ms"
            ) from exc
        except Exception as exc:
            self._record_timeout_sample(time.perf_counter() - t0, timeout_s)
            raise EntropyUnavailableError(f"gRPC entropy fetch failed: {exc}") from exc

    def _record_timeout_sample(self, elapsed_s: float, timeout_s: float) -> None:
        """Feed timeout-shaped failures into the adaptive-latency window.

        iter-53 (2026-06-09): only successful fetches used to call
        ``_update_latency``, which made the adaptive timeout a one-way
        ratchet — once P99 converged on a fast window (observed ~96 ms
        through the cloudflared tunnel → ~145 ms ceiling), any genuine
        upward drift in backend latency could NEVER be re-learned: every
        slower fetch was cut off at the old ceiling and discarded, so the
        window stayed frozen and the source flapped between timeout and
        fallback indefinitely.

        Recording the cut-off duration as a latency sample lets P99 climb
        toward the observed (censored) latency, raising the next adaptive
        timeout by up to ``cb_timeout_multiplier`` per re-learn round,
        hard-capped by ``grpc_timeout_ms``. Fast failures (connection
        refused, channel teardown) are excluded — they say nothing about
        latency and would wrongly DEFLATE the window.
        """
        elapsed_ms = elapsed_s * 1000.0
        if elapsed_ms >= 0.8 * timeout_s * 1000.0:
            self._update_latency(elapsed_ms)

    def _tcp_preprobe(self) -> None:
        """One-shot TCP-connect probe of the gRPC endpoint.

        Skips entirely when ``QR_GRPC_PREPROBE_ENABLED=0`` is set so a
        downstream consumer that knows the gRPC server is slow-to-listen
        (e.g. starting alongside the client) can opt back into the legacy
        retry-driven behaviour.

        Within ``_PREPROBE_BACKOFF_S`` of a previous failure, short-circuits
        to ``EntropyUnavailableError`` without re-touching the socket. This
        bounds the SYN rate to ~12/minute even when vLLM is sampling at 50
        tokens/sec, which matters because the kernel may otherwise rate-
        limit unrelated connections to the same port.
        """
        if not self._preprobe_enabled:
            return

        now = time.monotonic()
        if (now - self._last_preprobe_fail_monotonic) < _PREPROBE_BACKOFF_S:
            raise EntropyUnavailableError(
                "QRNG host unreachable (TCP pre-probe failed within last "
                f"{_PREPROBE_BACKOFF_S:.0f}s; backoff in effect)"
            )

        host, _, port_s = self._address.partition(":")
        try:
            port = int(port_s)
        except ValueError as exc:
            raise EntropyUnavailableError(
                f"Malformed gRPC server address {self._address!r}"
            ) from exc

        try:
            with socket.create_connection((host, port), timeout=_PREPROBE_TIMEOUT_S):
                pass
        except OSError as exc:
            self._last_preprobe_fail_monotonic = now
            logger.warning(
                "QRNG TCP pre-probe failed: %s:%s -- %s: %s",
                host,
                port,
                type(exc).__name__,
                exc,
                extra={
                    "event": "qrng.tcp_preprobe.failed",
                    "host": host,
                    "port": port,
                    "error_type": type(exc).__name__,
                },
            )
            raise EntropyUnavailableError(
                f"QRNG host {host}:{port} unreachable: {type(exc).__name__}: {exc}"
            ) from exc

    async def _fetch_async(self, n: int) -> bytes:
        """Route to the appropriate transport mode."""
        if self._mode == "unary":
            return await self._fetch_unary(n)
        elif self._mode == "server_streaming":
            return await self._fetch_server_streaming(n)
        elif self._mode == "bidi_streaming":
            return await self._fetch_bidi_streaming(n)
        else:
            raise EntropyUnavailableError(f"Unknown gRPC mode: {self._mode!r}")

    async def _fetch_unary(self, n: int) -> bytes:
        """Single request-response per call. Simplest. Higher overhead."""
        request_bytes = _encode_varint_request(n)
        timeout_s = self._get_timeout() / 1000.0
        raw_response: bytes = await self._unary_method(
            request_bytes,
            timeout=timeout_s,
            metadata=self._metadata or None,
        )
        return _decode_bytes_field1(raw_response)

    async def _fetch_server_streaming(self, n: int) -> bytes:
        """Use the streaming RPC in a request/response style.

        Sends one request and reads one response from the stream.
        The stream is re-established on each call for server-streaming semantics.
        """
        request_bytes = _encode_varint_request(n)

        async def request_iterator() -> Any:
            yield request_bytes

        if self._stream_method is None:  # pragma: no cover — validated in __init__
            raise EntropyUnavailableError("Stream method not initialized")
        call = self._stream_method(request_iterator(), metadata=self._metadata or None)
        raw_response: bytes | None = await call.read()
        if raw_response is None:
            raise EntropyUnavailableError("Server stream ended unexpectedly")
        call.cancel()
        return _decode_bytes_field1(raw_response)

    async def _fetch_bidi_streaming(self, n: int) -> bytes:
        """Use a persistent bidirectional stream for lowest latency.

        The stream is lazily initialized on first call and reused thereafter.
        If the stream breaks, it is re-established on the next call.
        """
        request_bytes = _encode_varint_request(n)

        try:
            if self._bidi_call is None:
                if self._stream_method is None:  # pragma: no cover — validated in __init__
                    raise EntropyUnavailableError("Stream method not initialized")
                self._bidi_call = self._stream_method(
                    metadata=self._metadata or None,
                )

            await self._bidi_call.write(request_bytes)
            raw_response: bytes | None = await self._bidi_call.read()
            if raw_response is None:
                # Stream ended — reset and retry.
                self._bidi_call = None
                raise EntropyUnavailableError("Bidi stream ended unexpectedly")
            return _decode_bytes_field1(raw_response)
        except EntropyUnavailableError:
            raise
        except Exception:
            # Stream broken — reset for next call.
            self._bidi_call = None
            raise

    # --- Circuit breaker ---

    def _update_latency(self, elapsed_ms: float) -> None:
        """Add a latency sample to the rolling window and recompute P99.

        Args:
            elapsed_ms: Time taken for the last fetch in milliseconds.
        """
        self._latency_window.append(elapsed_ms)
        if len(self._latency_window) >= 10:
            sorted_latencies = sorted(self._latency_window)
            idx = int(len(sorted_latencies) * 0.99)
            idx = min(idx, len(sorted_latencies) - 1)
            self._p99_ms = sorted_latencies[idx]

    def _get_timeout(self) -> float:
        """Compute the adaptive timeout in milliseconds.

        Returns:
            ``max(5ms, P99 * 1.5)`` or the configured timeout, whichever
            is smaller.
        """
        adaptive = max(self._cb_min_timeout_ms, self._p99_ms * self._cb_timeout_multiplier)
        return min(adaptive, self._timeout_ms)

    # --- Lifecycle ---

    def close(self) -> None:
        """Release the gRPC channel, event loop, and background thread.

        No-op when the channel was never opened (the source was
        constructed but never used). Phase 2's lazy init means
        ``__init__`` no longer creates the loop or the channel; both
        are optional state at shutdown time.
        """
        if self._closed:
            return
        self._closed = True

        if not self._channel_initialized:
            return

        loop = self._loop
        thread = self._thread
        assert loop is not None and thread is not None  # _ensure_channel set both

        async def _shutdown() -> None:
            if self._bidi_call is not None:
                self._bidi_call.cancel()
                self._bidi_call = None
            await self._channel.close()

        try:
            future = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            future.result(timeout=5.0)
        except Exception:
            logger.warning("Error during QuantumGrpcSource cleanup", exc_info=True)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5.0)

    def health_check(self) -> dict[str, Any]:
        """Return detailed health status including circuit breaker state.

        The API key is never included in the output. Only a boolean
        ``authenticated`` flag indicates whether auth is configured.

        Returns:
            Dictionary with source name, availability, circuit breaker state,
            P99 latency, and connection details.
        """
        return {
            "source": self.name,
            "healthy": self.is_available,
            "address": self._address,
            "mode": self._mode,
            "method_path": self._method_path,
            "authenticated": bool(self._api_key),
            "circuit_open": self._circuit_open,
            "p99_ms": round(self._p99_ms, 2),
            "consecutive_failures": self._consecutive_failures,
            "latency_samples": len(self._latency_window),
        }
