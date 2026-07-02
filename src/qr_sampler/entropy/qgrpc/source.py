"""``QuantumGrpcSource`` — the protocol-agnostic gRPC entropy source facade.

This is the primary production entropy source. It fetches random bytes
from a remote entropy server over gRPC, composing the package's four
concerns:

- :mod:`~qr_sampler.entropy.qgrpc.channel` — background loop + channel
  lifecycle (lazy bring-up, reset, close).
- :mod:`~qr_sampler.entropy.qgrpc.transport` — wire codec + unary /
  server-streaming / bidi dispatch.
- :mod:`~qr_sampler.entropy.qgrpc.breaker` — adaptive-P99 circuit breaker.
- :mod:`~qr_sampler.entropy.qgrpc.preprobe` — bounded TCP-connect
  fast-fail probe.

All modes satisfy the post-selection just-in-time constraint: the gRPC
request for token *N* is sent only after token *N-1* has been selected.
Two request timings implement that contract:

- **Serial** (``get_random_bytes()``): request fired after the next
  token's logits are computed — the engine blocks for the full round trip.
- **Pipelined** (``prefetch()`` + ``get_random_bytes_with_ticket()``):
  request fired the instant the previous token is selected, so the round
  trip overlaps the engine's forward pass. The request carries a 63-bit
  commitment nonce (derived from the just-selected token) in the proto's
  ``sequence_id`` field; servers that echo ``sequence_id`` make the
  post-selection ordering externally verifiable with zero server changes.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.qgrpc.breaker import AdaptiveCircuitBreaker
from qr_sampler.entropy.qgrpc.channel import GrpcChannel
from qr_sampler.entropy.qgrpc.preprobe import TcpPreprobe
from qr_sampler.entropy.qgrpc.transport import GrpcTransport, _FetchReply
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

logger = logging.getLogger("qr_sampler")

# Minimum seconds between consecutive ``qrng.quota_exhausted`` log events.
# A quota storm fails once per token; one structured event per minute is
# enough for the operator while keeping the log stream readable.
_QUOTA_LOG_THROTTLE_S = 60.0


def _is_quota_exhausted(exc: BaseException | None) -> bool:
    """True when *exc* (or its cause chain) is a gRPC RESOURCE_EXHAUSTED.

    String-compares the status-code name instead of importing ``grpc`` at
    module level (the module must import cleanly without grpcio for the
    registry's lazy-availability check). Walks ``__cause__`` so wrapped
    AioRpcErrors classify correctly.
    """
    seen = 0
    while exc is not None and seen < 8:  # bounded: defensive vs cause cycles
        code = getattr(exc, "code", None)
        if callable(code):
            try:
                if getattr(code(), "name", "") == "RESOURCE_EXHAUSTED":
                    return True
            except Exception:  # classification is best-effort
                pass
        exc = exc.__cause__
        seen += 1
    return False


class PrefetchTicket:
    """Handle for an in-flight pipelined entropy fetch.

    Returned by ``QuantumGrpcSource.prefetch()`` and redeemed by
    ``get_random_bytes_with_ticket()``. After redemption the diagnostic
    attributes (``hit``, ``wait_ms``, ``echo_verified``,
    ``server_timestamp_ns``) are populated so the sampling pipeline can
    record per-token verification + overlap telemetry.
    """

    __slots__ = (
        "echo_verified",
        "future",
        "hit",
        "n",
        "nonce",
        "server_timestamp_ns",
        "t_fire_monotonic",
        "wait_ms",
    )

    def __init__(self, future: Any, nonce: int, n: int) -> None:
        self.future = future
        self.nonce = nonce
        self.n = n
        self.t_fire_monotonic = time.monotonic()
        # Populated at redemption time.
        self.hit: bool | None = None
        self.wait_ms: float | None = None
        self.echo_verified: bool | None = None
        self.server_timestamp_ns: int | None = None

    def cancel(self) -> None:
        """Best-effort cancellation of the in-flight fetch."""
        with contextlib.suppress(Exception):
            self.future.cancel()


class QuantumGrpcSource(EntropySource):
    """Protocol-agnostic gRPC entropy source with configurable transport mode.

    Connects to any gRPC entropy server using configurable method paths and
    the standard protobuf wire format. All modes satisfy the just-in-time
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
        self._api_key = config.grpc_api_key
        self._closed = False

        # Validate streaming config upfront.
        if self._mode in ("server_streaming", "bidi_streaming") and not (
            config.grpc_stream_method_path
        ):
            raise ConfigValidationError(
                f"grpc_mode={self._mode!r} requires a non-empty grpc_stream_method_path"
            )

        # Build call metadata (empty tuple if no auth).
        metadata: tuple[tuple[str, str], ...] = ()
        if self._api_key:
            metadata = ((config.grpc_api_key_header, self._api_key),)

        self._breaker = AdaptiveCircuitBreaker(
            window_size=config.cb_window_size,
            min_timeout_ms=config.cb_min_timeout_ms,
            timeout_multiplier=config.cb_timeout_multiplier,
            max_timeout_ms=config.grpc_timeout_ms,
            recovery_window_s=config.cb_recovery_window_s,
            recovery_window_max_s=config.cb_recovery_window_max_s,
            max_consecutive_failures=config.cb_max_consecutive_failures,
        )
        self._preprobe = TcpPreprobe(self._address)
        self._channel = GrpcChannel(
            self._address,
            timeout_ms=config.grpc_timeout_ms,
            method_path=config.grpc_method_path,
            stream_method_path=config.grpc_stream_method_path,
        )
        self._transport = GrpcTransport(
            self._channel,
            mode=self._mode,
            metadata=metadata,
            timeout_ms_provider=self._breaker.timeout_ms,
        )

        # Quota telemetry state (see _is_quota_exhausted / _QUOTA_LOG_THROTTLE_S).
        # -inf, NOT 0.0: time.monotonic() has no defined epoch (it is seconds
        # since boot on Linux/Windows), so a 0.0 sentinel silently throttles
        # the FIRST quota log whenever the host booted less than
        # _QUOTA_LOG_THROTTLE_S ago (fresh CI runners, fresh VMs).
        self._last_quota_log_monotonic: float = float("-inf")
        self._quota_max_bytes_per_request = config.qrng_max_bytes_per_request
        self._quota_max_requests_per_minute = config.qrng_max_requests_per_minute
        self._quota_max_bytes_per_day = config.qrng_max_bytes_per_day
        if config.sample_count > self._quota_max_bytes_per_request:
            logger.warning(
                "sample_count=%d exceeds the QRNG service's documented "
                "per-request limit of %d bytes — every per-token fetch will "
                "return RESOURCE_EXHAUSTED and sampling will run entirely on "
                "the fallback source. Lower QR_SAMPLE_COUNT or request a "
                "larger per-request quota.",
                config.sample_count,
                self._quota_max_bytes_per_request,
                extra={
                    "event": "qrng.sample_count_exceeds_request_cap",
                    "sample_count": config.sample_count,
                    "max_bytes_per_request": self._quota_max_bytes_per_request,
                },
            )

        # Pipelined-fetch telemetry (exposed via health_check()).
        self._prefetch_fired: int = 0
        self._prefetch_hits: int = 0
        self._prefetch_misses: int = 0

    def warmup(self) -> None:
        """Eagerly open the gRPC channel + verify the server is reachable.

        Called by the engine adapter once at startup so that per-token
        ``get_random_bytes()`` calls land on an already-open,
        already-verified channel — no first-fetch connect cost.

        Soft-fail: if the channel can't open or the verification fetch
        fails, log a warning and return cleanly. ``FallbackEntropySource``
        will engage on subsequent fetches and the circuit breaker /
        TCP pre-probe will keep the system serving via system PRNG
        until the QRNG backend recovers.

        Idempotent: if a previous warmup left a stale channel, the
        verification fetch surfaces the stale state and we reset the
        channel before declaring warmup complete.
        """
        try:
            self._channel.ensure()
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
        # stale channel surfaces fast rather than blocking startup.
        try:
            self.get_random_bytes(8)
            logger.info(
                "QuantumGrpcSource.warmup: channel ready (%s)",
                self._address,
                extra={"event": "qrng.warmup.ok"},
            )
        except EntropyUnavailableError as exc:
            # First fetch failed — could be a stale channel, or a
            # genuinely unreachable backend. Reset the channel state so
            # the next fetch re-initializes cleanly, then return.
            # Fallback handles the remaining requests until QRNG recovers.
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

        Best-effort: errors during teardown are swallowed. Any bidi
        session is bound to the torn-down loop + call, so the transport
        forgets it too.
        """
        self._channel.reset()
        self._transport.clear_bidi()

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
        return not self._breaker.is_blocking

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
        if self._breaker.circuit_open:
            if self._breaker.try_half_open():
                logger.info("Circuit breaker half-open, attempting reconnection")
                # iter-53 (2026-06-09): reset the channel BEFORE the
                # half-open attempt. The dominant open-circuit cause is a
                # stale channel (the tunnel/backend restarted underneath
                # the frozen HTTP/2 state); testing recovery on the
                # suspect channel wastes the whole half-open cycle —
                # observed as a 2-cycle (~20 s, ~22 PRNG tokens) recovery
                # crawl. A reset costs one loopback reconnect (~ms) per
                # half-open, negligible against the recovery-window cadence.
                try:
                    self._reset_channel()
                except Exception as reset_exc:
                    logger.warning("half-open channel reset failed: %s", reset_exc)
            else:
                raise EntropyUnavailableError(
                    "Circuit breaker open: too many consecutive gRPC failures"
                )

        last_error: Exception | None = None
        for attempt in range(1 + self._retry_count):
            try:
                reply = self._fetch_sync(n)
                self._breaker.note_success(reply.elapsed_ms)
                self._preprobe.note_fetch_success()
                return reply.payload
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "gRPC entropy fetch attempt %d/%d failed: %s",
                    attempt + 1,
                    1 + self._retry_count,
                    exc,
                )

        # All retries exhausted.
        window = self._breaker.note_failure()
        if window is not None:
            logger.warning(
                "Circuit breaker opened after %d consecutive failures; "
                "next half-open in %.0fs (open #%d, backing off to max %.0fs)",
                self._breaker.consecutive_failures,
                window,
                self._breaker.circuit_open_count,
                self._breaker.recovery_window_max_s,
            )

        # iter-52d (2026-05-25): when every retry on an established
        # channel fails, the channel itself is likely stale — most
        # commonly because the tunnel/backend restarted underneath us.
        # Reset the channel so the next call re-initialises cleanly.
        # Best-effort: errors during teardown are swallowed.
        try:
            self._reset_channel()
        except Exception as reset_exc:
            logger.warning("channel reset after retry-exhaust failed: %s", reset_exc)

        raise EntropyUnavailableError(
            f"gRPC entropy fetch failed after {1 + self._retry_count} attempts: {last_error}"
        ) from last_error

    def prefetch(self, n: int, nonce: int | None = None) -> PrefetchTicket | None:
        """Fire an asynchronous entropy fetch; return a redeemable ticket.

        Called by the sampling pipeline the instant the previous token is
        selected, so the gRPC round trip overlaps the engine's next
        forward pass. The *nonce* (a 63-bit commitment derived from the
        just-selected token) rides in the request's ``sequence_id`` field;
        the server's echo binds the response to a request that could only
        exist after that selection — the post-selection ordering is
        thereby verifiable from the response stream alone.

        Never raises and never blocks on the network. Returns ``None``
        when the source is closed, the circuit breaker is open, the
        pre-probe backoff is active, or anything in the dispatch fails —
        the caller then degrades to the synchronous fetch path.

        Deliberately skips the half-open window too (checks the raw
        ``circuit_open`` flag, not the elapsed window): the dominant
        open-circuit cause is a stale channel, and only the serial path
        performs the half-open channel reset — a prefetch fired here would
        ride the suspect channel and fail anyway. The cost is one serial
        fetch per recovery cycle, after which prefetch re-engages on the
        next token.
        """
        if self._closed or self._breaker.circuit_open:
            return None
        if self._preprobe.backoff_active():
            return None
        try:
            self._channel.ensure()
            future = self._channel.submit(self._transport.fetch(n, nonce or 0))
        except Exception as exc:
            logger.debug("prefetch dispatch failed: %s", exc)
            return None
        self._prefetch_fired += 1
        return PrefetchTicket(future=future, nonce=nonce or 0, n=n)

    def get_random_bytes_with_ticket(self, n: int, ticket: Any | None) -> bytes:
        """Redeem a prefetched fetch, falling back to the serial path.

        On the happy path the response is already in flight (or already
        arrived) and the engine blocks only for the residual wait — the
        portion of the round trip the forward pass didn't cover. On any
        ticket failure (timeout, stream break, server error) this degrades
        to ``get_random_bytes()``, which carries the full retry, pre-probe
        and circuit-breaker machinery — so the pipelined path is never
        less robust than the serial one.
        """
        if ticket is None:
            return self.get_random_bytes(n)
        if self._closed:
            ticket.cancel()
            raise EntropyUnavailableError("QuantumGrpcSource is closed")

        timeout_s = self._breaker.timeout_ms() / 1000.0
        t0 = time.perf_counter()
        already_done = ticket.future.done()
        try:
            reply: _FetchReply = ticket.future.result(timeout=timeout_s)
        except Exception as exc:
            ticket.cancel()
            ticket.hit = False
            self._prefetch_misses += 1
            logger.debug("prefetch redeem failed (%s); falling back to serial fetch", exc)
            return self.get_random_bytes(n)

        # Success bookkeeping mirrors the serial path.
        ticket.hit = True
        ticket.wait_ms = 0.0 if already_done else (time.perf_counter() - t0) * 1000.0
        ticket.echo_verified = bool(ticket.nonce) and reply.sequence_id == ticket.nonce
        ticket.server_timestamp_ns = reply.generation_timestamp_ns or None
        self._prefetch_hits += 1
        self._breaker.note_success(reply.elapsed_ms)
        self._preprobe.note_fetch_success()
        return reply.payload

    def _fetch_sync(self, n: int) -> _FetchReply:
        """Dispatch an async fetch to the background loop and block.

        Fronts the dispatch with a bounded-time TCP-connect pre-probe so
        an unreachable server fails the request in ~500 ms rather than
        consuming the full multi-second gRPC retry budget (see
        :mod:`~qr_sampler.entropy.qgrpc.preprobe` for tunables;
        ``QR_GRPC_PREPROBE_ENABLED=0`` disables it entirely).
        """
        self._preprobe.check()

        # Lazy channel + background-loop bring-up. Runs only on the first
        # call; subsequent calls short-circuit on the initialized flag.
        self._channel.ensure()

        timeout_s = self._breaker.timeout_ms() / 1000.0
        future = self._channel.submit(self._transport.fetch(n))
        t0 = time.perf_counter()
        try:
            result: _FetchReply = future.result(timeout=timeout_s)
            return result
        except TimeoutError as exc:
            self._breaker.record_censored_latency(time.perf_counter() - t0, timeout_s)
            raise EntropyUnavailableError(
                f"gRPC entropy fetch timed out after {timeout_s * 1000:.0f}ms"
            ) from exc
        except Exception as exc:
            self._breaker.record_censored_latency(time.perf_counter() - t0, timeout_s)
            if _is_quota_exhausted(exc):
                now = time.monotonic()
                if (now - self._last_quota_log_monotonic) >= _QUOTA_LOG_THROTTLE_S:
                    self._last_quota_log_monotonic = now
                    logger.error(
                        "QRNG QUOTA EXHAUSTED: the server returned "
                        "RESOURCE_EXHAUSTED for n=%d. This is a rate/byte "
                        "limit verdict, NOT a connectivity failure — the "
                        "tunnel and channel are fine. Documented limits: "
                        "%d bytes/request, %d requests/minute, %d bytes/day. "
                        "Operator action: lower sample_count or concurrent "
                        "sequences, or request a larger quota from the QRNG "
                        "team.",
                        n,
                        self._quota_max_bytes_per_request,
                        self._quota_max_requests_per_minute,
                        self._quota_max_bytes_per_day,
                        extra={
                            "event": "qrng.quota_exhausted",
                            "bytes_requested": n,
                            "max_bytes_per_request": self._quota_max_bytes_per_request,
                            "max_requests_per_minute": self._quota_max_requests_per_minute,
                        },
                    )
                raise EntropyUnavailableError(
                    f"QRNG quota exhausted (RESOURCE_EXHAUSTED) for n={n}; "
                    "rate/byte limit hit, connectivity is fine"
                ) from exc
            raise EntropyUnavailableError(f"gRPC entropy fetch failed: {exc}") from exc

    # --- Lifecycle ---

    def close(self) -> None:
        """Release the gRPC channel, event loop, and background thread.

        No-op when the channel was never opened (the source was
        constructed but never used).
        """
        if self._closed:
            return
        self._closed = True
        self._channel.close(pre_close=self._transport.close_bidi)

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
            "circuit_open": self._breaker.circuit_open,
            "p99_ms": round(self._breaker.p99_ms, 2),
            "consecutive_failures": self._breaker.consecutive_failures,
            "latency_samples": self._breaker.latency_sample_count,
            "prefetch_fired": self._prefetch_fired,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
        }
