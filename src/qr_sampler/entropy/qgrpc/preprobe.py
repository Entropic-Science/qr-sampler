"""TCP-connect pre-probe state machine for the gRPC entropy source.

The pre-probe fronts every serial fetch with a bounded-time ``connect()``
to the gRPC server address; if the kernel does not return a listening
socket within ``_PREPROBE_TIMEOUT_S``, ``EntropyUnavailableError`` is
raised immediately and the probe backs off for ``_PREPROBE_BACKOFF_S``
before touching the socket again. This converts a ~15 s (3 retries x ~5 s
timeout) "QRNG unreachable" event into a ~500 ms one, letting the
``FallbackEntropySource`` take over before the user-facing request times
out.

Operators can opt out by setting ``QR_GRPC_PREPROBE_ENABLED=0`` in the
environment (default is enabled — failing fast is strictly better than
the legacy retry-driven behaviour for the in-tree deployment).
"""

from __future__ import annotations

import logging
import os
import socket
import time

from qr_sampler.exceptions import EntropyUnavailableError

logger = logging.getLogger("qr_sampler")

_PREPROBE_TIMEOUT_S = 0.5
_PREPROBE_BACKOFF_S = 5.0
PREPROBE_ENABLED_ENV_VAR = "QR_GRPC_PREPROBE_ENABLED"

# Healthy-path pre-probe suppression. A successful gRPC fetch within this
# window is strictly stronger evidence of reachability than a fresh TCP
# connect, so the pre-probe is skipped entirely while fetches keep
# succeeding. This removes one connect()/close() syscall pair + loopback
# handshake from EVERY steady-state token (the pre-probe previously ran
# unconditionally per fetch). The probe re-engages automatically once no
# fetch has succeeded for the window — i.e. exactly when its fast-fail
# behaviour is actually useful.
_PREPROBE_HEALTHY_WINDOW_S = 30.0


class TcpPreprobe:
    """Bounded TCP-connect probe with failure backoff + healthy suppression.

    Args:
        address: The gRPC server address (``host:port``).
    """

    def __init__(self, address: str) -> None:
        self._address = address
        self.enabled: bool = os.environ.get(PREPROBE_ENABLED_ENV_VAR, "1") != "0"
        # Monotonic timestamp of the most recent failed pre-probe, used to
        # short-circuit subsequent calls within the backoff window without
        # re-touching the socket.
        self._last_fail_monotonic: float = 0.0
        # Timestamp of the most recent successful fetch (serial or
        # pipelined) — suppresses the per-token probe while the channel is
        # demonstrably healthy.
        self._last_fetch_success_monotonic: float = 0.0

    def note_fetch_success(self) -> None:
        """Record a successful fetch (engages healthy-path suppression)."""
        self._last_fetch_success_monotonic = time.monotonic()

    def backoff_active(self) -> bool:
        """True while a recent probe failure's backoff window is in effect.

        Used by the prefetch path to skip speculative dispatch entirely.
        Always False when the probe is disabled.
        """
        return self.enabled and (time.monotonic() - self._last_fail_monotonic) < _PREPROBE_BACKOFF_S

    def check(self) -> None:
        """One-shot TCP-connect probe of the gRPC endpoint.

        Skips entirely when ``QR_GRPC_PREPROBE_ENABLED=0`` was set at
        source construction so a downstream consumer that knows the gRPC
        server is slow-to-listen (e.g. starting alongside the client) can
        opt back into the legacy retry-driven behaviour.

        Within the backoff window of a previous failure, short-circuits to
        ``EntropyUnavailableError`` without re-touching the socket. This
        bounds the SYN rate to ~12/minute even when the engine is sampling
        at 50 tokens/sec, which matters because the kernel may otherwise
        rate-limit unrelated connections to the same port.

        Raises:
            EntropyUnavailableError: When the endpoint is unreachable, the
                backoff window is active, or the address is malformed.
        """
        if not self.enabled:
            return

        now = time.monotonic()

        # Healthy-path suppression: a fetch succeeded recently, so the
        # endpoint is reachable by construction — skip the probe and its
        # per-token connect()/close() syscall pair entirely.
        if (
            self._last_fetch_success_monotonic
            and (now - self._last_fetch_success_monotonic) < _PREPROBE_HEALTHY_WINDOW_S
        ):
            return

        if (now - self._last_fail_monotonic) < _PREPROBE_BACKOFF_S:
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
            self._last_fail_monotonic = now
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
