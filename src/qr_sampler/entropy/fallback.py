"""Fallback entropy source — composition wrapper with transparent failover.

``FallbackEntropySource`` wraps a *primary* and a *fallback* source. When the
primary raises :class:`~qr_sampler.exceptions.EntropyUnavailableError`, the
wrapper transparently delegates to the fallback. **All other exceptions
propagate unchanged** — this is deliberate: only entropy-unavailability is a
recoverable condition.

Operator-visible logging
------------------------
A fallback emits a structured ``entropy.degraded`` warning plus a louder
human-readable ``entropy.degraded.alert``, BOTH rate-limited together to the
first fallback of a degraded window and at most once per minute thereafter.
This throttle is load-bearing: a sustained primary outage produces one entropy
fetch *per token*, so the previous unconditional warn-per-fallback emitted
hundreds of identical lines per request and drowned ``modal app logs``. The
running fallback count rides on every emitted record (and the status file +
``/health/entropy`` carry the exact live count) so nothing is lost.

The transition back from fallback->primary also emits a structured
``entropy.recovered`` event so the operator sees the all-clear in the
same log stream.

iter-49 defensive audit (2026-05-25)
------------------------------------
The "fallback NEVER aborts the completion" contract that the qr-llm-chat
iter-49 regenerate-banner depends on holds in this implementation:

* ``get_random_bytes`` ONLY catches :class:`EntropyUnavailableError` from
  the primary. On catch, the fallback (typically ``SystemEntropySource``,
  which wraps ``os.urandom``) is called unconditionally.
* ``os.urandom`` does not raise in practice on Linux containers; the only
  realistic raise path is a catastrophic fallback failure, in which case
  the wrapper re-raises ``EntropyUnavailableError`` — that propagates to
  vLLM and would fail the completion. This is the documented contract
  (see ``get_random_bytes`` docstring), but it has not been observed in
  any deploy to date.
* No "circuit breaker open" / "max fallback exhausted" path exists; the
  wrapper degrades on every fetch independently and recovers on the
  first successful primary call.

Conclusion: the iter-49 banner only ever needs to surface a soft warning
(QRNG was degraded), never an error.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from qr_sampler.entropy.base import EntropySource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.telemetry.status_file import write_entropy_status

logger = logging.getLogger("qr_sampler")

# Minimum seconds between consecutive ``entropy.degraded.alert`` log records.
# 60s strikes the balance described in the module docstring: loud enough that
# an operator paging on it sees it within a minute, quiet enough that a hot
# inference loop (one fetch per token) does not generate one alert per token.
_ALERT_THROTTLE_S: float = 60.0

# Minimum seconds between non-transition status-file refreshes. Transitions
# (ok→degraded, degraded→ok) always write immediately; mid-outage count
# refreshes are throttled so a hot sampling loop costs at most one tmpfs
# write per second instead of one per token. The /health/entropy banner
# compare only needs "did the count grow across this multi-second request",
# so 1 s of staleness is invisible to it.
_STATUS_REFRESH_MIN_INTERVAL_S: float = 1.0


class FallbackEntropySource(EntropySource):
    """Composition wrapper: tries primary, falls back on ``EntropyUnavailableError``.

    Only catches ``EntropyUnavailableError``. All other exceptions propagate.
    Reports which source was actually used via :attr:`last_source_used`.

    Args:
        primary: The preferred entropy source.
        fallback: The source to use when the primary is unavailable.
    """

    def __init__(self, primary: EntropySource, fallback: EntropySource) -> None:
        self._primary = primary
        self._fallback = fallback
        self._last_source_used: str = primary.name
        # Degradation telemetry. Lazily initialised so a process that never
        # falls back keeps the deque-free fast path.
        self._fallback_count: int = 0
        self._last_alert_monotonic: float = 0.0
        self._currently_degraded: bool = False
        # iter-53: cross-process status channel for /health/entropy (see
        # ``status_file`` module docstring). OPT-IN via
        # ``enable_status_publishing()`` because the vLLM adapter builds
        # one FallbackEntropySource per pre-init pipeline (quantum_grpc
        # AND system) — only the default/quantum lane may own the file,
        # else the system-primary wrapper overwrites it with its own
        # (always-healthy) state.
        self._publish_status: bool = False
        self._last_status_write_monotonic: float = 0.0

    @property
    def name(self) -> str:
        """Return a compound name: ``'<primary>+<fallback>'``."""
        return f"{self._primary.name}+{self._fallback.name}"

    @property
    def is_available(self) -> bool:
        """Returns ``True`` if either the primary or fallback is available."""
        return self._primary.is_available or self._fallback.is_available

    @property
    def primary_name(self) -> str:
        """Name of the primary entropy source."""
        return self._primary.name

    @property
    def last_source_used(self) -> str:
        """Name of the source that provided bytes on the last call."""
        return self._last_source_used

    @property
    def fallback_count(self) -> int:
        """Total number of fallbacks since process start. Test introspection."""
        return self._fallback_count

    @property
    def currently_degraded(self) -> bool:
        """True while the most recent fetch came from the fallback source.

        Flips back to False on the first successful primary fetch (the
        ``entropy.recovered`` transition).
        """
        return self._currently_degraded

    def get_random_bytes(self, n: int) -> bytes:
        """Fetch bytes from the primary source, falling back if unavailable.

        Only ``EntropyUnavailableError`` triggers fallback. All other
        exceptions propagate to the caller unchanged.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes from the primary or fallback source.

        Raises:
            EntropyUnavailableError: If **both** primary and fallback fail.
        """
        return self._fetch_via(n, lambda: self._primary.get_random_bytes(n))

    def prefetch(self, n: int, nonce: int | None = None) -> Any | None:
        """Delegate prefetch to the primary source.

        Never raises: a primary without async support (or one that fails
        to dispatch) yields ``None``, and the redeem path then takes the
        ordinary synchronous fetch-with-fallback route.
        """
        try:
            return self._primary.prefetch(n, nonce)
        except Exception:
            return None

    def get_random_bytes_with_ticket(self, n: int, ticket: Any | None) -> bytes:
        """Redeem a prefetch ticket with identical failover semantics.

        The primary's redeem already degrades internally (failed ticket →
        primary serial retry); only when the primary is *truly* unavailable
        does ``EntropyUnavailableError`` surface here and engage the
        fallback source — with exactly the same degradation telemetry,
        status-file writes, and recovery transitions as the serial path.
        """
        if ticket is None:
            return self.get_random_bytes(n)
        return self._fetch_via(n, lambda: self._primary.get_random_bytes_with_ticket(n, ticket))

    def _fetch_via(self, n: int, fetch_fn: Any) -> bytes:
        """Shared primary-then-fallback flow for serial and ticket fetches."""
        try:
            data: bytes = fetch_fn()
            recovered = self._currently_degraded
            if recovered:
                # Transition back to primary — emit a single all-clear event.
                logger.warning(
                    "entropy.recovered: primary source %r is healthy again "
                    "after %d fallback(s); resuming primary use",
                    self._primary.name,
                    self._fallback_count,
                    extra={
                        "event": "entropy.recovered",
                        "primary": self._primary.name,
                        "fallback": self._fallback.name,
                        "total_fallbacks": self._fallback_count,
                    },
                )
                self._currently_degraded = False
            self._last_source_used = self._primary.name
            if recovered:
                self._write_status(force=True)
            return data
        except EntropyUnavailableError as exc:
            self._fallback_count += 1
            first_fallback = not self._currently_degraded
            self._log_degraded(exc, n)
            data = self._fallback.get_random_bytes(n)
            self._last_source_used = self._fallback.name
            self._write_status(force=first_fallback)
            return data

    def _log_degraded(self, exc: EntropyUnavailableError, n: int) -> None:
        """Emit rate-limited structured warning + alert for a fallback.

        BOTH the structured ``entropy.degraded`` WARNING and the louder
        ``entropy.degraded.alert`` ERROR are throttled together: they fire
        (a) immediately on the FIRST fallback of a degraded window (so the
        operator sees it without waiting up to a minute) and (b) at most once
        per ``_ALERT_THROTTLE_S`` thereafter.

        Why throttle the WARNING too: a sustained primary outage produces one
        fallback PER GENERATED TOKEN, so an unconditional warn-per-fallback
        (the previous behaviour) emitted hundreds of identical
        ``entropy.degraded`` lines per request and drowned ``modal app logs``
        — the operator-flagged spam. The ``fallback_count`` carried on each
        emitted record still lets a grep recover the running total, and the
        status file + ``/health/entropy`` carry the exact live count for
        anything that needs per-request precision.
        """
        now = time.monotonic()
        first_time = not self._currently_degraded
        self._currently_degraded = True

        if not (first_time or (now - self._last_alert_monotonic) >= _ALERT_THROTTLE_S):
            return

        self._last_alert_monotonic = now

        logger.warning(
            "entropy.degraded: primary source %r unavailable (n=%d, err=%r); falling back to %r",
            self._primary.name,
            n,
            str(exc),
            self._fallback.name,
            extra={
                "event": "entropy.degraded",
                "primary": self._primary.name,
                "fallback": self._fallback.name,
                "bytes_requested": n,
                "error": str(exc),
                "fallback_count": self._fallback_count,
            },
        )

        logger.error(
            "ENTROPY DEGRADED: quantum source %r is unavailable; "
            "serving %r (urandom-class) for sampling. "
            "Total fallbacks since process start: %d. "
            "Last error: %s. Operator action: check the entropy server's "
            "health and the gRPC channel to it (qr_sampler.entropy.qgrpc "
            "logs).",
            self._primary.name,
            self._fallback.name,
            self._fallback_count,
            str(exc),
            extra={
                "event": "entropy.degraded.alert",
                "primary": self._primary.name,
                "fallback": self._fallback.name,
                "fallback_count": self._fallback_count,
                "error": str(exc),
            },
        )

    def enable_status_publishing(self) -> None:
        """Mark this wrapper as the owner of the cross-process status file.

        Called by the engine adapter on the DEFAULT pipeline's source
        only. The initial write publishes "no fallbacks yet" so the
        APIServer-side middleware can tell "EngineCore is up, all
        quantum" apart from "EngineCore not initialised yet" (file
        absent). Idempotent.
        """
        self._publish_status = True
        self._write_status(force=True)

    def _write_status(self, *, force: bool) -> None:
        """Publish current state to the cross-process status file.

        No-op unless ``enable_status_publishing()`` was called.
        ``force=True`` on state transitions (and init) writes
        unconditionally; ``force=False`` (mid-outage count refresh) is
        throttled to one write per ``_STATUS_REFRESH_MIN_INTERVAL_S``.
        Best-effort: ``write_entropy_status`` never raises, and the
        throttle timestamp advances even on failed writes so a broken
        tmpdir costs at most one syscall per second, not one per token.
        """
        if not self._publish_status:
            return
        now = time.monotonic()
        if not force and (now - self._last_status_write_monotonic) < (
            _STATUS_REFRESH_MIN_INTERVAL_S
        ):
            return
        self._last_status_write_monotonic = now
        write_entropy_status(
            {
                "primary_name": self._primary.name,
                "fallback_name": self._fallback.name,
                "last_source_used": self._last_source_used,
                "fallback_count": self._fallback_count,
                "currently_degraded": self._currently_degraded,
            }
        )

    def close(self) -> None:
        """Close both primary and fallback sources."""
        self._primary.close()
        self._fallback.close()

    def warmup(self) -> None:
        """Forward warmup to both primary and fallback (both no-op for
        sources without a connection lifecycle).

        The primary's warmup is what matters for the QRNG case — it
        eagerly opens the gRPC channel + verifies reachability. The
        fallback's warmup is a no-op for system PRNG. Both are called
        so any future source with its own connection (e.g. a future
        secondary HTTP-based QRNG) gets the same eager treatment.
        """
        try:
            self._primary.warmup()
        except Exception as exc:
            logger.warning(
                "FallbackEntropySource: primary warmup raised (%s); "
                "fallback path will engage on first fetch",
                exc,
            )
        try:
            self._fallback.warmup()
        except Exception as exc:
            logger.warning(
                "FallbackEntropySource: fallback warmup raised (%s)",
                exc,
            )

    def health_check(self) -> dict[str, Any]:
        """Return health status for both sources.

        Returns:
            Dictionary with overall health and individual source status.
        """
        primary_health = self._primary.health_check()
        fallback_health = self._fallback.health_check()
        return {
            "source": self.name,
            "healthy": self.is_available,
            "primary": primary_health,
            "fallback": fallback_health,
            "last_source_used": self._last_source_used,
        }
