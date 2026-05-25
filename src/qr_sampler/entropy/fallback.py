"""Fallback entropy source — composition wrapper with transparent failover.

``FallbackEntropySource`` wraps a *primary* and a *fallback* source. When the
primary raises :class:`~qr_sampler.exceptions.EntropyUnavailableError`, the
wrapper transparently delegates to the fallback. **All other exceptions
propagate unchanged** — this is deliberate: only entropy-unavailability is a
recoverable condition.

Operator-visible logging
------------------------
Every fallback emits a structured ``entropy.degraded`` warning, and a louder
``entropy.degraded.alert`` is rate-limited to once per minute. The two-tier
shape is important: a noisy primary outage produces one entropy fetch *per
token*, so an unconditional warn-per-fallback would drown ``modal app logs``.
The per-minute alert is the human-readable signal; the per-event warning
gives the diagnostic count when you grep for it.

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

logger = logging.getLogger("qr_sampler")

# Minimum seconds between consecutive ``entropy.degraded.alert`` log records.
# 60s strikes the balance described in the module docstring: loud enough that
# an operator paging on it sees it within a minute, quiet enough that a hot
# inference loop (one fetch per token) does not generate one alert per token.
_ALERT_THROTTLE_S: float = 60.0


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
        try:
            data = self._primary.get_random_bytes(n)
            if self._currently_degraded:
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
            return data
        except EntropyUnavailableError as exc:
            self._fallback_count += 1
            self._log_degraded(exc, n)
            data = self._fallback.get_random_bytes(n)
            self._last_source_used = self._fallback.name
            return data

    def _log_degraded(self, exc: EntropyUnavailableError, n: int) -> None:
        """Emit per-event + rate-limited-alert structured warnings.

        Per-event ``entropy.degraded`` lands at WARNING and carries enough
        structured context for downstream filtering (primary name, fallback
        name, byte count, error stringification, monotonic count). The
        per-event log lets the operator confirm the EXACT request that
        degraded; the alert below makes sure they notice in the first place.

        ``entropy.degraded.alert`` is throttled to once per ``_ALERT_THROTTLE_S``
        seconds so a sustained outage (one fallback per generated token) does
        not flood ``modal app logs``. The alert message is intentionally more
        human-readable so it grabs attention in a paging dashboard.
        """
        now = time.monotonic()
        first_time = not self._currently_degraded
        self._currently_degraded = True

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

        # Emit the louder alert (a) immediately on the FIRST fallback of a
        # degraded window (so the operator sees it without waiting up to a
        # minute) and (b) at most once per throttle window thereafter.
        if first_time or (now - self._last_alert_monotonic) >= _ALERT_THROTTLE_S:
            self._last_alert_monotonic = now
            logger.error(
                "ENTROPY DEGRADED: quantum source %r is unavailable; "
                "serving %r (urandom-class) for sampling. "
                "Total fallbacks since process start: %d. "
                "Last error: %s. Operator action: check the cloudflared "
                "sidecar (qr_sampler.cloudflared logs) and the QRNG service "
                "health.",
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

    def close(self) -> None:
        """Close both primary and fallback sources."""
        self._primary.close()
        self._fallback.close()

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
