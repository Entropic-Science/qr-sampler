"""Adaptive-P99 circuit breaker — a pure state machine (no I/O, no logging).

Tracks a rolling latency window, computes an adaptive per-call timeout from
its P99, counts consecutive failures, and manages the open/half-open cycle
with exponential backoff on the recovery window. The gRPC entropy source
composes this class and does all the logging/channel work itself.

Thread safety: the engine adapter samples batch rows on concurrent worker
threads (perf tranche 2026-07), so entropy fetches — and therefore the
breaker's bookkeeping — run concurrently. Compound state transitions are
guarded by an internal lock; the simple attribute reads used for health
reporting stay lock-free (single-word reads are atomic under the GIL).
"""

from __future__ import annotations

import threading
import time
from collections import deque


class AdaptiveCircuitBreaker:
    """Rolling-P99 adaptive timeout + consecutive-failure circuit breaker.

    State is deliberately public (simple attributes) — the owner reads it
    for health reporting and tests assert on it directly.

    Args:
        window_size: Rolling latency window size for P99 computation.
        min_timeout_ms: Floor for the adaptive timeout.
        timeout_multiplier: Multiplier applied to P99 for the timeout.
        max_timeout_ms: Hard cap for the adaptive timeout (the configured
            gRPC call timeout); also the initial P99 seed.
        recovery_window_s: BASE seconds before the first half-open retry.
        recovery_window_max_s: Ceiling for the exponentially-backed-off
            recovery window.
        max_consecutive_failures: Failures before the circuit opens.
    """

    def __init__(
        self,
        *,
        window_size: int,
        min_timeout_ms: float,
        timeout_multiplier: float,
        max_timeout_ms: float,
        recovery_window_s: float,
        recovery_window_max_s: float,
        max_consecutive_failures: int,
    ) -> None:
        self._min_timeout_ms = min_timeout_ms
        self._timeout_multiplier = timeout_multiplier
        self._max_timeout_ms = max_timeout_ms
        self._recovery_window_s = recovery_window_s
        self.recovery_window_max_s = recovery_window_max_s
        self.max_consecutive_failures = max_consecutive_failures

        self._lock = threading.Lock()
        self._latency_window: deque[float] = deque(maxlen=window_size)
        self.p99_ms: float = max_timeout_ms
        self.consecutive_failures: int = 0
        self.circuit_open: bool = False
        self.circuit_open_until: float = 0.0
        # Number of consecutive circuit opens WITHOUT an intervening
        # successful fetch. Drives the exponential backoff on the recovery
        # window so a sustained outage is probed ever less often instead of
        # every base-window seconds. Reset to 0 on the first success.
        self.circuit_open_count: int = 0

    @property
    def is_blocking(self) -> bool:
        """True while the circuit is open AND the recovery window has not elapsed."""
        return self.circuit_open and time.monotonic() < self.circuit_open_until

    @property
    def latency_sample_count(self) -> int:
        """Number of samples currently in the rolling window."""
        return len(self._latency_window)

    def record_latency(self, elapsed_ms: float) -> None:
        """Add a latency sample to the rolling window and recompute P99.

        Args:
            elapsed_ms: Time taken for the last fetch in milliseconds.
        """
        with self._lock:
            self._latency_window.append(elapsed_ms)
            if len(self._latency_window) >= 10:
                sorted_latencies = sorted(self._latency_window)
                idx = int(len(sorted_latencies) * 0.99)
                idx = min(idx, len(sorted_latencies) - 1)
                self.p99_ms = sorted_latencies[idx]

    def record_censored_latency(self, elapsed_s: float, timeout_s: float) -> None:
        """Feed timeout-shaped failures into the adaptive-latency window.

        iter-53 (2026-06-09): only successful fetches used to call
        ``record_latency``, which made the adaptive timeout a one-way
        ratchet — once P99 converged on a fast window (observed ~96 ms
        through the tunnel → ~145 ms ceiling), any genuine upward drift in
        backend latency could NEVER be re-learned: every slower fetch was
        cut off at the old ceiling and discarded, so the window stayed
        frozen and the source flapped between timeout and fallback
        indefinitely.

        Recording the cut-off duration as a latency sample lets P99 climb
        toward the observed (censored) latency, raising the next adaptive
        timeout by up to ``timeout_multiplier`` per re-learn round,
        hard-capped by ``max_timeout_ms``. Fast failures (connection
        refused, channel teardown) are excluded — they say nothing about
        latency and would wrongly DEFLATE the window.
        """
        elapsed_ms = elapsed_s * 1000.0
        if elapsed_ms >= 0.8 * timeout_s * 1000.0:
            self.record_latency(elapsed_ms)

    def timeout_ms(self) -> float:
        """Compute the adaptive timeout in milliseconds.

        Returns:
            ``max(min_timeout, P99 * multiplier)`` or the configured hard
            cap, whichever is smaller.
        """
        adaptive = max(self._min_timeout_ms, self.p99_ms * self._timeout_multiplier)
        return min(adaptive, self._max_timeout_ms)

    def note_success(self, elapsed_ms: float) -> None:
        """Record one successful fetch: latency sample + failure-state reset."""
        self.record_latency(elapsed_ms)
        with self._lock:
            self.consecutive_failures = 0
            self.circuit_open_count = 0

    def try_half_open(self) -> bool:
        """Attempt the half-open transition on an open circuit.

        Returns:
            True when the recovery window has elapsed — the circuit flips
            closed for one trial request. False while still inside the
            window (the caller must keep failing fast).
        """
        with self._lock:
            if time.monotonic() >= self.circuit_open_until:
                self.circuit_open = False
                return True
            return False

    def note_failure(self) -> float | None:
        """Record one retry-exhausted failure; open the circuit at the threshold.

        The FIRST open uses the base recovery window (short — so a transient
        stale channel recovers in one half-open cycle); each subsequent open
        without an intervening success doubles the wait, capped at
        ``recovery_window_max_s``. This is what stops a genuine outage from
        being re-probed every few seconds — each half-open tears down +
        rebuilds the channel and fires fresh connects, so a fixed short
        cadence would hammer a dead server. ``circuit_open_count`` resets to
        0 on the next successful fetch (see :meth:`note_success`).

        Returns:
            The recovery window (seconds) when this failure opened the
            circuit, else ``None``.
        """
        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                window: float = min(
                    self._recovery_window_s * float(2**self.circuit_open_count),
                    self.recovery_window_max_s,
                )
                self.circuit_open = True
                self.circuit_open_until = time.monotonic() + window
                self.circuit_open_count += 1
                return window
            return None
