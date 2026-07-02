"""Unit tests for the pure AdaptiveCircuitBreaker state machine."""

from __future__ import annotations

import time

from qr_sampler.entropy.qgrpc.breaker import AdaptiveCircuitBreaker


def _make_breaker(**overrides: object) -> AdaptiveCircuitBreaker:
    params: dict[str, object] = {
        "window_size": 100,
        "min_timeout_ms": 5.0,
        "timeout_multiplier": 1.5,
        "max_timeout_ms": 5000.0,
        "recovery_window_s": 10.0,
        "recovery_window_max_s": 60.0,
        "max_consecutive_failures": 3,
    }
    params.update(overrides)
    return AdaptiveCircuitBreaker(**params)  # type: ignore[arg-type]


class TestAdaptiveTimeout:
    """P99 tracking + adaptive timeout math."""

    def test_p99_seeds_at_max_timeout(self) -> None:
        breaker = _make_breaker()
        assert breaker.p99_ms == 5000.0

    def test_update_latency_and_timeout(self) -> None:
        """P99 and timeout should update from the latency window."""
        breaker = _make_breaker()
        for i in range(20):
            breaker.record_latency(float(i))

        # P99 should be near the max of the window.
        assert breaker.p99_ms >= 15.0

        # Adaptive timeout: max(5ms, P99 * 1.5), capped at max_timeout_ms.
        timeout = breaker.timeout_ms()
        assert timeout >= 5.0
        assert timeout <= 5000.0

    def test_p99_not_recomputed_below_ten_samples(self) -> None:
        breaker = _make_breaker()
        for i in range(9):
            breaker.record_latency(float(i))
        assert breaker.p99_ms == 5000.0  # still the seed

    def test_timeout_floor_and_cap(self) -> None:
        breaker = _make_breaker(min_timeout_ms=50.0, max_timeout_ms=100.0)
        for _ in range(10):
            breaker.record_latency(1.0)  # tiny latencies
        assert breaker.timeout_ms() == 50.0  # floor engages
        for _ in range(10):
            breaker.record_latency(10_000.0)
        assert breaker.timeout_ms() == 100.0  # hard cap engages

    def test_censored_latency_recorded_only_near_timeout(self) -> None:
        """iter-53: timeout-shaped failures feed the window; fast failures don't."""
        breaker = _make_breaker()
        # Fast failure (well under 80% of the budget): ignored.
        breaker.record_censored_latency(elapsed_s=0.01, timeout_s=1.0)
        assert breaker.latency_sample_count == 0
        # Timeout-shaped failure (>= 80% of the budget): recorded.
        breaker.record_censored_latency(elapsed_s=0.9, timeout_s=1.0)
        assert breaker.latency_sample_count == 1


class TestOpenHalfOpenCycle:
    """Consecutive-failure threshold + exponential recovery backoff."""

    def test_opens_at_threshold_with_base_window(self) -> None:
        breaker = _make_breaker(max_consecutive_failures=3, recovery_window_s=10.0)
        assert breaker.note_failure() is None
        assert breaker.note_failure() is None
        window = breaker.note_failure()
        assert window == 10.0
        assert breaker.circuit_open is True
        assert breaker.circuit_open_count == 1
        assert breaker.is_blocking is True

    def test_backoff_doubles_and_caps(self) -> None:
        breaker = _make_breaker(
            max_consecutive_failures=1, recovery_window_s=10.0, recovery_window_max_s=25.0
        )
        assert breaker.note_failure() == 10.0
        assert breaker.note_failure() == 20.0
        assert breaker.note_failure() == 25.0  # capped
        assert breaker.note_failure() == 25.0

    def test_success_resets_failures_and_backoff(self) -> None:
        breaker = _make_breaker(max_consecutive_failures=1)
        breaker.note_failure()
        breaker.note_success(1.0)
        assert breaker.consecutive_failures == 0
        assert breaker.circuit_open_count == 0

    def test_try_half_open_after_window(self) -> None:
        breaker = _make_breaker(max_consecutive_failures=1, recovery_window_s=0.0)
        breaker.note_failure()
        assert breaker.circuit_open is True
        assert breaker.try_half_open() is True
        assert breaker.circuit_open is False

    def test_try_half_open_blocked_inside_window(self) -> None:
        breaker = _make_breaker(max_consecutive_failures=1, recovery_window_s=100.0)
        breaker.note_failure()
        assert breaker.circuit_open_until > time.monotonic()
        assert breaker.try_half_open() is False
        assert breaker.circuit_open is True
