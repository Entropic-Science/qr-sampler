"""Per-stage sampling-cost telemetry for the vLLM adapter."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger("qr_sampler")


class _PerfAggregator:
    """Rolling per-stage sampling-cost window + periodic publication.

    iter-55. The qr_sampler logger's INFO lines are invisible in the
    production EngineCore process (root logger at WARNING), which left
    per-token costs unobservable. This aggregator keeps a rolling window
    of per-stage timings plus cumulative prefetch/echo counters, and
    publishes them through two visible channels:

    * the perf status file (read by /health/entropy's ``perf`` block in
      the APIServer process), refreshed at most every
      ``PUBLISH_MIN_INTERVAL_S`` / ``PUBLISH_EVERY_TOKENS``;
    * a rate-limited WARNING log line (``qr.sampling.stats``) so the
      breakdown also lands in the serving host's log stream.

    Single-threaded by construction: ``note()`` is only called from the
    engine's apply loop.
    """

    WINDOW = 512
    PUBLISH_EVERY_TOKENS = 256
    PUBLISH_MIN_INTERVAL_S = 15.0
    LOG_MIN_INTERVAL_S = 60.0

    _STAGES = (
        "to_numpy",
        "temperature",
        "entropy_wait",
        "amplify",
        "select",
        "onehot",
        "total",
    )

    def __init__(self) -> None:
        self._stages: dict[str, deque[float]] = {
            name: deque(maxlen=self.WINDOW) for name in self._STAGES
        }
        self.tokens_total = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.echo_verified = 0
        self.fallback_tokens = 0
        self._since_publish = 0
        self._last_publish_monotonic = 0.0
        self._last_log_monotonic = 0.0

    def note(self, record: Any, to_numpy_ms: float, onehot_ms: float) -> None:
        """Fold one token's timings into the window; publish when due."""
        self.tokens_total += 1
        self._since_publish += 1
        self._stages["to_numpy"].append(to_numpy_ms)
        self._stages["onehot"].append(onehot_ms)
        for stage, value in (
            ("temperature", record.temperature_ms),
            ("entropy_wait", record.entropy_fetch_ms),
            ("amplify", record.amplify_ms),
            ("select", record.select_ms),
            ("total", record.total_sampling_ms),
        ):
            if value is not None:
                self._stages[stage].append(float(value))
        if record.entropy_prefetch_hit is True:
            self.prefetch_hits += 1
            if record.entropy_echo_verified:
                self.echo_verified += 1
        elif record.entropy_prefetch_hit is False:
            self.prefetch_misses += 1
        if record.entropy_is_fallback:
            self.fallback_tokens += 1

        now = time.monotonic()
        due = self._since_publish >= self.PUBLISH_EVERY_TOKENS or (
            (now - self._last_publish_monotonic) >= self.PUBLISH_MIN_INTERVAL_S
        )
        if due:
            self._publish(now)

    def snapshot(self) -> dict[str, Any]:
        """Build the publishable aggregate payload."""
        stage_ms: dict[str, dict[str, float]] = {}
        for name, window in self._stages.items():
            if not window:
                continue
            values = sorted(window)
            p95 = values[min(int(len(values) * 0.95), len(values) - 1)]
            stage_ms[name] = {
                "avg": round(sum(values) / len(values), 3),
                "p95": round(p95, 3),
            }
        fired = self.prefetch_hits + self.prefetch_misses
        return {
            "window_tokens": len(self._stages["total"]),
            "tokens_total": self.tokens_total,
            "stage_ms": stage_ms,
            "prefetch": {
                "hits": self.prefetch_hits,
                "misses": self.prefetch_misses,
                "hit_ratio": round(self.prefetch_hits / fired, 4) if fired else None,
                "echo_verified_ratio": (
                    round(self.echo_verified / self.prefetch_hits, 4)
                    if self.prefetch_hits
                    else None
                ),
            },
            "fallback_tokens_total": self.fallback_tokens,
        }

    def _publish(self, now: float) -> None:
        self._since_publish = 0
        self._last_publish_monotonic = now
        payload = self.snapshot()
        try:
            from qr_sampler.telemetry.status_file import write_perf_status

            write_perf_status(payload)
        except Exception:
            # Telemetry must never break the sampling loop.
            pass
        if (now - self._last_log_monotonic) >= self.LOG_MIN_INTERVAL_S:
            self._last_log_monotonic = now
            total = payload["stage_ms"].get("total", {})
            prefetch = payload["prefetch"]
            # WARNING on purpose: INFO from this logger is filtered out of
            # the production EngineCore log stream (see class docstring).
            logger.warning(
                "qr.sampling.stats: total avg=%.1fms p95=%.1fms over %d-token "
                "window; prefetch hit_ratio=%s echo_verified_ratio=%s",
                total.get("avg", -1.0),
                total.get("p95", -1.0),
                payload["window_tokens"],
                prefetch["hit_ratio"],
                prefetch["echo_verified_ratio"],
                extra={"event": "qr.sampling.stats", "stats": payload},
            )
