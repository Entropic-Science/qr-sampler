"""Passive ``/health/entropy`` route for a bare ``vllm serve`` host.

The retired Modal deployment answered ``GET /health/entropy`` from a custom
serving wrapper; a bare-metal ``vllm serve`` (the qr-server shared-engine
profile) has no such route, so every downstream health consumer — the OWUI
setup guard, the ``/api/qr-status`` chip, and the comparison Pipe's
no-silent-PRNG banner — went blind (404 = "unknown", fail-open: no false
banner, but no banner on a *real* degrade either).

This module restores the route as a stock-vLLM **ASGI middleware**, wired
with zero code changes to vLLM itself::

    vllm serve ... --middleware qr_sampler.engines.vllm.health.entropy_health_middleware

vLLM resolves the value by ``rsplit(".", 1)`` (``module.callable`` — NOT
``module:callable``) and, because :func:`entropy_health_middleware` is a
coroutine function, applies it via ``app.middleware("http")``.

**Passive by design.** vLLM runs the sampler in a separate EngineCore
process; module globals do not cross that boundary. The middleware therefore
answers exclusively from the cross-process status files that
``FallbackEntropySource`` (entropy state) and the perf aggregator already
write (``qr_sampler.telemetry.status_file`` — the file-IPC bridge). It never
opens a gRPC channel or touches the QRNG: a health probe must be O(one file
read) and must not add per-probe load to the entropy daemon.

Payload contract (shape consumed by the OWUI-side readers):

* ``rpc_ok``   — ``True`` when the quantum leg is healthy, ``False`` during a
  labelled-fallback window, ``None`` when no status is available yet (the
  sampler has not drawn since engine start, or telemetry is disabled).
* ``tcp_ok``   — ``True`` iff a status snapshot exists (the engine-side
  sampler is alive and publishing).
* ``summary``  — one human-readable line for logs/operators.
* ``fallback_count`` / ``last_source_used`` / ``primary_name`` — lifted from
  the status snapshot (the Pipe compares ``fallback_count`` across a request
  boundary).
* ``sampler``  — ``{"currently_degraded": bool, "age_s": float}`` — the
  iter-53 degraded-window block the Pipe/Filter banner logic keys on.
* ``gate``     — coherence-gate snapshot (``gate_open`` / ``gate_boost`` /
  ``coherence_valid``) when the writer has published one.
* ``perf``     — the rolling per-stage sampling-cost aggregate, when present.
"""

from __future__ import annotations

import time
from typing import Any

from qr_sampler.telemetry.status_file import read_entropy_status, read_perf_status

#: Path served by the middleware; everything else passes straight through.
HEALTH_PATH = "/health/entropy"

#: Age beyond which the snapshot is flagged stale in the summary. The status
#: file refreshes on every state transition and at ≥1 write/s during draws,
#: so a very old snapshot just means "no draws lately" — still honest data,
#: but worth surfacing to a human reading the summary.
STALE_AFTER_S: float = 900.0


def build_entropy_health_payload(now: float | None = None) -> dict[str, Any]:
    """Compose the ``/health/entropy`` JSON body from the status files.

    Pure and side-effect-free beyond the two file reads; ``now`` is
    injectable for tests. Never raises: an absent/unreadable status file
    degrades to the ``rpc_ok: None`` "unknown" shape (readers treat that as
    fail-open, exactly like the historical 404).
    """
    if now is None:
        now = time.time()

    status = read_entropy_status()
    perf = read_perf_status()

    if not isinstance(status, dict):
        return {
            "tcp_ok": False,
            "rpc_ok": None,
            "summary": (
                "no entropy status yet: the sampler has not drawn since engine "
                "start (or status-file telemetry is disabled)"
            ),
            "fallback_count": None,
            "last_source_used": None,
            "primary_name": None,
            "sampler": None,
            "updated_at": None,
            "age_s": None,
            "perf": perf,
        }

    updated_at = status.get("updated_at")
    age_s: float | None = None
    if isinstance(updated_at, (int, float)):
        age_s = max(0.0, now - float(updated_at))

    degraded = bool(status.get("currently_degraded"))
    fallback_count = status.get("fallback_count")
    last_source = status.get("last_source_used")
    primary_name = status.get("primary_name")
    fallback_name = status.get("fallback_name") or "system"

    if degraded:
        summary = (
            f"quantum entropy DEGRADED — serving labelled '{fallback_name}' "
            f"(fallbacks={fallback_count}, last_source={last_source})"
        )
    else:
        summary = f"quantum entropy ok (last_source={last_source}, fallbacks={fallback_count})"
    if age_s is not None and age_s > STALE_AFTER_S:
        summary += f" [snapshot {age_s:.0f}s old — no recent draws]"

    payload: dict[str, Any] = {
        "tcp_ok": True,
        "rpc_ok": not degraded,
        "summary": summary,
        "fallback_count": fallback_count,
        "last_source_used": last_source,
        "primary_name": primary_name,
        "sampler": {"currently_degraded": degraded, "age_s": age_s},
        "updated_at": updated_at,
        "age_s": age_s,
        "perf": perf,
    }

    # Coherence-gate block: present only when the gate writer has published.
    if "gate_open" in status:
        payload["gate"] = {
            "gate_open": bool(status.get("gate_open")),
            "gate_boost": status.get("gate_boost"),
            "coherence_valid": status.get("coherence_valid"),
        }

    return payload


async def entropy_health_middleware(request: Any, call_next: Any) -> Any:
    """ASGI http middleware: answer ``GET /health/entropy``, pass the rest.

    Wired via ``vllm serve --middleware
    qr_sampler.engines.vllm.health.entropy_health_middleware``. Only the
    exact health path is intercepted; every other request flows to vLLM
    unchanged. The starlette import is deferred so importing this module
    never requires a serving stack (qr-sampler itself does not depend on
    starlette; the vLLM host always provides it).
    """
    if request.url.path == HEALTH_PATH and request.method in ("GET", "HEAD"):
        from starlette.responses import JSONResponse

        return JSONResponse(build_entropy_health_payload())
    return await call_next(request)
