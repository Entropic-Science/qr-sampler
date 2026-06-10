"""ASGI middleware that adds a ``GET /health/entropy`` route to vLLM's app.

iter-49 (2026-05-25). vLLM's ``vllm serve`` owns its FastAPI app and we
cannot register routes on it directly — but it does accept
``--middleware <fqn>`` for ASGI middlewares that run before the app's
routing. This module exposes that callable.

The middleware short-circuits ``GET /health/entropy`` with a JSON
payload describing the qr-sampler's current entropy-source health:

    {
        "rpc_ok": <bool|null>,      # overall verdict (see semantics below)
        "tcp_ok": <bool|null>,      # bare TCP connect to the gRPC address
        "summary": <str>,           # one-line human-readable state
        "fallback_count": <int|null>,   # process-wide fallback counter
        "last_source_used": <str|null>, # e.g. "quantum_grpc" or "system"
        "primary_name": <str|null>,     # "quantum_grpc" etc.
        "probe": {                  # live QRNG round-trip from THIS process
            "ok": <bool|null>, "tcp_ok": <bool|null>,
            "latency_ms": <float|null>, "error": <str|null>
        },
        "sampler": { ... } | null,  # EngineCore state + "age_s" staleness
        "sampler_source": "in_process" | "status_file" | "none"
    }

``tcp_ok`` and ``summary`` honour the contract the qr-llm-chat
``setup_orchestrator._probe_vllm_entropy`` deploy-guard phase has
expected since iter-14 (tcp_ok distinguishes "cloudflared sidecar not
listening" from "sidecar up but QRNG/auth failing" in its warn copy).

iter-53 (2026-06-09): cross-process wiring FIXED
------------------------------------------------
vLLM runs APIServer (this middleware) and EngineCore (the qr-sampler
LogitsProcessor, where fallbacks actually happen) as separate processes
with independent module globals, so the iter-49 ``set_fallback_source``
in-process channel never populated in production and the endpoint
returned an unconditional 503. Two signal paths now close that gap:

1. **Status file** (EngineCore → APIServer): ``FallbackEntropySource``
   atomically writes ``qr_entropy_status.json`` (see
   ``qr_sampler.entropy.status_file``) on every degraded/recovered
   transition plus throttled mid-outage count refreshes. This carries
   ``fallback_count`` for the qr-llm-chat iter-49 regenerate-banner
   (snapshot-before/after-request compare) and ``currently_degraded``
   for the verdict.

2. **Live gRPC probe** (APIServer-local): a lazily-constructed
   ``QuantumGrpcSource`` (retry 0, ≤1 s timeout, verdict cached
   ``_PROBE_TTL_S``) fetches 8 bytes end-to-end through the cloudflared
   sidecar on each poll. This answers "is the QRNG lane up RIGHT NOW"
   even when the sampler state is stale because no tokens have been
   generated recently. The probe is skipped (``ok: null``) when the
   configured primary isn't ``quantum_grpc`` or grpcio is unavailable.

``rpc_ok`` verdict semantics
----------------------------
* ``false`` — live probe failed, OR the sampler reported a degraded
  window within the last ``_SAMPLER_DEGRADED_FRESH_S`` seconds. Either
  way: responses are (or imminently will be) PRNG-sampled.
* ``true``  — live probe succeeded and no fresh degraded state.
* ``null`` (HTTP 503, ``error: not_initialised``) — no probe AND no
  sampler state; genuinely unknown (cold-start window before EngineCore
  builds its first pipeline, on a non-quantum deploy).

The ``set_fallback_source`` in-process channel is retained: when it IS
populated (single-process unit tests, future in-process engines) it is
strictly fresher than the file and takes precedence.

Pass-through path
-----------------
Any request whose path is not ``/health/entropy`` is forwarded to vLLM
unchanged via ``await call_next(request)``. Performance impact on the
hot inference path is one path-string comparison per request.

Wire-up location
----------------
The qr-sampler-side ``app.py`` ``_start_and_sleep`` method imports
this module on the deploy host BEFORE passing the flag to vllm serve;
if the import fails, ``--middleware`` is skipped and the endpoint
stays 404. The middleware FQN passed to vllm serve uses dots
throughout (vLLM rsplits on the rightmost ``.``)::

    "--middleware",
    "qr_sampler.connectors.modal.health_entropy_middleware.health_entropy_middleware"

NOT the ``module:callable`` colon form that ``--logits-processors``
uses — vLLM treats the entire post-rightmost-dot string as the
attribute name, so a colon there yields an unfindable attribute
``foo:foo`` and the container crashes AFTER engine init.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

from qr_sampler.entropy.status_file import read_entropy_status

if TYPE_CHECKING:
    from fastapi import Request

_FALLBACK_SRC: Any = None

# Seconds a probe verdict (success OR failure) stays cached. The OWUI
# qr-status poller caches for 10 s and the filter probes once per chat
# request, so 5 s keeps the worst-case staleness well under one chip
# refresh while bounding QRNG probe traffic to ~12 fetches/minute.
_PROBE_TTL_S = 5.0

# Hard cap on the probe's gRPC timeout. The qr-llm-chat consumers give
# the whole HTTP request a 1.5 s budget; the probe must come in under
# that including FastAPI overhead.
_PROBE_TIMEOUT_MS = 1000.0

# Probe fetch size. 8 bytes keeps QRNG bandwidth cost negligible.
_PROBE_BYTES = 8

# How fresh a sampler-reported degraded window must be to drive the
# verdict. Past this age with no traffic, the engine would re-try the
# primary on its next fetch anyway (circuit breaker half-opens after
# 10 s), so the live probe is the more truthful signal.
_SAMPLER_DEGRADED_FRESH_S = 30.0

_probe_lock = threading.Lock()
_probe_source: Any | None = None
_probe_disabled_reason: str | None = None
# (monotonic_expires_at, payload) — payload is the "probe" response field.
_probe_cache: tuple[float, dict[str, Any]] | None = None


def set_fallback_source(src: Any) -> None:
    """Register the FallbackEntropySource the endpoint should report on.

    Called by ``VLLMAdapter.__init__`` once its pipelines are built.
    Only effective when the adapter shares the middleware's process
    (unit tests, in-process engines); the production vLLM split-process
    layout uses the status file instead. Idempotent: a second call
    replaces the reference.
    """
    global _FALLBACK_SRC
    _FALLBACK_SRC = src


def get_fallback_source() -> Any:
    """Test introspection — returns whatever ``set_fallback_source`` last stored."""
    return _FALLBACK_SRC


def reset_probe_state_for_tests() -> None:
    """Drop the cached probe source/verdict so tests get a clean slate."""
    global _probe_source, _probe_disabled_reason, _probe_cache
    with _probe_lock:
        if _probe_source is not None:
            with contextlib.suppress(Exception):
                _probe_source.close()
        _probe_source = None
        _probe_disabled_reason = None
        _probe_cache = None


def _live_probe_sync() -> dict[str, Any]:
    """Round-trip ``_PROBE_BYTES`` through the QRNG lane, with a TTL cache.

    Runs in a worker thread (``asyncio.to_thread``) because
    ``QuantumGrpcSource.get_random_bytes`` blocks — the APIServer event
    loop must keep streaming tokens while a probe is in flight. The lock
    serialises concurrent probes; followers see the fresh cache.

    Returns the ``probe`` payload: ``{"ok": bool|None, "latency_ms":
    float|None, "error": str|None}``. ``ok=None`` means the probe is not
    applicable (non-quantum primary) or could not be constructed.
    """
    global _probe_source, _probe_disabled_reason, _probe_cache
    with _probe_lock:
        now = time.monotonic()
        if _probe_cache is not None and _probe_cache[0] > now:
            return _probe_cache[1]

        if _probe_disabled_reason is not None:
            return {
                "ok": None,
                "tcp_ok": None,
                "latency_ms": None,
                "error": _probe_disabled_reason,
            }

        if _probe_source is None:
            try:
                from qr_sampler.config import QRSamplerConfig

                cfg = QRSamplerConfig()
                if cfg.entropy_source_type != "quantum_grpc":
                    _probe_disabled_reason = (
                        f"probe n/a: primary is {cfg.entropy_source_type!r}"
                    )
                    return {
                        "ok": None,
                        "tcp_ok": None,
                        "latency_ms": None,
                        "error": _probe_disabled_reason,
                    }
                from qr_sampler.entropy.quantum import QuantumGrpcSource

                probe_cfg = QRSamplerConfig(
                    grpc_timeout_ms=min(cfg.grpc_timeout_ms, _PROBE_TIMEOUT_MS),
                    grpc_retry_count=0,
                )
                _probe_source = QuantumGrpcSource(probe_cfg)
            except Exception as exc:  # ImportError, ConfigValidationError, ...
                _probe_disabled_reason = (
                    f"probe init failed: {type(exc).__name__}: {exc}"
                )
                return {
                    "ok": None,
                    "tcp_ok": None,
                    "latency_ms": None,
                    "error": _probe_disabled_reason,
                }

        tcp_ok = _tcp_connect_ok(getattr(_probe_source, "_address", ""))
        t0 = time.perf_counter()
        try:
            _probe_source.get_random_bytes(_PROBE_BYTES)
            payload: dict[str, Any] = {
                "ok": True,
                "tcp_ok": tcp_ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "error": None,
            }
        except Exception as exc:
            payload = {
                "ok": False,
                "tcp_ok": tcp_ok,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "error": f"{type(exc).__name__}: {exc}",
            }
        _probe_cache = (now + _PROBE_TTL_S, payload)
        return payload


def _tcp_connect_ok(address: str) -> bool | None:
    """Bare TCP connect to ``host:port``; the deploy-guard's sidecar signal.

    Distinguishes "cloudflared sidecar not listening" (``False``) from
    "sidecar up, gRPC/auth failing behind it" (``True`` with a failed
    fetch). ``None`` for unparseable addresses (unix sockets etc.).
    """
    host, _, port_s = address.partition(":")
    try:
        port = int(port_s)
    except ValueError:
        return None
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _resolve_sampler_state() -> tuple[dict[str, Any] | None, str]:
    """Read EngineCore sampler state: in-process ref first, then the file.

    Returns ``(state, source_tag)`` where ``state`` carries
    ``primary_name`` / ``last_source_used`` / ``fallback_count`` /
    ``currently_degraded`` / ``age_s``, or ``(None, "none")`` when
    neither channel has data yet.
    """
    src = _FALLBACK_SRC
    if src is not None:
        last = str(getattr(src, "last_source_used", ""))
        primary = str(getattr(src, "primary_name", ""))
        degraded = getattr(src, "currently_degraded", None)
        if not isinstance(degraded, bool):
            degraded = bool(primary) and last != primary
        return (
            {
                "primary_name": primary,
                "last_source_used": last,
                "fallback_count": int(getattr(src, "fallback_count", 0)),
                "currently_degraded": degraded,
                "age_s": 0.0,
            },
            "in_process",
        )

    data = read_entropy_status()
    if data is None:
        return (None, "none")
    try:
        age_s = max(0.0, time.time() - float(data.get("updated_at", 0.0)))
    except (TypeError, ValueError):
        age_s = float("inf")
    fc = data.get("fallback_count")
    return (
        {
            "primary_name": str(data.get("primary_name", "")),
            "last_source_used": str(data.get("last_source_used", "")),
            "fallback_count": fc if isinstance(fc, int) else None,
            "currently_degraded": bool(data.get("currently_degraded", False)),
            "age_s": round(age_s, 1),
        },
        "status_file",
    )


def _combine_rpc_ok(
    probe_ok: bool | None, sampler: dict[str, Any] | None
) -> bool | None:
    """Fold the live probe and the sampler state into one verdict.

    Precedence: an explicit probe failure or a FRESH degraded window is
    definitive ``False``; a probe success (absent fresh degradation) is
    ``True``; with no probe, fall back to the sampler's degraded flag at
    any age; ``None`` when neither signal exists.
    """
    degraded_fresh = bool(
        sampler is not None
        and sampler.get("currently_degraded")
        and sampler.get("age_s", float("inf")) < _SAMPLER_DEGRADED_FRESH_S
    )
    if probe_ok is False:
        return False
    if degraded_fresh:
        return False
    if probe_ok is True:
        return True
    if sampler is not None:
        return not sampler.get("currently_degraded", False)
    return None


def _compose_summary(
    rpc_ok: bool | None,
    probe: dict[str, Any],
    sampler: dict[str, Any] | None,
) -> str:
    """One human-readable line for the deploy-guard splash + log greps."""
    if rpc_ok is True:
        latency = probe.get("latency_ms")
        if probe.get("ok") and latency is not None:
            return f"quantum entropy OK ({latency:.0f} ms round-trip)"
        return "quantum entropy OK (sampler-reported)"
    if rpc_ok is False:
        if probe.get("ok") is False:
            return f"QRNG unreachable: {probe.get('error') or 'probe failed'}"
        count = sampler.get("fallback_count") if sampler else None
        return f"QRNG degraded: sampler in PRNG fallback (count={count})"
    return "entropy state unknown (not initialised)"


async def health_entropy_middleware(request: Request, call_next: Any) -> Any:
    """ASGI middleware: short-circuit ``GET /health/entropy``, pass through everything else.

    The qr-llm-chat OWUI side polls this endpoint for the amber
    entropy-degraded chip, the inlet "sampling from PRNG" warning, the
    iter-49 regenerate-banner fallback-count compare, and the deploy-
    guard splash phase. See module docstring for the payload contract.
    """
    if request.url.path == "/health/entropy" and request.method == "GET":
        sampler, sampler_source = _resolve_sampler_state()
        probe = await asyncio.to_thread(_live_probe_sync)
        rpc_ok = _combine_rpc_ok(probe.get("ok"), sampler)
        payload: dict[str, Any] = {
            "rpc_ok": rpc_ok,
            "tcp_ok": probe.get("tcp_ok"),
            "summary": _compose_summary(rpc_ok, probe, sampler),
            "fallback_count": sampler.get("fallback_count") if sampler else None,
            "last_source_used": sampler.get("last_source_used") if sampler else None,
            "primary_name": sampler.get("primary_name") if sampler else None,
            "probe": probe,
            "sampler": sampler,
            "sampler_source": sampler_source,
        }
        if rpc_ok is None:
            payload["error"] = "not_initialised"
            return JSONResponse(payload, status_code=503)
        return JSONResponse(payload)
    return await call_next(request)
