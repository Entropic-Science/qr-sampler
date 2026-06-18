"""ASGI middleware that adds a *passive* ``GET /health/entropy`` route to vLLM.

vLLM's ``vllm serve`` owns its FastAPI app and we cannot register routes on
it directly — but it accepts ``--middleware <fqn>`` for ASGI middlewares
that run before the app's routing. This module is that callable.

Design: PASSIVE reporting, never a live probe
----------------------------------------------
The endpoint reports the qr-sampler's *last-known* entropy health, read
from the cross-process status file that the EngineCore's
``FallbackEntropySource`` writes whenever it actually samples a token
(healthy → degraded → recovered transitions, see
``qr_sampler.entropy.status_file``). It does NOT open a gRPC channel or
fetch bytes from the QRNG when polled.

Why passive: every ``GET`` to this route is an external request that wakes
/ keeps-warm the H100 Modal container, and the old design ALSO fired a
live 8-byte gRPC round-trip at the QRNG on each poll. A status chip does
not need either. The one-and-only liveness check happens once, at
container startup, in ``QuantumGrpcSource.warmup()`` (open channel +
verify + fall back on failure). After that we "just go" and report what
real traffic observed — no recurring health-checking of the gRPC lane.

Payload shape::

    {
        "rpc_ok": <bool|null>,          # verdict; null => not initialised
        "tcp_ok": null,                 # retained key; no live TCP probe
        "summary": <str>,               # one-line human-readable state
        "fallback_count": <int|null>,   # process-wide fallback counter
        "last_source_used": <str|null>, # e.g. "quantum_grpc" / "system"
        "primary_name": <str|null>,     # configured primary
        "sampler": { ... } | null,      # EngineCore state + "age_s"
        "sampler_source": "in_process" | "status_file" | "none",
        "perf": { ... } | null          # iter-55 per-stage sampling perf
    }

``rpc_ok`` semantics
--------------------
* ``true``  — sampler state known and not currently in PRNG fallback.
* ``false`` — sampler currently in PRNG fallback (responses are, or were
  most recently, PRNG-sampled). The qr-llm-chat filter surfaces this as
  the amber "using PRNG fallback" banner.
* ``null`` (HTTP 503, ``error: not_initialised``) — no sampler state yet
  (cold-start window before EngineCore builds its first pipeline, or a
  deploy with no status publisher).

Pass-through: any path other than ``/health/entropy`` is forwarded
unchanged. Hot-path cost is one path-string comparison per request.

Wire-up: the FQN passed to ``vllm serve --middleware`` uses dots
throughout (vLLM rsplits on the rightmost ``.``)::

    "--middleware",
    "qr_sampler.connectors.modal.health_entropy_middleware.health_entropy_middleware"

NOT the ``module:callable`` colon form — vLLM treats the whole
post-rightmost-dot string as the attribute name, so a colon there yields
an unfindable ``foo:foo`` and the container crashes AFTER engine init.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse

from qr_sampler.entropy.status_file import read_entropy_status

if TYPE_CHECKING:
    from fastapi import Request

# In-process FallbackEntropySource reference. Populated by
# ``VLLMAdapter.__init__`` via ``set_fallback_source`` and only effective
# when the adapter shares this middleware's process (unit tests,
# in-process engines). Production vLLM splits APIServer / EngineCore into
# separate processes, so there the status file is the channel.
_FALLBACK_SRC: Any = None


def set_fallback_source(src: Any) -> None:
    """Register the FallbackEntropySource the endpoint should report on.

    Idempotent; a second call replaces the reference. Only effective in a
    single-process layout — see module docstring.
    """
    global _FALLBACK_SRC
    _FALLBACK_SRC = src


def get_fallback_source() -> Any:
    """Test introspection — returns whatever ``set_fallback_source`` last stored."""
    return _FALLBACK_SRC


def _resolve_sampler_state() -> tuple[dict[str, Any] | None, str]:
    """Read EngineCore sampler state: in-process ref first, then the file.

    Returns ``(state, source_tag)`` where ``state`` carries
    ``primary_name`` / ``last_source_used`` / ``fallback_count`` /
    ``currently_degraded`` / ``age_s``, or ``(None, "none")`` when neither
    channel has data yet.
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


def _rpc_ok_from_sampler(sampler: dict[str, Any] | None) -> bool | None:
    """Passive verdict: healthy unless the sampler is in fallback.

    ``None`` when there is no sampler state at all (not initialised).
    """
    if sampler is None:
        return None
    return not bool(sampler.get("currently_degraded", False))


def _compose_summary(rpc_ok: bool | None, sampler: dict[str, Any] | None) -> str:
    """One human-readable line for the deploy-guard splash + log greps."""
    if rpc_ok is True:
        return "quantum entropy OK (sampler-reported)"
    if rpc_ok is False:
        count = sampler.get("fallback_count") if sampler else None
        return f"QRNG degraded: sampler in PRNG fallback (count={count})"
    return "entropy state unknown (not initialised)"


async def health_entropy_middleware(request: Request, call_next: Any) -> Any:
    """ASGI middleware: short-circuit ``GET /health/entropy``, pass through the rest.

    Reports last-known entropy health from the status file (or in-process
    source); never opens a gRPC channel. See module docstring for the
    payload contract and the passive-by-design rationale.
    """
    if request.url.path == "/health/entropy" and request.method == "GET":
        sampler, sampler_source = _resolve_sampler_state()
        rpc_ok = _rpc_ok_from_sampler(sampler)

        # iter-55: per-stage sampling-perf aggregate rides along when the
        # EngineCore adapter publishes it. Best-effort: absent file → null
        # block, never an error.
        perf: dict[str, Any] | None = None
        try:
            from qr_sampler.entropy.status_file import read_perf_status

            perf = read_perf_status()
            if perf is not None and isinstance(perf.get("updated_at"), (int, float)):
                perf["age_s"] = round(time.time() - float(perf["updated_at"]), 1)
        except Exception:
            perf = None

        payload: dict[str, Any] = {
            "rpc_ok": rpc_ok,
            "tcp_ok": None,  # retained for shape compat; no live TCP probe
            "summary": _compose_summary(rpc_ok, sampler),
            "fallback_count": sampler.get("fallback_count") if sampler else None,
            "last_source_used": sampler.get("last_source_used") if sampler else None,
            "primary_name": sampler.get("primary_name") if sampler else None,
            "sampler": sampler,
            "sampler_source": sampler_source,
            "perf": perf,
        }
        if rpc_ok is None:
            payload["error"] = "not_initialised"
            return JSONResponse(payload, status_code=503)
        return JSONResponse(payload)
    return await call_next(request)
