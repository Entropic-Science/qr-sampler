"""ASGI middleware that adds a ``GET /health/entropy`` route to vLLM's app.

iter-49 (2026-05-25). vLLM's ``vllm serve`` owns its FastAPI app and we
cannot register routes on it directly — but it does accept
``--middleware <fqn>`` for ASGI middlewares that run before the app's
routing. This module exposes that callable.

The middleware short-circuits ``GET /health/entropy`` with a JSON
payload describing the qr-sampler's current entropy-source health:

    {
        "rpc_ok": <bool>,           # True iff last fetch hit the primary
        "fallback_count": <int>,    # Process-wide fallback counter
        "last_source_used": <str>,  # e.g. "quantum_grpc" or "system"
        "primary_name": <str>,      # "quantum_grpc" or whatever the
                                    #   FallbackEntropySource's primary
                                    #   reports
    }

A snapshot-before-then-after comparison of ``fallback_count`` (from the
qr-llm-chat OWUI side) is the signal channel for the iter-49
regenerate-banner: if the count incremented across a request, the
quantum lane fell back to PRNG for at least one token, so the response
is no longer purely quantum-random and the user is shown a banner
suggesting they hit OWUI's existing Regenerate button.

State wire-up
-------------
The middleware reads from a module-level ``_FALLBACK_SRC`` reference
populated by ``VLLMAdapter.__init__`` via ``set_fallback_source(...)``.
Set late (after engine init), so the endpoint returns ``503`` with a
``not_initialised`` error tag during the cold-start window before the
LogitsProcessor's first pipeline is built. The qr-llm-chat probe
treats non-200 as "unknown" and no-ops cleanly.

Pass-through path
-----------------
Any request whose path is not ``/health/entropy`` is forwarded to vLLM
unchanged via ``await call_next(request)``. Performance impact on the
hot inference path is one path-string comparison per request.

Wire-up location
----------------
Add to the ``vllm serve`` argv block in
``qr_sampler.connectors.modal.app`` for every active class::

    "--middleware",
    "qr_sampler.connectors.modal.health_entropy_middleware:health_entropy_middleware",
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

_FALLBACK_SRC: Any = None


def set_fallback_source(src: Any) -> None:
    """Register the FallbackEntropySource the endpoint should report on.

    Called by ``VLLMAdapter.__init__`` once its pipelines are built.
    Idempotent: a second call replaces the reference (useful for
    in-process test reinit). Stores in a module-level so the middleware
    callable (loaded by vLLM by string path) reads the same object
    without needing constructor injection.
    """
    global _FALLBACK_SRC
    _FALLBACK_SRC = src


def get_fallback_source() -> Any:
    """Test introspection — returns whatever ``set_fallback_source`` last stored."""
    return _FALLBACK_SRC


async def health_entropy_middleware(request: Request, call_next: Any) -> Any:
    """ASGI middleware: short-circuit ``GET /health/entropy``, pass through everything else.

    The qr-llm-chat OWUI side polls this endpoint to detect QRNG
    fallback engagement across a request boundary. See module docstring.
    """
    if request.url.path == "/health/entropy" and request.method == "GET":
        src = _FALLBACK_SRC
        if src is None:
            return JSONResponse(
                {"rpc_ok": None, "error": "not_initialised"},
                status_code=503,
            )
        last = getattr(src, "last_source_used", "")
        primary = getattr(src, "primary_name", "")
        return JSONResponse(
            {
                "rpc_ok": bool(last == primary) if primary else None,
                "fallback_count": int(getattr(src, "fallback_count", 0)),
                "last_source_used": last,
                "primary_name": primary,
            }
        )
    return await call_next(request)
