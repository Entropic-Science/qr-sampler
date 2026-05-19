"""Entrypoint for the Modal-hosted vLLM container (B200, one model per container).

This module is imported by each per-model ``@app.cls`` in ``app.py`` (e.g.
``VllmQrGemma.load`` / ``VllmQrQwen.load``) inside ``@modal.enter(snap=True)``.
It builds ONE ``AsyncLLMEngine`` and exposes an ASGI app serving the
OpenAI-compatible surface for that single model.

The per-model split (one @app.cls per model, sharing the image) means each
container scales to zero independently — picking Gemma in OWUI's model
dropdown never wakes the Qwen container, and vice versa. Comparison mode
(``--qr-vs-prng`` pseudo-models) routes both columns to the SAME base
model's container, differentiated only by entropy source (see
``examples/open-webui/qr_comparison_pipe.py``).

Snapshot integrity invariants
-----------------------------
All construction performed by ``build_engine()`` must obey the rules from
``spec.md`` §5.5 / ``CROSS-REPO-INTEGRATION.md`` §2.4:

1. **No live gRPC channel captured in the snapshot.** qr-sampler's
   ``quantum_grpc`` source uses lazy channel creation, so constructing the
   entropy pipelines here only registers configuration — the gRPC client
   does not open a socket until the first per-token entropy fetch *after*
   snapshot restore.
2. **No process-relative state captured.** No ``os.getpid()``-keyed caches,
   no in-process locks. vLLM engines internally avoid these.
3. **Secrets are mounted after restore.** Env reads happen here at call
   time (inside ``@modal.enter(snap=True)``), so Modal's post-restore
   secret injection populates them correctly.

Per-request entropy switching
-----------------------------
The engine's qr-sampler ``VLLMAdapter`` (auto-installed via vLLM's
``vllm.logits_processors`` entry point) pre-initialises both ``quantum_grpc``
and ``system`` pipelines (from ``QR_PREINIT_ENTROPY_SOURCES``). The OWUI
comparison Pipe issues per-request ``qr_entropy_source_type`` overrides via
``extra_body``; the adapter's ``update_state()`` selects the matching
pipeline. The just-in-time invariant is preserved — entropy is still fetched
*after* logits are computed.
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    # vllm is only available inside the GPU container at runtime.
    from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger("qr_sampler.modal.vllm_serve")


_DEFAULT_MAX_MODEL_LEN = 65536
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.90


async def build_engine(
    served_model_name: str,
    *,
    hf_repo_id: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> AsyncLLMEngine:
    """Construct one ``AsyncLLMEngine`` at the model's native precision.

    Loads weights and KV cache at the model's default dtype (typically bf16
    for the Gemma 4 / Qwen 3.6 base repos). FP8 quantization was dropped
    when we switched to the full-precision repos `google/gemma-4-31B` +
    `Qwen/Qwen3.6-27B` — re-enable here if you swap back to an FP8-quantized
    variant.

    Keep heavy imports inside the function so the module is importable at
    snapshot-build time without vLLM present (smoke-test ergonomics).
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=hf_repo_id,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        # qr-sampler's logits processor is auto-discovered via the
        # vllm.logits_processors entry point. No manual wiring needed.
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


def _accepted_bearer_secrets() -> list[str]:
    """Return the rolling-secret vector from ``SERVICE_TOKEN_SECRETS``.

    Comma-separated. Empty entries (extra commas, surrounding whitespace) are
    dropped. The signer side (Open WebUI's ``OPENAI_API_KEY``) uses the first
    entry; this verifier accepts any entry, which is what enables prepend-then-
    redeploy rotation without lockstep.
    """
    raw = os.environ.get("SERVICE_TOKEN_SECRETS", "")
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


def _verify_bearer(token: str) -> bool:
    """Constant-time membership check against the rolling-secret vector.

    Returns ``True`` if ``token`` matches any entry in ``SERVICE_TOKEN_SECRETS``.
    Returns ``False`` for an empty token, a token absent from the vector, or
    when the vector itself is empty (no secret provisioned = closed by default).

    Compares against every entry even on early success so the branch count is
    constant per call regardless of which entry — if any — matched.
    """
    if not token:
        return False
    matched = False
    for accepted in _accepted_bearer_secrets():
        if hmac.compare_digest(token, accepted):
            matched = True
    return matched


def _check_vllm_api_key(request: Request) -> None:
    """Bearer-token gate. The signer is the OWUI deployment in front of
    this vLLM endpoint; both sides read ``SERVICE_TOKEN_SECRETS`` from the
    same rolling-secret vector (see modal_secrets.md).

    Fail-closed: when ``SERVICE_TOKEN_SECRETS`` is empty, requests are rejected
    with 503 unless ``ALLOW_UNAUTHENTICATED_INFERENCE=1`` is set as an explicit
    operator opt-in (smoke-test / local dev only). A silent open-by-default
    would expose the GPU endpoint to the public internet if the secret slot
    were ever provisioned blank.
    """
    if not _accepted_bearer_secrets():
        if os.environ.get("ALLOW_UNAUTHENTICATED_INFERENCE") == "1":
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference endpoint is not configured for authentication: set "
                "SERVICE_TOKEN_SECRETS, or set ALLOW_UNAUTHENTICATED_INFERENCE=1 "
                "to opt-in to unauthenticated access (dev/smoke only)."
            ),
        )
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not _verify_bearer(header[len("Bearer ") :]):
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def build_app(served_model_name: str, engine: AsyncLLMEngine) -> FastAPI:
    """ASGI app for a single model.

    Surfaces ``/v1/models``, ``/v1/chat/completions``, ``/v1/completions`` for
    one engine. The request body's ``model`` field is no longer used for
    routing — there's exactly one engine per container, so any value is
    accepted (vLLM still validates it matches ``served_model_name`` for echo
    in the response). Other vLLM OpenAI routes (``/v1/embeddings`` etc.) are
    not currently surfaced — add here if a future feature needs them.
    """
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

    serving_chat_holder: dict[str, OpenAIServingChat] = {}
    serving_completion_holder: dict[str, OpenAIServingCompletion] = {}

    async def _init_serving() -> None:
        model_config = await engine.get_model_config()
        serving_chat_holder["instance"] = OpenAIServingChat(
            engine_client=engine,
            model_config=model_config,
            served_model_names=[served_model_name],
            response_role="assistant",
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
            chat_template=None,
        )
        serving_completion_holder["instance"] = OpenAIServingCompletion(
            engine_client=engine,
            model_config=model_config,
            served_model_names=[served_model_name],
            lora_modules=None,
            prompt_adapters=None,
            request_logger=None,
        )

    app = FastAPI(title=f"vllm-qr ({served_model_name})")

    @app.on_event("startup")
    async def _on_startup() -> None:
        await _init_serving()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "model": served_model_name}

    @app.get("/health/entropy")
    async def health_entropy() -> dict[str, Any]:
        """Quantum-entropy source connectivity probe.

        Called by OWUI's setup-orchestrator (see qr_llm_chat.setup_orchestrator)
        to confirm the per-container cloudflared sidecar + Cipherstone gRPC
        backend are reachable. NOT auth-gated for the same reason ``/health``
        is not — operators and the OWUI status poller need to read it without
        a bearer token round-trip.

        Probe sequence:

        1. **TCP-connect** to ``QR_GRPC_SERVER_ADDRESS`` (default
           ``127.0.0.1:50051``) with a 1s deadline. Proves the cloudflared
           sidecar is listening. A failure here points at the sidecar's
           own startup (check ``cloudflared.*`` events in modal logs).
        2. **RPC ping** — a single ``GetRandomBytes(16)`` call against the
           gRPC service with a 2s deadline. Proves the full path
           container -> sidecar -> Cloudflare Access -> Cipherstone backend
           is up AND that the ``QR_GRPC_API_KEY`` is currently accepted.
           Consumes 16 entropy bytes (negligible).

        Response shape (always 200 — the probe's job is to report status,
        not to fail the request; OWUI reads the JSON fields)::

            {
                "model": "<served_model_name>",
                "address": "127.0.0.1:50051",
                "tcp_ok": true,
                "tcp_error": null,
                "rpc_ok": true,
                "rpc_error": null,
                "rpc_latency_ms": 42.3,
                "bytes_received": 16,
                "summary": "quantum_grpc reachable (rpc=42ms)"
            }
        """
        import socket
        import time as _time

        address = os.environ.get("QR_GRPC_SERVER_ADDRESS", "127.0.0.1:50051")
        host, _, port_s = address.partition(":")
        try:
            port = int(port_s)
        except ValueError:
            return {
                "model": served_model_name,
                "address": address,
                "tcp_ok": False,
                "tcp_error": f"malformed address: {address!r}",
                "rpc_ok": False,
                "rpc_error": "skipped (tcp failed)",
                "rpc_latency_ms": None,
                "bytes_received": 0,
                "summary": f"misconfigured QR_GRPC_SERVER_ADDRESS={address!r}",
            }

        tcp_ok = False
        tcp_error: str | None = None
        try:
            with socket.create_connection((host, port), timeout=1.0):
                tcp_ok = True
        except OSError as err:
            tcp_error = f"{type(err).__name__}: {err}"

        if not tcp_ok:
            return {
                "model": served_model_name,
                "address": address,
                "tcp_ok": False,
                "tcp_error": tcp_error,
                "rpc_ok": False,
                "rpc_error": "skipped (tcp failed)",
                "rpc_latency_ms": None,
                "bytes_received": 0,
                "summary": f"cloudflared sidecar unreachable at {address}",
            }

        rpc_ok = False
        rpc_error: str | None = None
        rpc_latency_ms: float | None = None
        bytes_received = 0
        try:
            from qr_sampler.config import QRSamplerConfig
            from qr_sampler.entropy.quantum import QuantumGrpcSource

            cfg = QRSamplerConfig(entropy_source_type="quantum_grpc")
            src = QuantumGrpcSource(cfg)
            try:
                t0 = _time.perf_counter()
                data = src.get_random_bytes(16)
                rpc_latency_ms = (_time.perf_counter() - t0) * 1000.0
                bytes_received = len(data)
                rpc_ok = bytes_received == 16
                if not rpc_ok:
                    rpc_error = f"short read: got {bytes_received} bytes, expected 16"
            finally:
                src.close()
        except Exception as err:  # noqa: BLE001 — probe must always return 200
            rpc_error = f"{type(err).__name__}: {err}"

        if rpc_ok:
            summary = f"quantum_grpc reachable (rpc={rpc_latency_ms:.0f}ms)"
        else:
            summary = f"gRPC probe failed: {rpc_error}"

        return {
            "model": served_model_name,
            "address": address,
            "tcp_ok": tcp_ok,
            "tcp_error": tcp_error,
            "rpc_ok": rpc_ok,
            "rpc_error": rpc_error,
            "rpc_latency_ms": rpc_latency_ms,
            "bytes_received": bytes_received,
            "summary": summary,
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        _check_vllm_api_key(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "qr-sampler",
                    "root": served_model_name,
                    "parent": None,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        _check_vllm_api_key(request)
        body = await request.json()
        # Single-engine container: ignore body["model"] for routing (vLLM
        # echoes served_model_name in the response either way). If clients
        # send a mismatched model id we still serve — OWUI's connection-per-
        # endpoint config means cross-model requests don't reach this
        # container in normal operation.
        chat_req = ChatCompletionRequest(**body)
        generator = await serving_chat_holder["instance"].create_chat_completion(
            chat_req, request
        )
        # ``create_chat_completion`` returns either a StreamingResponse or a
        # pydantic model depending on the request's ``stream`` flag — pass
        # through as-is so streaming works end-to-end.
        if hasattr(generator, "body_iterator"):
            return generator
        return JSONResponse(content=generator.model_dump(exclude_none=True))

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        _check_vllm_api_key(request)
        body = await request.json()
        comp_req = CompletionRequest(**body)
        generator = await serving_completion_holder["instance"].create_completion(
            comp_req, request
        )
        if hasattr(generator, "body_iterator"):
            return generator
        return JSONResponse(content=generator.model_dump(exclude_none=True))

    return app


async def build_dispatcher_for(
    served_model_name: str, hf_repo_id: str
) -> FastAPI:
    """Build a single-model engine + ASGI app. Called by each per-model
    @app.cls's ``load()`` method inside ``@modal.enter(snap=True)``.

    Reads ``VLLM_MAX_MODEL_LEN`` and ``VLLM_GPU_MEMORY_UTILIZATION`` from env
    (typically populated by the ``qr-sampler-prod`` Modal Secret).
    """
    max_model_len = int(
        os.environ.get("VLLM_MAX_MODEL_LEN", _DEFAULT_MAX_MODEL_LEN)
    )
    gpu_mem = float(
        os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", _DEFAULT_GPU_MEMORY_UTILIZATION)
    )

    logger.info(
        "Building vLLM engine: served_model_name=%s repo_id=%s "
        "max_model_len=%d gpu_memory_utilization=%.2f",
        served_model_name,
        hf_repo_id,
        max_model_len,
        gpu_mem,
    )

    engine = await build_engine(
        served_model_name,
        hf_repo_id=hf_repo_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
    )
    return build_app(served_model_name, engine)
