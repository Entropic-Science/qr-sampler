"""Entrypoint for the Modal-hosted vLLM container (B200, dual-model, qr-sampler).

This module is imported by ``VllmQr.load`` inside ``@modal.enter(snap=True)``.
It builds two ``AsyncLLMEngine`` siblings in the same process — one for each
served model — and exposes an ASGI app that dispatches OpenAI-compatible
requests to the matching engine on the request body's ``model`` field.

Snapshot integrity invariants
-----------------------------
All construction performed by ``build_engines()`` must obey the rules from
``spec.md`` §5.5 / ``CROSS-REPO-INTEGRATION.md`` §2.4:

1. **No live gRPC channel captured in the snapshot.** qr-sampler's
   ``quantum_grpc`` source uses lazy channel creation, so constructing both
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
Each engine's qr-sampler ``VLLMAdapter`` (auto-installed via vLLM's
``vllm.logits_processors`` entry point) pre-initialises both ``quantum_grpc``
and ``system`` pipelines (from ``QR_PREINIT_ENTROPY_SOURCES``). The OWUI
comparison Pipe issues per-request ``qr_entropy_source_type`` overrides via
``extra_body``; the adapter's ``update_state()`` selects the matching
pipeline. The just-in-time invariant is preserved — entropy is still fetched
*after* logits are computed.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    # vllm is only available inside the GPU container at runtime.
    from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger("qr_sampler.modal.vllm_serve")


_DEFAULT_MODELS = "gemma-4-31b-reasoning,qwen-3.6-27b-reasoning"
_DEFAULT_MAX_MODEL_LEN = 65536
_DEFAULT_GPU_MEMORY_UTILIZATION_PER_ENGINE = 0.45

# Hugging Face repo ids the served-model-names resolve to. Updated alongside
# the at-deploy-time fallback chain documented in deployments/modal/README.md.
_HF_REPO_FOR_MODEL: dict[str, str] = {
    "gemma-4-31b-reasoning": "google/gemma-4-31b-reasoning",
    "qwen-3.6-27b-reasoning": "Qwen/Qwen-3.6-27B-Reasoning",
}


def _resolve_served_models() -> list[str]:
    raw = os.environ.get("VLLM_MODELS", _DEFAULT_MODELS)
    names = [m.strip() for m in raw.split(",") if m.strip()]
    if not names:
        raise RuntimeError(f"VLLM_MODELS must list at least one served-model-name (got {raw!r})")
    return names


def _resolve_default_model(models: list[str]) -> str:
    default = os.environ.get("VLLM_DEFAULT_MODEL", models[0])
    if default not in models:
        raise RuntimeError(f"VLLM_DEFAULT_MODEL={default!r} not in VLLM_MODELS={models!r}")
    return default


async def _build_engine(
    served_model_name: str,
    *,
    hf_repo_id: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> AsyncLLMEngine:
    """Construct one ``AsyncLLMEngine`` configured for FP8 weights + FP8 KV.

    Keep heavy imports inside the function so the module is importable at
    snapshot-build time without vLLM present (smoke-test ergonomics).
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=hf_repo_id,
        served_model_name=served_model_name,
        quantization="fp8",
        kv_cache_dtype="fp8",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        # qr-sampler's logits processor is auto-discovered via the
        # vllm.logits_processors entry point. No manual wiring needed.
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def build_engines() -> dict[str, AsyncLLMEngine]:
    """Build all configured engines and return a ``served-model-name`` map.

    Called from ``VllmQr.load`` inside ``@modal.enter(snap=True)``. Each
    engine pre-loads its weights (from the region-local ``llm-weights``
    Volume) and warms its qr-sampler ``VLLMAdapter``, which in turn
    pre-initialises both entropy pipelines per
    ``QR_PREINIT_ENTROPY_SOURCES=quantum_grpc,system``. The snapshot
    captures the post-init state of all four ``(engine, source)`` pipelines.
    """
    models = _resolve_served_models()
    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", _DEFAULT_MAX_MODEL_LEN))
    gpu_mem = float(
        os.environ.get(
            "VLLM_GPU_MEMORY_UTILIZATION_PER_ENGINE",
            _DEFAULT_GPU_MEMORY_UTILIZATION_PER_ENGINE,
        )
    )

    engines: dict[str, AsyncLLMEngine] = {}
    for name in models:
        repo_id = _HF_REPO_FOR_MODEL.get(name)
        if repo_id is None:
            raise RuntimeError(
                f"No Hugging Face repo mapping for served-model-name={name!r}. "
                f"Update _HF_REPO_FOR_MODEL in vllm_serve.py to include it."
            )
        logger.info(
            "Building vLLM engine: served_model_name=%s repo_id=%s "
            "max_model_len=%d gpu_memory_utilization=%.2f",
            name,
            repo_id,
            max_model_len,
            gpu_mem,
        )
        engines[name] = await _build_engine(
            name,
            hf_repo_id=repo_id,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
        )
    return engines


def _check_vllm_api_key(request: Request) -> None:
    """Bearer-token gate. Only ``owui_edge`` (internal) knows the key."""
    expected = os.environ.get("VLLM_API_KEY", "")
    if not expected:
        return  # explicit opt-out (smoke-test only); production sets it
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if header[len("Bearer ") :] != expected:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


def build_app(engines: dict[str, AsyncLLMEngine], default_model: str) -> FastAPI:
    """ASGI dispatcher in front of all configured engines.

    Routes ``/v1/models``, ``/v1/chat/completions``, and ``/v1/completions``
    on the request body's ``model`` field to the matching engine's vLLM
    OpenAI-protocol handler. Other vLLM OpenAI routes (``/v1/embeddings``
    etc.) are not currently surfaced — add here if a future feature needs
    them.
    """
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        CompletionRequest,
    )
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

    serving_chat: dict[str, OpenAIServingChat] = {}
    serving_completion: dict[str, OpenAIServingCompletion] = {}

    async def _init_serving() -> None:
        for name, engine in engines.items():
            model_config = await engine.get_model_config()
            serving_chat[name] = OpenAIServingChat(
                engine_client=engine,
                model_config=model_config,
                served_model_names=[name],
                response_role="assistant",
                lora_modules=None,
                prompt_adapters=None,
                request_logger=None,
                chat_template=None,
            )
            serving_completion[name] = OpenAIServingCompletion(
                engine_client=engine,
                model_config=model_config,
                served_model_names=[name],
                lora_modules=None,
                prompt_adapters=None,
                request_logger=None,
            )

    app = FastAPI(title="vllm-qr (Modal dispatcher)")

    @app.on_event("startup")
    async def _on_startup() -> None:
        await _init_serving()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "models": list(engines.keys())}

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        _check_vllm_api_key(request)
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "qr-sampler",
                    "root": name,
                    "parent": None,
                }
                for name in engines
            ],
        }

    def _route_target(body: dict[str, Any]) -> str:
        requested = body.get("model")
        if not isinstance(requested, str) or not requested:
            return default_model
        if requested not in engines:
            raise HTTPException(
                status_code=400,
                detail=(f"Unknown model {requested!r}. Available: {sorted(engines.keys())}"),
            )
        return requested

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        _check_vllm_api_key(request)
        body = await request.json()
        target = _route_target(body)
        chat_req = ChatCompletionRequest(**body)
        generator = await serving_chat[target].create_chat_completion(chat_req, request)
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
        target = _route_target(body)
        comp_req = CompletionRequest(**body)
        generator = await serving_completion[target].create_completion(comp_req, request)
        if hasattr(generator, "body_iterator"):
            return generator
        return JSONResponse(content=generator.model_dump(exclude_none=True))

    return app


async def build_dispatcher() -> FastAPI:
    """Compose ``build_engines`` + ``build_app`` for the Modal class method."""
    engines = await build_engines()
    default_model = _resolve_default_model(list(engines.keys()))
    return build_app(engines, default_model)
