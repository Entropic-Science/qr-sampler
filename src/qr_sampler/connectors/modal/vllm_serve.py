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
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

_DEFAULT_MAX_MODEL_LEN = 65536
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.90

# Env vars whose VALUES are safe to disclose in the secret-diag log line.
# Anything not in this set is reported as ``set (N chars)`` only.
#
# Why an allow-list (not a deny-list): a deny-list defaults to "leak" when
# we add a new env var and forget to classify it. The allow-list defaults
# to "redact" — the worst case is one extra deploy cycle to add a benign
# var here, not a credential leaking into ``modal app logs``.
#
# How the K-2 / K-3 fix lands: the earlier ``set (6 chars)`` format hid
# the fact that ``QR_ENTROPY_SOURCE_TYPE`` was literally the string
# ``"system"`` (6 chars, matching the per-Dockerfile default of
# ``"quantum_grpc"`` purely by coincidence on character count). Putting
# the value inline turns a multi-hour investigation into a one-line read.
_SECRET_DIAG_VALUE_DISCLOSURE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "QR_ENTROPY_SOURCE_TYPE",
        "QR_PREINIT_ENTROPY_SOURCES",
        "QR_FALLBACK_MODE",
        "QR_SAMPLE_COUNT",
        "QR_GRPC_SERVER_ADDRESS",
        "QR_GRPC_MODE",
        "QR_GRPC_METHOD_PATH",
        "QR_GRPC_API_KEY_HEADER",
        "QR_MAX_MODEL_LEN",
        "QR_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_GPU_MEMORY_UTILIZATION",
        "QRNG_TUNNEL_HOSTNAME",
        "ALLOW_UNAUTHENTICATED_INFERENCE",
    }
)

# Env vars expected at vLLM-container runtime, grouped by purpose. The
# diagnostic helper below logs presence/absence; values are disclosed for
# names in ``_SECRET_DIAG_VALUE_DISCLOSURE_ALLOWLIST`` and redacted
# elsewhere (service tokens, API keys, CF Access credentials).
_SECRET_DIAG_GROUPS: dict[str, tuple[str, ...]] = {
    # qr-sampler runtime configuration (set by the qr-sampler-prod Secret
    # OR by the Dockerfile's ENV layer; see Dockerfile.vllm).
    "qr_sampler_runtime": (
        "QR_ENTROPY_SOURCE_TYPE",
        "QR_PREINIT_ENTROPY_SOURCES",
        "QR_FALLBACK_MODE",
        "QR_SAMPLE_COUNT",
        "QR_GRPC_SERVER_ADDRESS",
        "QR_GRPC_MODE",
        "QR_GRPC_METHOD_PATH",
        "QR_GRPC_API_KEY_HEADER",
    ),
    # QRNG cloudflared sidecar (qr-sampler-prod Secret). All three are
    # REQUIRED for the sidecar to start; missing any one triggers the
    # soft-fail path and the container serves with system entropy.
    "qrng_cloudflared": (
        "QRNG_TUNNEL_HOSTNAME",
        "CF_ACCESS_CLIENT_ID",
        "CF_ACCESS_CLIENT_SECRET",
        "QRNG_API_KEY",
    ),
    # vLLM tunables (qr-sampler-prod Secret, with both QR_* and VLLM_*
    # spellings honoured during the rename window).
    "vllm_tunables": (
        "QR_MAX_MODEL_LEN",
        "QR_GPU_MEMORY_UTILIZATION",
        "VLLM_MAX_MODEL_LEN",
        "VLLM_GPU_MEMORY_UTILIZATION",
    ),
    # Bearer-token gate. Without ``SERVICE_TOKEN_SECRETS`` set, the only
    # path through ``_check_vllm_api_key`` is
    # ``ALLOW_UNAUTHENTICATED_INFERENCE=1`` (smoke-test/dev only).
    "auth_gate": (
        "SERVICE_TOKEN_SECRETS",
        "ALLOW_UNAUTHENTICATED_INFERENCE",
    ),
    # HuggingFace token (huggingface-secret), needed by the weights
    # downloader and any HF-hosted dynamic config fetches at runtime.
    "huggingface": (
        "HUGGING_FACE_HUB_TOKEN",
        "HF_TOKEN",
    ),
}


def _emit_modal_secret_diag(served_model_name: str) -> None:
    """Log presence/absence of every expected env var, grouped by purpose.

    For names in ``_SECRET_DIAG_VALUE_DISCLOSURE_ALLOWLIST``, the actual
    value is included as ``set: <value>``. For every other name (service
    tokens, API keys, CF Access credentials, HF tokens) the status stays
    redacted as ``set (N chars)``. This balances the K-2 investigation
    cost (operator could not tell ``"system"`` from ``"quantum_grpc"``
    when both showed as ``set (...)``) against the leak risk of dumping
    every secret env var verbatim.

    Emits one structured event per group so a grep over modal logs returns
    a focused diagnosis: e.g.
    ``modal app logs qr-llm-chat | grep modal.secret_diag.qrng_cloudflared``.
    Also emits one aggregated event (``modal.secret_diag.summary``) listing
    every group's missing-var names so an operator sees the full to-fill
    list in a single line.
    """
    from obs.logging import get_logger

    log = get_logger(f"qr_sampler.modal.vllm_serve.{served_model_name}.secrets")

    aggregate_missing: dict[str, list[str]] = {}
    for group_name, var_names in _SECRET_DIAG_GROUPS.items():
        statuses: dict[str, str] = {}
        missing: list[str] = []
        for var in var_names:
            val = os.environ.get(var)
            if val is None:
                statuses[var] = "missing"
                missing.append(var)
            elif val.strip() == "":
                statuses[var] = "empty"
                missing.append(var)
            elif var in _SECRET_DIAG_VALUE_DISCLOSURE_ALLOWLIST:
                statuses[var] = f"set: {val}"
            else:
                statuses[var] = f"set ({len(val)} chars)"
        if missing:
            aggregate_missing[group_name] = missing
        log.info(
            "secret group %s: %d of %d set",
            group_name,
            len(var_names) - len(missing),
            len(var_names),
            extra={
                "event": f"modal.secret_diag.{group_name}",
                "served_model_name": served_model_name,
                "group": group_name,
                "statuses": statuses,
                "missing": missing,
            },
        )
    log.info(
        "modal secret diagnostics complete (%d groups checked, %d with gaps)",
        len(_SECRET_DIAG_GROUPS),
        len(aggregate_missing),
        extra={
            "event": "modal.secret_diag.summary",
            "served_model_name": served_model_name,
            "groups_with_missing": aggregate_missing,
            "any_missing": bool(aggregate_missing),
        },
    )


def _install_mm_probe_skip_patch() -> None:
    """Force ``mm_config.skip_mm_profiling=True`` before vLLM V1's profile_run.

    vLLM V1's ``profile_run`` runs an MM dummy probe unconditionally for any
    HF model whose ``architectures`` ends in ``*ForConditionalGeneration``,
    even when no MM input will ever flow. The probe crashes for Qwen3.6-27B
    (and Qwen3.5-9B, and several other "text-mostly" models whose HF config
    carries a populated vision_config) inside
    ``transformers.processing_utils.get_text_with_replacements`` with
    ``StopIteration``.

    vLLM 0.17.0 has a supported escape hatch at gpu_model_runner.py:5226 —
    if ``mm_config.skip_mm_profiling`` is True, the entire MM section
    short-circuits. There is no public knob to set this from
    ``AsyncEngineArgs`` in 0.17.0, so we monkey-patch ``profile_run`` to
    flip the flag at entry.

    Risk surface: if vLLM upgrades rename ``profile_run`` or remove the
    ``skip_mm_profiling`` flag, the patch silently no-ops AND the
    ``vllm.mm.probe_attempted`` event stops firing. Absence of that event
    in any cold-start log = patch lost its hook → re-read
    ``gpu_model_runner.py`` and pick a new patch point.

    Idempotent: re-entering this helper (e.g. inside a hot-reload) detects
    its own marker on the patched method and returns early instead of
    stacking patches.
    """
    from obs.events import VLLM_MM_PROBE_ATTEMPTED, VLLM_MM_PROBE_SKIPPED
    from obs.logging import get_logger

    log = get_logger("qr_sampler.modal.vllm_serve.mm_probe_patch")

    try:
        import vllm.v1.worker.gpu_model_runner as _gmr
    except ImportError:
        log.warning("vllm.v1.worker.gpu_model_runner not importable; MM-probe patch skipped")
        return

    if getattr(_gmr.GPUModelRunner.profile_run, "_qr_patched", False):
        return

    _orig = _gmr.GPUModelRunner.profile_run

    def _patched(self):  # type: ignore[no-untyped-def]
        mc = getattr(self.model_config, "multimodal_config", None)
        log.info(
            "MM probe attempted (has_multimodal_config=%s)",
            mc is not None,
            extra={
                "event": VLLM_MM_PROBE_ATTEMPTED,
                "has_multimodal_config": mc is not None,
            },
        )
        if mc is not None:
            mc.skip_mm_profiling = True
            log.info(
                "MM probe SKIPPED via skip_mm_profiling=True (text-only workload)",
                extra={"event": VLLM_MM_PROBE_SKIPPED},
            )
        return _orig(self)

    _patched._qr_patched = True  # type: ignore[attr-defined]
    _gmr.GPUModelRunner.profile_run = _patched


async def build_engine(
    served_model_name: str,
    *,
    hf_repo_id: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Any:
    """Construct one V1 ``AsyncLLM`` engine serving the configured model.

    Per the official vLLM Qwen3.6-27B recipe (recipes.vllm.ai/Qwen/
    Qwen3.6-27B), the model "works out of the box" with no special MM
    flags — multimodality is enabled by default and OWUI can send
    image attachments via ``/v1/chat/completions``.

    Cold-start crash history (2026-05-20):
      * Iteration 1 (no flags) — crashed in
        ``transformers.processing_utils.get_text_with_replacements``
        with ``StopIteration``. Root cause: the pinned transformers
        commit ``52b82b2`` had a broken Qwen3VLProcessor dummy
        generator (text without paired images).
      * Iteration 2 (``limit_mm_per_prompt={"image":0,...}``) — would
        have suppressed the MM probe but contradicts the user's intent
        to enable full MM. Not pursued.
      * Iteration 3 (current) — transformers bumped to v5.5.4 in
        Dockerfile.vllm; full MM enabled; ``_install_mm_probe_skip_patch``
        kept as defensive fallback.

    The ``vllm.hf.config_probe`` event below records the HF
    ``architectures`` + ``has_vision_config`` BEFORE engine construction
    so a future MM-routing failure surfaces with actionable diagnostics
    in one event rather than three deploy iterations.

    Keep heavy imports inside the function so the module is importable at
    snapshot-build time without vLLM present (smoke-test ergonomics).
    """
    import time
    import traceback as _tb

    from obs.events import (
        VLLM_ENGINE_ARGS_RESOLVED,
        VLLM_ENGINE_BUILD_FAILED,
        VLLM_ENGINE_BUILT,
        VLLM_HF_CONFIG_PROBE,
        VLLM_MODEL_LOAD_DONE,
        VLLM_MODEL_LOAD_START,
    )
    from obs.logging import get_logger
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import (
        AsyncLLMEngine,  # noqa: F401  -- kept for type-checking imports
    )

    # vLLM 0.17.0 uses V1 by default. The V1 OpenAI server's
    # init_app_state expects an ``AsyncLLM`` (vllm.v1.engine.async_llm),
    # NOT the V0 ``AsyncLLMEngine``. Passing V0 here causes
    # ``/health`` to return 503 (engine_client.is_dead reports True
    # because V1's EngineClient ABC isn't satisfied), and every
    # /v1/chat/completions request gets a spurious "EngineCore
    # encountered an issue" 500 with no traceback because the chat
    # serving layer raises ``EngineDeadError`` at request start
    # without ever invoking the engine. Use AsyncLLM for V1.
    from vllm.v1.engine.async_llm import AsyncLLM

    log = get_logger(f"qr_sampler.modal.vllm_serve.{served_model_name}")

    # HF AutoConfig probe — the smoking-gun check. If ``architectures``
    # includes a ``*ForConditionalGeneration`` entry AND ``vision_config``
    # is populated, vLLM will route through the VL model class and run
    # the V1 ``profile_run`` MM dummy probe regardless of
    # ``limit_mm_per_prompt``. Recording that here turns a 3-iteration
    # cold-start dig into a 1-event diagnosis.
    try:
        from transformers import AutoConfig  # type: ignore[import-untyped]

        hf_cfg = AutoConfig.from_pretrained(hf_repo_id, trust_remote_code=False)
        architectures = list(getattr(hf_cfg, "architectures", []) or [])
        vc = getattr(hf_cfg, "vision_config", None)
        has_vision = vc is not None
        log.info(
            "HF AutoConfig probe for %s: model_type=%s architectures=%s has_vision_config=%s",
            hf_repo_id,
            getattr(hf_cfg, "model_type", "?"),
            architectures,
            has_vision,
            extra={
                "event": VLLM_HF_CONFIG_PROBE,
                "hf_repo_id": hf_repo_id,
                "model_type": getattr(hf_cfg, "model_type", None),
                "architectures": architectures,
                "has_vision_config": has_vision,
                "max_position_embeddings": getattr(hf_cfg, "max_position_embeddings", None),
                "probe_error": None,
            },
        )
    except Exception as err:
        log.warning(
            "HF AutoConfig probe failed for %s: %s: %s",
            hf_repo_id,
            type(err).__name__,
            err,
            extra={
                "event": VLLM_HF_CONFIG_PROBE,
                "hf_repo_id": hf_repo_id,
                "model_type": None,
                "architectures": None,
                "has_vision_config": None,
                "max_position_embeddings": None,
                "probe_error": f"{type(err).__name__}: {err}",
            },
        )

    # Install the MM-probe skip patch defensively (it costs nothing if it
    # never fires). The actual cold-start fix is the transformers version
    # bump in Dockerfile.vllm — older transformers commits crashed
    # ``Qwen3VLProcessor`` when applying the dummy MM input during engine
    # init. With transformers @ v5.5.4 the dummy generator emits properly
    # paired (text, images) data and the processor accepts it. Full MM
    # is enabled by default; OWUI sends image attachments through to
    # ``/v1/chat/completions`` and vLLM routes them to the vision encoder.
    _install_mm_probe_skip_patch()

    engine_args = AsyncEngineArgs(
        model=hf_repo_id,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=False,
        # qr-sampler's logits processor is auto-discovered via the
        # ``vllm.logits_processors`` entry point. No manual wiring needed.
    )
    log.info(
        "AsyncEngineArgs resolved for %s (repo=%s, max_model_len=%d, "
        "gpu_memory_utilization=%.2f, enforce_eager=False)",
        served_model_name,
        hf_repo_id,
        max_model_len,
        gpu_memory_utilization,
        extra={
            "event": VLLM_ENGINE_ARGS_RESOLVED,
            "served_model_name": served_model_name,
            "hf_repo_id": hf_repo_id,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": False,
            "dtype": None,
            "kv_cache_dtype": None,
            "max_num_batched_tokens": None,
            "limit_mm_per_prompt": None,
        },
    )

    log.info(
        "vLLM engine load starting for %s",
        served_model_name,
        extra={
            "event": VLLM_MODEL_LOAD_START,
            "served_model_name": served_model_name,
            "hf_repo_id": hf_repo_id,
        },
    )
    t0 = time.perf_counter()
    try:
        # vllm 0.17.0 V1: AsyncLLM is the V1 engine. Its
        # ``from_engine_args`` accepts ``AsyncEngineArgs`` and
        # returns an ``AsyncLLM`` that init_app_state will accept
        # as the engine_client for V1's chat/completion handlers.
        #
        # Tried: ``usage_context=UsageContext.OPENAI_API_SERVER`` here —
        # caused engine to hang in profile_run with the new compile range
        # (1, 8192). Reverted to default usage_context. The engine builds
        # and serves /v1/models, but /v1/chat/completions still returns
        # 500 with "EngineCore encountered an issue" — root cause is in
        # the V1 EngineCore↔main-process IPC (engine_client.is_dead
        # reports True even with subprocess clearly alive); needs deeper
        # vLLM-level investigation. See goofy-cooking-badger.md.
        engine = AsyncLLM.from_engine_args(engine_args)
    except BaseException as err:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        tb_text = "".join(_tb.format_exception(type(err), err, err.__traceback__))
        tb_tail = "\n".join(tb_text.splitlines()[-30:])
        log.error(
            "vLLM engine build FAILED for %s after %.0fms: %s: %s",
            served_model_name,
            duration_ms,
            type(err).__name__,
            err,
            extra={
                "event": VLLM_ENGINE_BUILD_FAILED,
                "served_model_name": served_model_name,
                "hf_repo_id": hf_repo_id,
                "duration_ms": duration_ms,
                "error_type": type(err).__name__,
                "error_msg": str(err),
                "traceback_tail": tb_tail,
            },
        )
        raise

    duration_ms = (time.perf_counter() - t0) * 1000.0

    # Best-effort engine metadata for the BUILT event. A failure here is
    # diagnostic-only -- the engine is already constructed and serving.
    try:
        model_cfg = await engine.get_model_config()
        model_dtype = str(getattr(model_cfg, "dtype", "?"))
        model_max_len = int(getattr(model_cfg, "max_model_len", max_model_len))
    except Exception as err:
        model_dtype = f"<unavailable: {type(err).__name__}>"
        model_max_len = max_model_len

    log.info(
        "vLLM model load done for %s in %.0fms",
        served_model_name,
        duration_ms,
        extra={
            "event": VLLM_MODEL_LOAD_DONE,
            "served_model_name": served_model_name,
            "duration_ms": duration_ms,
        },
    )
    log.info(
        "vLLM engine BUILT for %s in %.0fms (dtype=%s, max_model_len=%d)",
        served_model_name,
        duration_ms,
        model_dtype,
        model_max_len,
        extra={
            "event": VLLM_ENGINE_BUILT,
            "served_model_name": served_model_name,
            "hf_repo_id": hf_repo_id,
            "duration_ms": duration_ms,
            "model_dtype": model_dtype,
            "model_max_len": model_max_len,
        },
    )

    # Commit the vllm-cache volume so torch.compile / AOT compile
    # artefacts written during the build above are visible to the next
    # cold-start. Modal volumes auto-commit on container shutdown, but
    # we want the second cold-start to read the cache even if the first
    # container hasn't shut down yet (parallel container churn during
    # active development). Soft-fails — a commit failure does not
    # invalidate the engine that's already built.
    try:
        import modal  # type: ignore[import-untyped]

        vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=False)
        vllm_cache.commit()
        log.info(
            "vllm-cache volume committed",
            extra={
                "event": "modal.volume.committed",
                "volume": "vllm-cache",
                "served_model_name": served_model_name,
            },
        )
    except Exception as exc:
        log.warning(
            "vllm-cache volume commit failed: %s: %s",
            type(exc).__name__,
            exc,
            extra={
                "event": "modal.volume.commit_failed",
                "volume": "vllm-cache",
                "served_model_name": served_model_name,
                "error_type": type(exc).__name__,
                "error_msg": str(exc),
            },
        )

    return engine


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


async def build_app(served_model_name: str, hf_repo_id: str, engine: Any) -> FastAPI:
    """ASGI app for a single model.

    Uses vLLM 0.17.0's official ``build_app(args, supported_tasks)`` +
    ``init_app_state(engine_client, app.state, args, supported_tasks)``
    pattern so we inherit every OpenAI-compatible route vLLM exposes
    (``/v1/models``, ``/v1/chat/completions``, ``/v1/completions``,
    ``/v1/embeddings``, …) without hand-rolling the serving classes.
    Adds two custom layers on top:

    1. ``/health/entropy`` — QRNG sidecar reachability probe (no auth).
    2. Bearer-token middleware that gates ``/v1/*`` paths against the
       rolling ``SERVICE_TOKEN_SECRETS`` vector (skipped for
       ``/v1/models`` so OWUI can enumerate without a token).

    Rebuild rationale (2026-05-20 iteration 5): vLLM 0.17.0 refactored
    the OpenAI serving classes (``OpenAIServingChat``/``Completion``
    constructors now require ``OpenAIServingModels`` + dozen other
    kwargs; protocol module moved to ``chat_completion``/``completion``
    subpackages). Reusing vLLM's ``build_app`` is more robust than
    porting the constructor calls and keeps us in lockstep with future
    vLLM releases.
    """
    from argparse import Namespace as _Namespace  # noqa: F401 -- typing only

    from vllm.entrypoints.openai.api_server import (
        build_app as _vllm_build_app,
    )
    from vllm.entrypoints.openai.api_server import (
        init_app_state as _vllm_init_app_state,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vllm-qr")
    parser = make_arg_parser(parser)
    args = parser.parse_args(
        [
            "--model",
            hf_repo_id,
            "--served-model-name",
            served_model_name,
        ]
    )

    supported_tasks: tuple[Any, ...] = ("generate",)
    app = _vllm_build_app(args, supported_tasks=supported_tasks)

    # Install our bearer-token gate as a PURE ASGI middleware (not
    # BaseHTTPMiddleware). Starlette's BaseHTTPMiddleware has a long-
    # standing bug with downstream StreamingResponse (it consumes the
    # body iterator into memory and breaks SSE/chunked transport),
    # AND it propagates exceptions from downstream as 500s with
    # "EngineCore encountered an issue"-style spurious messages because
    # the response is already started by the time the wrapper sees the
    # error. vLLM 0.17.0's /v1/chat/completions returns a StreamingResponse
    # even with stream=false — that hits the bug and surfaces as a
    # silent 500 with no traceback in vLLM's stderr. (See
    # https://github.com/encode/starlette/issues/1438 and the linked
    # https://github.com/vllm-project/vllm/issues/2683 family.)
    #
    # Pure ASGI middleware operates on (scope, receive, send) tuples
    # directly, so it never touches the response body. For our use
    # (header check + pass-through OR short-circuit JSON) this is a
    # 15-line implementation with no streaming surface to break.
    from starlette.middleware import Middleware
    from starlette.types import (  # noqa: TC002 -- local-scope import beside runtime Middleware import
        ASGIApp,
        Receive,
        Scope,
        Send,
    )

    class _QrBearerTokenGate:
        def __init__(self, app: ASGIApp) -> None:
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self.app(scope, receive, send)
                return
            path = scope.get("path", "")
            needs_auth = path.startswith("/v1/") and path != "/v1/models"
            if needs_auth:
                # Build a transient Request just to use the existing
                # _check_vllm_api_key helper (which reads
                # request.headers["authorization"]).
                req = Request(scope, receive=receive)
                try:
                    _check_vllm_api_key(req)
                except HTTPException as exc:
                    response = JSONResponse(
                        status_code=exc.status_code,
                        content={"detail": exc.detail},
                    )
                    await response(scope, receive, send)
                    return
            await self.app(scope, receive, send)

    app.user_middleware.insert(0, Middleware(_QrBearerTokenGate))

    # vLLM's normal flow runs ``init_app_state`` *before* serving begins —
    # the lifespan handler (server_utils.py:lifespan) then reads
    # ``app.state.log_stats`` to decide whether to spin up the metrics
    # logger. If we deferred ``init_app_state`` to ``router.on_startup`` it
    # would fire AFTER the lifespan startup phase, so the lifespan crashes
    # with ``AttributeError: 'State' object has no attribute 'log_stats'``
    # (observed 2026-05-20 iteration H during cold-start, after engine +
    # cloudflared had already initialised successfully).
    #
    # ``build_app`` is async (callers must ``await`` it) so we can
    # ``await`` ``init_app_state`` on the caller's running loop — that
    # loop is the same one the engine was built on inside
    # ``build_dispatcher_for``, so no loop-mismatch concerns. (Earlier
    # attempts used ``asyncio.run`` here, which raises
    # ``RuntimeError: asyncio.run() cannot be called from a running event
    # loop`` because Modal's ``@modal.enter`` body wraps the call in
    # ``asyncio.run`` already — see app.py:load.)
    await _vllm_init_app_state(engine, app.state, args, supported_tasks)

    async def _qr_health() -> dict[str, Any]:
        return {"status": "ok", "model": served_model_name}

    app.router.add_api_route("/health", _qr_health, methods=["GET"])

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
        except Exception as err:
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

    # /v1/models, /v1/chat/completions, /v1/completions are provided by
    # vLLM's build_app — no hand-rolled routes needed here. The auth
    # middleware above gates them via SERVICE_TOKEN_SECRETS.

    return app


async def build_dispatcher_for(served_model_name: str, hf_repo_id: str) -> FastAPI:
    """Build a single-model engine + ASGI app. Called by each per-model
    @app.cls's ``load()`` method inside ``@modal.enter(snap=True)``.

    Reads ``VLLM_MAX_MODEL_LEN`` and ``VLLM_GPU_MEMORY_UTILIZATION`` from env
    (typically populated by the ``qr-sampler-prod`` Modal Secret).
    """
    from obs.events import VLLM_DISPATCHER_READY
    from obs.logging import get_logger

    log = get_logger(f"qr_sampler.modal.vllm_serve.{served_model_name}")

    # Emit secret diagnostics FIRST so an operator can grep one cold-start
    # for ``modal.secret_diag.summary`` and immediately see which Modal
    # Secret slot is mis-populated — before the engine spends 5 min loading
    # weights only to surface a cloudflared-config-missing warning at the
    # end. Values for allow-listed names are disclosed (see
    # ``_SECRET_DIAG_VALUE_DISCLOSURE_ALLOWLIST``); credentials stay
    # redacted.
    _emit_modal_secret_diag(served_model_name)

    # Promote legacy ``VLLM_*`` spellings to their ``QR_*`` equivalents and
    # POP the originals from ``os.environ`` BEFORE any vLLM import. vLLM's
    # envs.py:1710 scans os.environ for VLLM_* prefixes at module import
    # time and emits a WARNING for every name it does not recognise. Both
    # ``VLLM_MAX_MODEL_LEN`` and ``VLLM_GPU_MEMORY_UTILIZATION`` are
    # qr-sampler-app-level, NOT native vLLM tunables, so vLLM rightly
    # flags them — but the warning is pure noise once we have the QR_*
    # equivalent. Popping the originals AFTER promoting the value gives
    # us the best of both worlds: the value still drives the engine, and
    # the cold-start log loses the K-4 noise.
    #
    # If the user later sets a GENUINE ``VLLM_*`` tunable (e.g. one of
    # vLLM's own envs like ``VLLM_USE_V1``), this loop ignores it because
    # those names are not in the explicit list below — only the two we
    # know we own pass through.
    for legacy in ("VLLM_MAX_MODEL_LEN", "VLLM_GPU_MEMORY_UTILIZATION"):
        val = os.environ.pop(legacy, None)
        if val is not None:
            qr_name = "QR_" + legacy[len("VLLM_") :]
            # Explicit `in` check — `os.environ.setdefault(...) is val` is
            # NOT a reliable "did we insert" signal because os._Environ
            # re-decodes the string on every read, so the returned object
            # is never identity-equal to the value we passed in (even on
            # the path where setdefault did insert). Earlier the misread
            # was harmless (the value still landed correctly) but the
            # ``qr_was_already_set`` log field was inverted.
            qr_was_already_set = qr_name in os.environ
            if not qr_was_already_set:
                os.environ[qr_name] = val
            log.info(
                "promoted legacy env var %s -> %s (qr_was_already_set=%s)",
                legacy,
                qr_name,
                qr_was_already_set,
                extra={
                    "event": "modal.env.promoted",
                    "legacy": legacy,
                    "qr": qr_name,
                    "qr_was_already_set": qr_was_already_set,
                },
            )

    # Sanity-check QR_GRPC_SERVER_ADDRESS. The cloudflared sidecar lives
    # in this same container and binds to loopback by design (see
    # cloudflared_sidecar.DEFAULT_BIND_HOST=127.0.0.1). A non-loopback
    # value here means the gRPC client will dial off-container and the
    # sidecar's tunnel will never carry traffic — which usually surfaces
    # as the TCP pre-probe failing every fetch and the request silently
    # falling back to system entropy. This is the K-2-cluster failure
    # mode the operator already saw on 2026-05-20.
    grpc_addr = os.environ.get("QR_GRPC_SERVER_ADDRESS", "")
    grpc_host = grpc_addr.partition(":")[0]
    if grpc_host and grpc_host not in {"127.0.0.1", "localhost", "[::1]", "::1"}:
        log.warning(
            "QR_GRPC_SERVER_ADDRESS=%r is NOT loopback. The cloudflared "
            "sidecar in this container binds to 127.0.0.1 only — gRPC "
            "fetches will fail the TCP pre-probe and fall back to system "
            "entropy. Update the qr-sampler-prod Modal Secret to "
            "QR_GRPC_SERVER_ADDRESS=127.0.0.1:50051.",
            grpc_addr,
            extra={
                "event": "modal.grpc_address.non_loopback",
                "served_model_name": served_model_name,
                "configured_address": grpc_addr,
                "configured_host": grpc_host,
            },
        )

    # Prefer the QR_*-prefixed env vars. Fall back to VLLM_* defensively
    # (the sweep above should have pre-promoted any VLLM_* value, but
    # keeping the chain here means a partial rollout of this fix still
    # finds the value).
    qr_mml_raw = os.environ.get("QR_MAX_MODEL_LEN")
    vllm_mml_raw = os.environ.get("VLLM_MAX_MODEL_LEN")
    max_model_len = int(qr_mml_raw or vllm_mml_raw or _DEFAULT_MAX_MODEL_LEN)
    max_model_len_source = (
        "QR_MAX_MODEL_LEN" if qr_mml_raw else "VLLM_MAX_MODEL_LEN" if vllm_mml_raw else "default"
    )

    qr_gmu_raw = os.environ.get("QR_GPU_MEMORY_UTILIZATION")
    vllm_gmu_raw = os.environ.get("VLLM_GPU_MEMORY_UTILIZATION")
    gpu_mem = float(qr_gmu_raw or vllm_gmu_raw or _DEFAULT_GPU_MEMORY_UTILIZATION)
    gpu_mem_source = (
        "QR_GPU_MEMORY_UTILIZATION"
        if qr_gmu_raw
        else "VLLM_GPU_MEMORY_UTILIZATION"
        if vllm_gmu_raw
        else "default"
    )

    log.info(
        "Building vLLM engine: served_model_name=%s repo_id=%s "
        "max_model_len=%d (source=%s) gpu_memory_utilization=%.2f (source=%s)",
        served_model_name,
        hf_repo_id,
        max_model_len,
        max_model_len_source,
        gpu_mem,
        gpu_mem_source,
        extra={
            "event": "modal.vllm.tunables_resolved",
            "served_model_name": served_model_name,
            "max_model_len": max_model_len,
            "max_model_len_source": max_model_len_source,
            "gpu_memory_utilization": gpu_mem,
            "gpu_memory_utilization_source": gpu_mem_source,
        },
    )

    engine = await build_engine(
        served_model_name,
        hf_repo_id=hf_repo_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
    )
    asgi_app = await build_app(served_model_name, hf_repo_id, engine)

    log.info(
        "vLLM dispatcher ready for %s",
        served_model_name,
        extra={
            "event": VLLM_DISPATCHER_READY,
            "served_model_name": served_model_name,
            "hf_repo_id": hf_repo_id,
        },
    )
    return asgi_app
