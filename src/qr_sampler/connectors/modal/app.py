"""Modal app definition — vllm-qr (H200, per-model containers).

Layout (matches spec.md §5.5 / §4.1, with the labs-cutover per-model split):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQrQwen         — @app.cls (H100:1) running Qwen/Qwen3.6-27B-FP8 alone

Each model is its own scale-to-zero @app.cls so OWUI's model picker wakes
only the requested container. Open WebUI itself is provided by the
`OWUIService` class defined below; the OWUI-specific lifecycle code
(admin bootstrap, SvelteKit base-path patch, Function bundle import)
lives in the downstream `qr-llm-chat` package, imported lazily inside
the @modal.enter hooks so this module stays usable without that
dependency installed.

The active served model is configured at the top of this file via the
``MODEL_*`` constants block. Swapping models is a four-line edit
(``MODEL_HF_REPO_ID`` / ``MODEL_SERVED_NAME`` / ``MODEL_REVISION`` /
``MODEL_GPU_MEMORY_UTILIZATION``) plus running ``download_weights`` for
the new repo. ``MODEL_SERVED_NAME`` MUST stay in lockstep with
``_QWEN_ID`` in ``qr_llm_chat/bootstrap_connections.py`` and the Pipe's
``base_models`` default.

Gemma 4 31B pause + Qwen 3.* MM-probe monkey-patch
--------------------------------------------------
1. ``VllmQrGemma`` (google/gemma-4-31B) is paused while the vLLM/
   transformers ecosystem stabilises around the gemma-4 GDN architecture.
   vLLM 0.17.0 does not register ``Gemma4ForConditionalGeneration``.
   Restore Gemma when a vLLM release ships gemma-4 GDN support.

2. ``VllmQrQwen`` currently serves Qwen/Qwen3.6-27B-FP8 (HF-published
   FP8 build, ~27 GiB resident weights, served as ``qwen3.6-27b``).
   vLLM auto-detects the FP8 quantization config from the model's
   ``config.json`` so no explicit ``--quantization fp8`` flag is needed
   on the ``vllm serve`` command line. The bf16 build of the same model
   was tried first but its ~54 GiB resident weight set exceeded Modal's
   CRIU+CUDA checkpointer's reliable restore window
   (``CudaCheckpointException: Get state command timed out``), so we
   moved to the FP8 build whose half-size footprint sits comfortably
   below the empirical ceiling. Both 9B and 27B Qwen3.* variants carry
   a populated HF ``vision_config`` so vLLM V1's ``profile_run`` would
   otherwise run an unconditional MM dummy probe that crashes in
   ``transformers.processing_utils.get_text_with_replacements`` with
   ``StopIteration``. The load-bearing fix is in
   ``qr_sampler.connectors.modal.vllm_serve._install_mm_probe_skip_patch``,
   which monkey-patches ``GPUModelRunner.profile_run`` to set
   ``mm_config.skip_mm_profiling=True`` at entry — vLLM's own supported
   short-circuit at gpu_model_runner.py:5226 in v0.17.0. The patch fires
   ONE event (``vllm.mm.probe_skipped``) on every cold-start; absence of
   that event on a future cold-start means the patch lost its hook.

Both prior model directories (Qwen3.5-9B, gemma-4-31B) remain on the
``llm-weights`` volume; restoring either is a code-only change.

Deploy:
    modal deploy -m qr_sampler.connectors.modal.app

One-shot weights download (before first deploy / on model upgrade):
    modal run -m qr_sampler.connectors.modal.app::download_weights

Snapshot-failure fallback (Pre-flight §11.7): if memory-snapshot restore
fails on H200, set `enable_memory_snapshot=False` on the affected class
and redeploy. Cold start becomes ~30-45s instead of ~10-15s; pre-baked
weights still cut the majority of init time. Do NOT add `keep_warm=1`
on either class — always-on H200 cost is unacceptable per Pre-flight §11.7.

GPU history: this app shipped initially on B200 pinned to us-east-1.
The B200 pool in that region was capacity-starved at first deploy time
(2026-05-19) — Modal scheduling sat in a "waiting to be scheduled" state
for both classes. Stepping down one tier to H200 (still fits both
9B model at bf16 with the cmd-configured max_model_len=32768)
widens the schedulable pool while keeping the us-east-1 region pin in
place. If H200 still queues in us-east-1, the next knob is to relax
that region pin (see comment in `_CLS_KWARGS`).
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "qr-llm-chat"

# ---- Served model: single source of truth ----------------------------------
# To swap the served model, edit these four constants, then:
#   1. modal run -m qr_sampler.connectors.modal.app::download_weights
#   2. modal deploy -m qr_sampler.connectors.modal.app
#
# ``MODEL_SERVED_NAME`` MUST stay in lockstep with ``_QWEN_ID`` in
# ``qr_llm_chat/bootstrap_connections.py`` and with the comparison Pipe's
# ``base_models`` default in
# ``qr_llm_chat/functions/_sources/qr_comparison_pipe.py``. A mismatch breaks
# OWUI routing (vLLM returns 404 on /v1/chat/completions when ``model:``
# does not match ``/v1/models``).
MODEL_HF_REPO_ID = "Qwen/Qwen3.6-27B-FP8"  # HF-published FP8 build (~27 GiB resident, vs ~54 GiB for bf16)
MODEL_SERVED_NAME = "qwen3.6-27b"  # /v1/models id; lockstep with qr-llm-chat _QWEN_ID (precision is implementation detail, not picker-visible)
MODEL_REVISION = os.environ.get("QWEN_REVISION", "")  # optional commit SHA pin
MODEL_GPU_MEMORY_UTILIZATION = "0.8"  # was 0.9; lowered to match Modal's lfm_snapshot.py example for snapshot-restore reliability (smaller resident GPU state = faster cuda-checkpoint enumeration)

# Phase 2 R6: anchor for the cold-start budget event. Captured once at
# module import (i.e. each Modal container's Python process boot). The
# end-of-_wake event uses ``time.monotonic() - _CONTAINER_START_MONOTONIC``
# to report total cold-start elapsed time independent of any external
# clock — the metric §15.1-15.5 success-criteria check against.
_CONTAINER_START_MONOTONIC: float = time.monotonic()

# Repo root, computed relative to this file
# (src/qr_sampler/connectors/modal/app.py — four parents up from here).
# Only meaningful at image-build time (locally). In a Modal container Modal
# mounts app.py at /root/app.py, which has no parents[4] — we fall back to a
# placeholder because _REPO_ROOT is only read by `Image.from_dockerfile`'s
# `context_dir` arg at build time.
try:
    _REPO_ROOT = Path(__file__).resolve().parents[4]
except IndexError:
    _REPO_ROOT = Path("/")


def _qr_llm_chat_functions_dir() -> Path:
    """Resolve the on-disk location of `qr_llm_chat/functions/` at deploy time.

    Used by `_OWUI_IMAGE.add_local_dir(...)` below to ship the JSON
    Function envelopes (`qr_sampler_filter.json`, `qr_comparison_pipe.json`)
    into the container alongside the Python source. Resolves the package via
    Python's own import machinery so the path is correct whether qr-llm-chat
    is installed as a `pip install -e` sibling editable or a published wheel.

    Container-restore tolerance
    ---------------------------
    This module is imported in EVERY container in the deploy (OWUI, both
    vLLM classes) because Modal does ``importlib.import_module`` on the
    class's defining module when restoring the container. The vLLM
    containers do NOT ship qr_llm_chat (their image only adds qr_sampler),
    so `find_spec("qr_llm_chat")` returns None there.

    At RESTORE time the image is already built; ``.add_local_dir(...)``
    is metadata Modal does not re-read. So returning a placeholder Path
    when qr_llm_chat is missing keeps module import working in vLLM
    containers without affecting the OWUI image build (which always
    runs on the deploy host, where qr_llm_chat is installed and the
    real path is returned).

    Hard-failing the deploy host's case is still important — if the
    operator forgot ``pip install -e qr-llm-chat`` before
    ``modal deploy``, ``add_local_dir`` would silently upload an empty
    placeholder and the OWUI Function-bundle import would fail at first
    restore. So we keep the loud error, but gate it on
    ``MODAL_TASK_ID`` being unset (= we are on the deploy host, not in
    a Modal container). Modal sets ``MODAL_TASK_ID`` for every container
    at runtime; on the deploy host it is absent.
    """
    spec = importlib.util.find_spec("qr_llm_chat")
    in_modal_container = bool(os.environ.get("MODAL_TASK_ID"))
    if spec is None or spec.origin is None:
        if in_modal_container:
            # vLLM containers ship qr_sampler but not qr_llm_chat. They
            # never consume the OWUI image, so returning a placeholder
            # path here is safe — Modal does not rebuild images at
            # restore time and ``.add_local_dir(...)`` metadata is not
            # re-validated. The placeholder is intentionally an obvious
            # marker so a future regression that DOES try to use the
            # path produces a greppable error.
            return Path("/__qr_llm_chat_functions_unavailable_in_container__")
        raise RuntimeError(
            "qr_llm_chat is not importable in the deploy host's Python env. "
            "Run `pip install -e <path/to/qr-llm-chat>` in the venv you use "
            "for `modal deploy` and try again."
        )
    functions_dir = Path(spec.origin).resolve().parent / "functions"
    if not functions_dir.is_dir():
        if in_modal_container:
            return Path("/__qr_llm_chat_functions_unavailable_in_container__")
        raise RuntimeError(
            f"Expected `qr_llm_chat/functions/` at {functions_dir}; not found. "
            "If the package layout has changed, update _qr_llm_chat_functions_dir "
            "in qr_sampler/connectors/modal/app.py."
        )
    return functions_dir


# --- Volumes ---------------------------------------------------------------

# Currently only Qwen 3.6 27B-FP8 (HF-published FP8 build) is actively
# served; the volume also retains prior model directories (Qwen 3.6 27B
# bf16, Qwen 3.5 9B, Gemma 4 31B) for warm-cache resume if any of those
# return as the active ``MODEL_HF_REPO_ID``.
# Populated by `download_weights`; each class mounts it read-only and reads
# only its own subdirectory at engine init.
weights_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

# vLLM torch.compile / AOT compile artefact cache. Without this volume the
# 2-min Dynamo bytecode transform + 35s AOT compile re-runs on every cold-
# start because /root/.cache/vllm/ is ephemeral container storage. With the
# volume mounted, the second cold-start sees ``Reusing cached graph...`` and
# the dispatcher-ready latency drops by ~130s. Per-VLLM-version cache key
# (``torch_compile_cache/<10-char-hash>/...``) means a vLLM upgrade
# automatically invalidates the cache without manual cleanup.
vllm_cache_volume = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# --- Secrets ---------------------------------------------------------------

# Provisioned via `modal secret create` — see modal_secrets.md (co-located).
qr_sampler_prod_secret = modal.Secret.from_name("qr-sampler-prod")
hf_token_secret = modal.Secret.from_name("huggingface-secret")

# --- Images ----------------------------------------------------------------

# Lightweight image for the one-shot weights downloader.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    # The full qr-sampler runtime dep set (mirroring pyproject.toml) is
    # needed here because Modal's harness imports the entire defining
    # module (``qr_sampler.connectors.modal.app``) to introspect
    # ``download_weights`` before launching it. That import walks
    # ``qr_sampler.__init__`` → ``qr_sampler.core.pipeline`` (numpy) →
    # ``qr_sampler.entropy.registry`` (grpcio/protobuf) →
    # ``qr_sampler.config`` (pydantic/pyyaml). Missing any one of these
    # crashes the container at module-import time before download_weights
    # is ever called.
    .pip_install(
        "huggingface_hub>=0.24",
        "numpy>=1.26,<3",
        "pydantic>=2.5.0,<3",
        "pydantic-settings>=2.5.0,<3",
        "grpcio>=1.68.0",
        "protobuf>=5.26.0",
        "pyyaml>=6.0",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

# GPU image built from Dockerfile.vllm, including the qr-sampler source.
#
# Why NO `add_python` kwarg here (was previously `add_python="3.12"`):
# `add_python="X"` makes Modal install an ADDITIONAL Python interpreter as a
# parallel layer, with its own site-packages. The Dockerfile's
# `pip install --no-cache-dir .` (and the `vllm/vllm-openai:v0.6.6` base's
# preinstalled vllm + torch + CUDA wheels) live in the BASE image's Python
# only — invisible to a parallel 3.12. Result with the old setting: container
# restore crashed with `ModuleNotFoundError: No module named 'vllm'` on
# `from vllm.engine.arg_utils import AsyncEngineArgs` in
# `qr_sampler.connectors.modal.vllm_serve`, after Modal's qr_sampler import
# happened against the empty parallel 3.12.
#
# Dropping `add_python` makes Modal use the base image's Python (verified
# safe via Modal SDK source: `modal/image.py:1852-1897` — stage-2 setup
# skips Python install entirely when `add_python=None`). The base ships
# Python 3.12 at /usr/local/lib/python3.12/dist-packages with vllm + torch
# + CUDA already installed; that is now also where `add_local_python_source`
# below ships qr_sampler, where the Dockerfile installed its runtime deps,
# and where Modal restores from snapshot.
#
# No `.pip_install(...)` layer here: the Dockerfile's
# `pip install --no-cache-dir .` already installs qr-sampler (which pulls
# numpy / pydantic / pydantic-settings / grpcio / protobuf / pyyaml
# transitively via pyproject.toml [project].dependencies), and the
# Dockerfile separately installs `huggingface_hub`, `fastapi`, `httpx`,
# and `grpcio`. A redundant `.pip_install()` here would actually BREAK
# the build: Modal's build harness invokes `python -m pip install ...` on
# this layer, but the vllm/vllm-openai base image ships only
# `/usr/local/bin/python3.12` (no bare `python` alias) — with `add_python`
# removed there is no parallel-interpreter `python` either, so the layer
# fails with `/bin/sh: 1: python: not found`. Keep deps in the Dockerfile.
#
# `.add_local_python_source("qr_sampler", copy=True)` is kept because it
# ships the LOCAL source on every deploy. Without it, only the Dockerfile-
# baked qr_sampler is reachable, which goes stale between image rebuilds.
# This step is a file-level copy (it does not invoke Python in the image)
# so it is unaffected by the `python` alias absence.
#
# Currently shared by VllmQrQwen only (Gemma class paused — see module
# docstring); the image is still parameterised across classes because the
# per-model split is purely runtime, not build-time.
#
# ``obs`` is shipped alongside ``qr_sampler`` because the structured
# logger in ``qr_sampler.connectors.modal.vllm_serve`` calls
# ``obs.logging.get_logger`` to emit the ``vllm.*`` events
# (engine-args / model-load / built / build_failed / dispatcher_ready).
# Without this layer the container would crash at import time with
# ``ModuleNotFoundError: No module named 'obs'`` before the engine ever
# tried to load. The package lives at qr-llm-chat's repo root (not under
# src/) — Modal resolves it via Python's normal import machinery, so the
# deploy host must have qr-llm-chat editable-installed.
vllm_image = (
    modal.Image.from_dockerfile(
        str(Path(__file__).parent / "Dockerfile.vllm"),
        context_dir=str(_REPO_ROOT),
    )
    .add_local_python_source("qr_sampler", copy=True)
    .add_local_python_source("obs", copy=True)
)

# --- App -------------------------------------------------------------------

app = modal.App(APP_NAME)


# ----- One-shot weights download -------------------------------------------


# 2026-05-22: Qwen/Qwen3.6-27B-FP8 is the active build target. Both 9B and
# 27B Qwen3.* variants (including the FP8 build) carry a populated HF
# ``vision_config``; the MM-probe patch in
# ``qr_sampler.connectors.modal.vllm_serve._install_mm_probe_skip_patch``
# (monkey-patches ``GPUModelRunner.profile_run`` to flip
# ``mm_config.skip_mm_profiling=True``, vLLM's supported short-circuit at
# gpu_model_runner.py:5226) suppresses the MM dummy probe for both sizes,
# so the served model can swap without touching the patch.
#
# The HF repo + revision below alias the module-level ``MODEL_HF_REPO_ID`` /
# ``MODEL_REVISION`` declared near the top of this file. To swap models,
# edit the ``MODEL_*`` block — not these aliases. The underscore-prefixed
# names are retained for back-compat with anything that imports them.
_QWEN_REPO = MODEL_HF_REPO_ID
_QWEN_REVISION = MODEL_REVISION


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": weights_volume},
    secrets=[hf_token_secret],
    timeout=60 * 60,  # 1 hour — download typically finishes in 10-20 min
)
def download_weights() -> dict[str, str]:
    """Populate the ``llm-weights`` Volume with the Qwen model directory.

    Run once per model switch / version bump:
        modal run -m qr_sampler.connectors.modal.app::download_weights

    Idempotent — re-running just re-validates the cache. The active repo +
    optional revision pin come from the module-level ``MODEL_HF_REPO_ID`` /
    ``MODEL_REVISION`` constants (aliased here as ``_QWEN_REPO`` /
    ``_QWEN_REVISION``). As of 2026-05-22 this targets Qwen/Qwen3.6-27B-FP8
    (the HF-published FP8 build, ~27 GiB resident weights — chosen over
    the bf16 build because the latter's 54 GiB footprint exceeded Modal's
    CRIU+CUDA checkpointer's reliable restore window). The MM-probe
    monkey-patch in ``vllm_serve._install_mm_probe_skip_patch`` makes the
    Qwen3.* family cold-startable regardless of the precision build.
    Prior weight directories (Qwen/Qwen3.6-27B bf16, Qwen/Qwen3.5-9B,
    google/gemma-4-31B) remain on the volume so resuming any of them is
    a code-only change to ``MODEL_HF_REPO_ID`` with a warm cache hit.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    qwen_kwargs: dict[str, Any] = {"repo_id": _QWEN_REPO}
    if _QWEN_REVISION:
        qwen_kwargs["revision"] = _QWEN_REVISION

    qwen_path = snapshot_download(**qwen_kwargs)
    weights_volume.commit()  # type: ignore[attr-defined]

    return {
        "qwen_path": qwen_path,
        "qwen_revision": _QWEN_REVISION or "(latest)",
    }


# ----- Per-model GPU classes -----------------------------------------------
#
# Each class:
#   * Loads ONE model into ONE H200 container.
#   * Scales to zero independently via `scaledown_window=180` (3 min idle).
#   * Exposes its own public Modal URL via `@modal.asgi_app()`.
#
# OWUI's bundled qr_comparison_pipe routes per-request to the right
# Modal URL using its `valves.model_base_urls` map (written at boot by
# `qr_llm_chat.bootstrap_connections`); the model the user picks in the
# OWUI dropdown determines which container wakes.
#
# Only ``VllmQrQwen`` is currently registered (the Gemma class is paused
# — see module docstring). The shared ``_CLS_KWARGS`` below is sized for
# both, so re-introducing a sibling Gemma class is a copy-paste away.


_CLS_KWARGS: dict[str, Any] = {
    "image": vllm_image,
    # H100:1 sized for Qwen3.5-9B at bf16 (~18 GB weights, plus KV cache).
    # Phase 2 cutover from H200 (bf16 27B) to H100:1 (bf16 9B) shrinks the
    # per-token memory footprint dramatically — one H100 serves the model
    # with comfortable headroom for snapshot+wake. FP8 quantization was
    # tried in earlier iter-06 drafts but dropped 2026-05-21 once we
    # confirmed bf16 fits on this hardware budget; the FP8 init path
    # (Qwen3 GDN List[Tensor] in init_fp8_kv_scales) was an avoidable
    # complexity for the storage savings.
    "gpu": "H100:1",
    # Region pool: ``"us"`` (the broad-region option per Modal's docs at
    # docs/guide/region-selection). Broad regions widen the schedulable
    # pool to ALL US-resident clusters (us-east + us-central + us-south +
    # us-west) which is what we need to survive the bursty H200 supply
    # that left earlier narrow combinations capacity-queued (2026-05-19),
    # AND the broad multiplier is the cheaper of the two tiers (1.5x
    # vs 1.75x narrow). Cheaper + wider pool + simpler config — no
    # downside vs the explicit triple it replaces.
    #
    # QRNG-LATENCY CAVEAT (documented honestly so the next operator
    # decides knowingly)
    # ------------------------------------------------------------
    # qr-sampler's ``QuantumGrpcSource`` runs a synchronous GetRandomBytes
    # gRPC RPC PER TOKEN on the inference hot path — vLLM's
    # ``LogitsProcessor`` blocks token emission until the QRNG returns.
    # The Cloudflare-Access front (cloudflared sidecar → CF PoP → CF
    # backbone → CF PoP → QRNG origin) accelerates only the edge hops,
    # not the backbone hop between PoPs; that hop is fiber-bound by the
    # physical distance between the container and the QRNG origin. The
    # Cipherstone QRNG service is colocated in **central US**. A west-
    # coast H200 would add ~30-50 ms RTT to every sampled token - at
    # 50 tok/s that is 1.5-2.5 s of added wall-clock per second of
    # generated output, plainly visible in OWUI's streaming UI.
    #
    # We accept that penalty for the workloads that land on us-west / us-
    # south under ``"us"`` because the alternative is workloads that do
    # not land at all. If steady-state QRNG latency becomes the dominant
    # pain point and H200 capacity in us-central + us-east stabilises,
    # narrow this back to ``["us-central", "us-east"]`` (paying the 1.75x
    # narrow multiplier) — the prior comment block in git history
    # explains the latency math in detail.
    "region": "us",
    "volumes": {
        "/root/.cache/huggingface": weights_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    "secrets": [qr_sampler_prod_secret, hf_token_secret],
    "enable_memory_snapshot": True,
    # GPU snapshot: captures the warm-but-asleep vLLM engine so cold-starts
    # restore in ~10-15 s instead of the ~3-5 min full engine rebuild.
    # Phase 2 substep 4 (snapshot wiring). The engine is put to sleep at
    # snap=True (POST /sleep?level=1, requires VLLM_SERVER_DEV_MODE=1) and
    # woken at snap=False (POST /wake_up).
    "experimental_options": {"enable_gpu_snapshot": True},
    # 3 min idle -> shutdown. With GPU snapshot enabled, restore is
    # cheap (~10-15 s) so scaling to zero aggressively is safe — the
    # snapshot-restore is the warm path for any pause longer than the
    # window. Tight scaledown keeps idle GPU cost minimal; a longer
    # window (e.g. 1800 s) trades cost for occasionally skipping the
    # restore on "let me think" pauses inside a chat session.
    "scaledown_window": 180,
    "max_containers": 1,  # Pre-flight §11.8 cost ceiling, per model
    # Phase 3 iter-06 (2026-05-21): 2-hour container timeout (was 1 hour).
    # The snap=True path can legitimately consume up to:
    #   _STARTUP_TIMEOUT_S (1200) + /sleep (300) + CRIU snapshot (~120) +
    #   margin for Modal worker contention ~= 30 min worst-case.
    # Doubling to 7200 means a single container can absorb one retry
    # (failed first snapshot -> Modal auto-retry within the same container
    # lifecycle) without hitting the hard cap. This deliberately overrides
    # the prior "fail-fast" preference: snapshot generation is a one-time
    # cost we'd rather pay generously than miss.
    "timeout": 60 * 60 * 2,
}


# 2026-05-22: switched the 27B from native bf16 (54 GiB resident weights —
# beyond Modal's CRIU+CUDA checkpointer's reliable restore window, which
# produced intermittent ``CudaCheckpointException: Get state command
# timed out`` failures after scale-to-zero) to the HF-published FP8 build
# ``Qwen/Qwen3.6-27B-FP8`` at ~27 GiB. The smaller resident footprint
# fits comfortably within the checkpointer's headroom (the 9B at
# ~10 GiB checkpointed reliably; 27 GiB is well below the empirical
# ceiling), so we keep the snapshot-restored lifecycle and the
# corresponding ~10-15 s warm restore. ``experimental_options`` is
# unchanged from ``_CLS_KWARGS``.
@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrQwen:
    """``vllm serve`` subprocess fronted by Modal's ``@modal.web_server``.

    Phase 2 (2026-05-21) rebuild: replaces the prior ``@modal.asgi_app() +
    asyncio.run(build_dispatcher_for(...))`` pattern. Three structural
    defects from Phase K research are fixed:

    1. **Subprocess stdio buffering** — ``vllm serve`` runs as its own
       process tree, so ``VLLM_LOGGING_LEVEL=DEBUG`` lands in the parent's
       stdout without the asyncio multiplexing that previously swallowed
       ``EngineCore_DP0`` tracebacks.
    2. **Broken EngineClient lifecycle** — ``vllm serve`` owns the engine
       process lifecycle end-to-end; we no longer have to thread
       ``AsyncLLMEngine`` through a custom dispatcher.
    3. **Homegrown snapshot/restore** — Modal's
       ``experimental_options={"enable_gpu_snapshot": True}`` plus vLLM
       sleep mode (``--enable-sleep-mode`` + ``POST /sleep?level=1``)
       replace the prior ``enable_memory_snapshot=False`` workaround.

    Lifecycle:

    * ``@modal.enter(snap=True) _start_and_sleep`` — spawns ``vllm
      serve``, polls ``/health`` until ready, then ``POST /sleep?level=1``
      to free GPU memory before Modal takes the snapshot. Requires
      ``VLLM_SERVER_DEV_MODE=1`` (set in Dockerfile.vllm) to expose the
      sleep/wake endpoints.
    * ``@modal.enter(snap=False) _wake`` — starts the cloudflared sidecar
      (post-restore so no dead socket is captured), then ``POST
      /wake_up`` to re-allocate the engine's GPU state.
    * ``@modal.web_server(port=8000)`` — Modal proxies inbound traffic
      to the running ``vllm serve`` subprocess on localhost:8000.
    * ``@modal.exit() _stop`` — terminates the sidecar and the vLLM
      subprocess on container shutdown.

    Qwen3.6-27B-FP8's HF config carries a populated ``vision_config``
    which would otherwise crash vLLM V1's MM dummy probe. The load-bearing
    fix is the ``_install_mm_probe_skip_patch`` monkey-patch in
    ``qr_sampler.connectors.modal.vllm_serve`` — kept importable for the
    ``--logits-processors``-discovery path (the entry point in
    ``pyproject.toml`` registers ``VLLMAdapter`` as
    ``vllm.logits_processors.qr_sampler``, which ``vllm serve`` picks up
    automatically). We serve the HF-published FP8 build (vLLM auto-detects
    the ``quantization_config`` from the model's ``config.json``, so no
    explicit ``--quantization fp8`` is needed); the choice over the bf16
    build is documented in the module docstring + the ``@app.cls``
    comment above. The vision config is suppressed by the MM-probe
    patch loaded at the same entry-point hook.
    """

    # Machine-friendly ID echoed by vLLM's /v1/models endpoint and used as
    # the routing key throughout OWUI + the comparison Pipe. No spaces or
    # parens here -- the human-readable display label
    # ("qwen3.6-27b (quantum-random)") is set via an OWUI ``model`` table
    # row override seeded by ``qr_llm_chat.bootstrap_connections``. The id
    # MUST stay in lockstep with ``_QWEN_ID`` in
    # ``qr_llm_chat/bootstrap_connections.py`` and the Pipe's
    # ``base_models`` default. No precision suffix in the served name:
    # the model architecture id is the user-visible routing key, while
    # the actual precision (currently FP8 from the HF-published build)
    # is an implementation detail tracked via ``MODEL_HF_REPO_ID``.
    #
    # Both values below alias the module-level ``MODEL_SERVED_NAME`` /
    # ``MODEL_HF_REPO_ID`` constants declared at the top of this file —
    # edit those to swap models, not these class attributes.
    SERVED_MODEL_NAME = MODEL_SERVED_NAME
    # Hugging Face repo id; same value as ``_QWEN_REPO`` above. The downloader
    # (``download_weights``) populates ``llm-weights`` from this repo; vLLM
    # then loads from the cached snapshot at serve time.
    HF_REPO_ID = MODEL_HF_REPO_ID

    # vLLM serve HTTP endpoint inside the container. Modal's
    # @modal.web_server proxies inbound traffic here.
    _VLLM_PORT = 8000
    _VLLM_HOST = "127.0.0.1"
    _VLLM_BASE_URL = f"http://{_VLLM_HOST}:{_VLLM_PORT}"
    # Phase 3 iter-06 (2026-05-21): generous timeouts so snapshot generation
    # is never interrupted mid-flight. The trade-off — a stuck vllm serve
    # may now consume the full @app.cls timeout instead of failing fast —
    # is intentional: we want the snapshot to succeed on the first try, and
    # an inconclusive timeout is more useful diagnostically than a hard cap
    # masking a real but slow-to-surface init issue.
    #   _STARTUP_TIMEOUT_S: 20 min covers bf16 9B HF load (~3 min) + dynamo
    #     bytecode transform (~2 min) + torch.compile + AOT cache miss
    #     (~3 min) + V1 profile_run + the MM-probe patch, with margin.
    #     Iter-04 / iter-05 both fit in <10 min; doubling it leaves room
    #     for tail latency (Modal worker contention, HF bandwidth dips).
    #   _SLEEP_WAKE_TIMEOUT_S: 5 min covers /sleep?level=1 freeing the
    #     full ~58 GiB KV cache region observed in iter-05 logs. The
    #     iter-05 default 60s WAS enough for the API call itself, but
    #     left no margin for the post-sleep cudaFree storm to settle
    #     before snapshot fires. Symmetric on /wake_up: 5 min is plenty
    #     for KV-block re-allocation.
    _STARTUP_TIMEOUT_S = 1200  # 20 min — vllm serve + dynamo + MM patch
    _SLEEP_WAKE_TIMEOUT_S = 300  # 5 min — /sleep level=1 + cudaFree settle

    @modal.enter(snap=True)
    def _start_and_sleep(self) -> None:
        """Spawn ``vllm serve``, wait for /health, then put it to sleep.

        Runs PRE-snapshot. The snapshot captures the warm engine in its
        sleeping (GPU-released) state so restore is cheap.
        """
        import subprocess
        import time

        import httpx
        from obs.events import (
            VLLM_ARGV_UNRECOGNIZED,
            VLLM_ARGV_VALIDATED,
            VLLM_ENGINE_BUILD_FAILED,
            VLLM_SLEEP_FAIL,
            VLLM_SLEEP_OK,
        )
        from obs.logging import get_logger

        log = get_logger(f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}")

        # Forward the container's env into the subprocess so vllm serve
        # inherits VLLM_LOGGING_LEVEL=DEBUG, VLLM_SERVER_DEV_MODE=1,
        # HF_HUB_ENABLE_HF_TRANSFER=1 (set in Dockerfile.vllm) plus the
        # QR_* entropy config (set in Dockerfile.vllm and overridable
        # via the qr-sampler-prod Modal Secret).
        env = os.environ.copy()
        # Belt-and-braces: also set unbuffered I/O explicitly so vllm
        # serve's Python stdout streams in real time even if the parent
        # forgot the env var.
        env.setdefault("PYTHONUNBUFFERED", "1")

        # iter-09 (2026-05-21): iter-02's QR_ENTROPY_SOURCE_TYPE=system /
        # QR_PREINIT_ENTROPY_SOURCES=system overrides REMOVED. They were an
        # isolation workaround for the snapshot /wake_up 500, which iter-08
        # candidate E proved was actually Modal's edge proxy hanging on
        # vllm serve's --host 127.0.0.1 binding (see LEARNINGS.md iter-08).
        # With the binding fixed, the snapshot wake is reliable, so the
        # subprocess now inherits QR_* defaults from Dockerfile.vllm
        # (QR_ENTROPY_SOURCE_TYPE=quantum_grpc, QR_FALLBACK_MODE=system) and
        # the request path consults the QRNG via the cloudflared sidecar
        # started in _wake. The qr_sampler client's QuantumGrpcSource opens
        # the loopback gRPC channel lazily on first get_random_bytes() call
        # (see auto-memory qrng_tcp_preprobe), so there is no live socket
        # for the snapshot to capture — the lazy-init is the load-bearing
        # protection against a snapshot-captured-dead-channel.

        cmd = [
            "vllm",
            "serve",
            self.HF_REPO_ID,
            "--served-model-name",
            self.SERVED_MODEL_NAME,
            # Iter-08 candidate D finding: bind vllm serve to 0.0.0.0,
            # NOT to self._VLLM_HOST (127.0.0.1). Modal's
            # @modal.web_server(port=8000) proxies inbound traffic via
            # the container's *external* interface — a connection that
            # vllm serve refuses when bound to loopback only. Symptom is
            # silent TLS-accepted-then-hang on every external request,
            # NOT a vLLM 500. Internal /health/sleep/wake_up probes from
            # _start_and_sleep / _wake still use self._VLLM_HOST
            # (127.0.0.1) — they share the process tree and don't go
            # through Modal's proxy. modal-labs/modal-examples
            # 06_gpu_and_ml/llm-serving/vllm_inference.py uses the same
            # 0.0.0.0 binding for the same reason.
            "--host",
            "0.0.0.0",
            "--port",
            str(self._VLLM_PORT),
            # Phase 3 iter-06 (2026-05-21): --enable-sleep-mode RESTORED.
            # iter-05 dropped it and hit a deterministic snapshot crash:
            #   modal._runtime.gpu_memory_snapshot.CudaCheckpointException:
            #   Failed to checkpoint 1 processes: PID: 34
            #   Get state command timed out
            # Modal's CRIU-based CUDA checkpointer cannot enumerate
            # vLLM's live state (10.8 GiB weights + 3.55 GiB activation +
            # 58.36 GiB KV cache across 3621 blocks) within its timeout.
            # POST /sleep?level=1 releases all KV blocks pre-snapshot so
            # only the weights remain — the documented Modal+vLLM
            # pattern. The iter-04 NCCL TCPStore HeartbeatMonitor flood
            # we blamed previously is silenced by
            # TORCH_NCCL_ENABLE_MONITORING=0 in Dockerfile.vllm; it
            # was diagnostic noise, not a request-blocker.
            "--enable-sleep-mode",
            # Precision: weights are FP8 (the model is ``Qwen/Qwen3.6-27B-FP8``,
            # the HF-published FP8 build; vLLM auto-detects the FP8
            # ``quantization_config`` in the model's ``config.json``, so no
            # explicit ``--quantization fp8`` flag is needed on the serve
            # command line). KV cache stays at vLLM's default (bf16), which
            # keeps the ``init_fp8_kv_scales`` patch in ``vllm_patches.py``
            # dormant — its own ``cache_dtype.startswith("fp8")`` gate
            # short-circuits when the cmd does not request ``--kv-cache-dtype
            # fp8``. FP8 weights (~27 GiB) + bf16 KV cache (bounded by
            # max-num-seqs=16 × max-model-len=32768) + activation fits
            # comfortably under the 73 GiB usable budget on the H100:1 with
            # the gpu-memory-utilization=0.9 ceiling below.
            # Prefix caching (V1 default, explicit for documentation):
            # multi-turn web chat re-sends the full conversation each
            # turn — vLLM detects KV-block overlap with the previous
            # turn and skips re-prefill. The single most impactful UX
            # knob for a chatbot deployment. Empirically NOT the cause
            # of the intermittent cuda-checkpoint --get-state timeouts
            # on snapshot restore (2026-05-22 A/B test: dropping it did
            # not improve restore reliability), so kept on.
            "--enable-prefix-caching",
            # Context window: 32768 tokens. iter-04's 8192 capped real
            # multi-turn chat at ~4 turns before truncation, which is
            # unusable for OWUI's typical session. 32k accommodates
            # ~16 turns of dense conversation. KV pre-allocation cost is
            # bounded by max-num-seqs × per-token KV (vLLM packs
            # dynamically), not by max-model-len alone.
            "--max-model-len",
            "32768",
            # Concurrent generation slots inside the vLLM scheduler.
            # Lowered to 4 (was 16, vLLM default 256) to keep the KV
            # cache page count small enough for cuda-checkpoint to
            # enumerate during snapshot restore within its 180 s timeout.
            # @modal.concurrent(max_inputs=8) caps in-flight requests
            # per container at 8; with --max-num-seqs=4 we accept that 5+
            # simultaneous requests queue at the scheduler boundary
            # rather than fan out further inside vLLM.
            "--max-num-seqs",
            "4",
            # Cap the CUDA graph capture set explicitly to match
            # max-num-seqs above. vLLM auto-derives capture sizes from
            # max-num-seqs but Modal's lfm_snapshot.py example sets this
            # flag explicitly, which we mirror as a belt-and-braces
            # guarantee that no larger graph variants sneak into the
            # snapshotted state. Smaller captured graph set = less GPU
            # state for cuda-checkpoint to enumerate on restore.
            "--max-cudagraph-capture-size",
            "4",
            # Per-step batch budget. Higher = better prefill throughput
            # (one step packs a long system prompt + new user message),
            # at the cost of slightly slower per-step decode. 8192 is
            # the sweet spot for 1-4k system prompts + ~512 user
            # tokens.
            "--max-num-batched-tokens",
            "8192",
            # CPU-RAM swap (GiB) for KV cache eviction overflow. With a
            # 32k context window, evicting whole sequences to host RAM
            # rather than recomputing them is a big win. Default 4 is
            # tight; 16 gives ~3 32k-token sequences of headroom on
            # Modal's default container RAM (~100 GiB).
            "--swap-space",
            "16",
            # GPU memory utilization ceiling. Read from the module-level
            # ``MODEL_GPU_MEMORY_UTILIZATION`` constant so it tracks the
            # active model: 0.92 was right for Qwen3.5-9B (bf16 ~18 GiB
            # weights + KV cache fit easily on H100). For Qwen3.6-27B-FP8
            # (~27 GiB FP8 weights), 0.9 leaves ~7 GiB headroom on the
            # 79 GiB H100 for cuBLAS workspaces + CUDA contexts +
            # paged-attention overflow to CPU (driven by --max-num-seqs 16
            # + --swap-space 16 below). Drop further (0.88, 0.85) if
            # /wake_up surfaces OOM after a model swap.
            "--gpu-memory-utilization",
            MODEL_GPU_MEMORY_UTILIZATION,
            # Per-request body logging is OFF by default in vLLM 0.17+ —
            # the inverse opt-in flag is ``--enable-log-requests``. The
            # prior ``--disable-log-requests`` (from vLLM 0.6 era) was
            # removed in iter-07 (2026-05-21) after vllm serve rejected
            # it with rc=2 / "unrecognized arguments". No replacement
            # needed; the quiet behavior is now the default.
            # Phase 2 R2 (belt-and-braces LP registration): the
            # ``vllm.logits_processors`` entry-point group in
            # qr-sampler/pyproject.toml is the primary discovery path,
            # but vLLM 0.17.0 has historically been finicky about
            # entry-point loading order vs. engine init. Passing
            # --logits-processors explicitly guarantees registration
            # regardless of plugin discovery state.
            "--logits-processors",
            "qr_sampler.engines.vllm:VLLMAdapter",
            # iter-11 (2026-05-21): emit Qwen3-family ``<think>...</think>``
            # blocks as a separate ``reasoning_content`` field in the
            # OpenAI-compatible response. vLLM 0.17+ ships the ``qwen3``
            # reasoning parser that extracts the block into the dedicated
            # field; Open WebUI 0.9.5 then renders it as a collapsible
            # "Thought for N seconds" panel above the assistant text.
            # Without this flag, the model still emits the tags but they
            # land inline in ``content`` and OWUI shows them as raw
            # markdown. The qr-sampler comparison Pipe mirrors the same
            # surface (see qr_comparison_pipe.py _extract_delta_text).
            # Sources: https://docs.vllm.ai/en/latest/features/reasoning_outputs/
            # + https://docs.openwebui.com/features/chat-conversations/chat-features/reasoning-models/
            "--reasoning-parser",
            "qwen3",
        ]

        # Iter-08 / iter-10: vLLM's CLI churns between minor releases
        # (PR #21739 removed --disable-log-requests in v0.10/v0.11). Iter-08
        # spawned ``vllm serve --help`` and regex-parsed the output to build
        # a "supported flag" set, but vLLM 0.17 uses Rich for help output
        # (word-wrap + ANSI styling) which defeated the regex on every
        # cold-start. Iter-10 replaces the subprocess+regex with a direct
        # import of vLLM's own argparse builder — enumerates the canonical
        # action list, no subprocess wait, no regex fragility, no Rich
        # interaction. On the happy path, the ``vllm.argv.validated`` event
        # is the canonical "deploy went through the argv gate" grep target;
        # a real flag rename surfaces as ``vllm.argv.unrecognized`` with
        # the offending flag named in the traceback before vllm serve is
        # ever spawned.
        try:
            from vllm.entrypoints.openai.cli_args import make_arg_parser
            from vllm.utils.argparse_utils import FlexibleArgumentParser

            _argv_parser = make_arg_parser(FlexibleArgumentParser())
            supported_flags = {
                opt
                for action in _argv_parser._actions
                for opt in action.option_strings
                if opt.startswith("--")
            }
        except Exception as err:
            log.warning(
                "vllm argparse introspection failed; skipping argv validation",
                extra={
                    "event": "vllm.argv.help_probe_failed",
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                },
            )
            supported_flags = set()
        if supported_flags:
            unknown = [t for t in cmd if t.startswith("--") and t not in supported_flags]
            if unknown:
                sample = sorted(supported_flags)[:20]
                log.error(
                    "vllm serve argv contains unrecognized flag(s): %s",
                    unknown,
                    extra={
                        "event": VLLM_ARGV_UNRECOGNIZED,
                        "unknown_flags": unknown,
                        "supported_flags_sample": sample,
                    },
                )
                raise RuntimeError(
                    f"vllm serve argv contains unrecognized flag(s) {unknown}; "
                    f"supported sample={sample}"
                )
            log.info(
                "vllm serve argv validated against vllm.entrypoints.openai.cli_args",
                extra={
                    "event": VLLM_ARGV_VALIDATED,
                    "cmd": cmd,
                    "supported_flag_count": len(supported_flags),
                },
            )

        log.info(
            "Spawning vllm serve subprocess for %s",
            self.SERVED_MODEL_NAME,
            extra={
                "event": "vllm.subprocess.spawn",
                "served_model_name": self.SERVED_MODEL_NAME,
                "hf_repo_id": self.HF_REPO_ID,
                "cmd": cmd,
            },
        )

        # subprocess.Popen with no stdout/stderr capture: vllm serve's
        # logs go directly to the container's stdout/stderr where
        # ``modal app logs`` picks them up. This is the WHOLE POINT of
        # the lifecycle restructure — see Phase K research §"Subprocess
        # stdio buffering".
        self._vllm_proc = subprocess.Popen(cmd, env=env)

        # Poll /health every 2 s up to the startup budget.
        deadline = time.monotonic() + self._STARTUP_TIMEOUT_S
        health_url = f"{self._VLLM_BASE_URL}/health"
        ready = False
        while time.monotonic() < deadline:
            # If the subprocess died, surface the build failure as a
            # structured event before raising.
            rc = self._vllm_proc.poll()
            if rc is not None:
                log.error(
                    "vllm serve exited during startup (rc=%s)",
                    rc,
                    extra={
                        "event": VLLM_ENGINE_BUILD_FAILED,
                        "served_model_name": self.SERVED_MODEL_NAME,
                        "hf_repo_id": self.HF_REPO_ID,
                        "error_type": "SubprocessExited",
                        "error_msg": f"vllm serve exited with rc={rc}",
                        "phase": "VllmQrQwen._start_and_sleep",
                    },
                )
                raise RuntimeError(f"vllm serve exited rc={rc} during startup")
            try:
                r = httpx.get(health_url, timeout=5.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(2.0)

        if not ready:
            log.error(
                "vllm serve did not become healthy within %ds",
                self._STARTUP_TIMEOUT_S,
                extra={
                    "event": VLLM_ENGINE_BUILD_FAILED,
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "hf_repo_id": self.HF_REPO_ID,
                    "error_type": "StartupTimeout",
                    "error_msg": f"/health never returned 200 within {self._STARTUP_TIMEOUT_S}s",
                    "phase": "VllmQrQwen._start_and_sleep",
                },
            )
            raise RuntimeError(
                f"vllm serve /health did not return 200 within {self._STARTUP_TIMEOUT_S}s"
            )

        # 2026-05-22: warmup inference BEFORE /sleep, per Modal's vLLM
        # snapshot example pattern. The warmup runs trigger
        # torch.compile (inductor pass) + CUDA graph capture for the
        # realistic batch sizes the engine will see in production, so
        # the inductor cache + graph state is fully baked at snapshot
        # capture time. Without warmup, the snapshot captures partial
        # compile state and the cuda-checkpoint enumerator times out on
        # restore (paired with the ``TORCHINDUCTOR_COMPILE_THREADS=1``
        # env var in Dockerfile.vllm which serialises the inductor
        # compile so its driver state IS enumerable). max_tokens=8 +
        # temperature=0 keeps each iteration sub-second; 3 iterations
        # match Modal's reference example. Errors are soft-fail: if
        # warmup cannot connect or vllm returns 5xx, we log and proceed
        # to /sleep — the pre-snapshot path is best-effort and a partial
        # warmup is better than no warmup.
        warmup_url = f"{self._VLLM_BASE_URL}/v1/chat/completions"
        warmup_body = {
            "model": self.SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        }
        warmup_t0 = time.monotonic()
        warmup_ok = 0
        for i in range(3):
            try:
                r = httpx.post(warmup_url, json=warmup_body, timeout=120.0)
                r.raise_for_status()
                warmup_ok += 1
            except Exception as exc:
                log.warning(
                    "warmup request %d/3 failed: %s",
                    i + 1,
                    exc,
                    extra={
                        "event": "vllm.warmup.failed",
                        "served_model_name": self.SERVED_MODEL_NAME,
                        "attempt": i + 1,
                        "error_type": type(exc).__name__,
                    },
                )
        warmup_duration_ms = (time.monotonic() - warmup_t0) * 1000.0
        log.info(
            "vllm warmup complete: %d/3 successful in %.0f ms",
            warmup_ok,
            warmup_duration_ms,
            extra={
                "event": "vllm.warmup.ok",
                "served_model_name": self.SERVED_MODEL_NAME,
                "successful": warmup_ok,
                "duration_ms": warmup_duration_ms,
            },
        )

        # Phase 3 iter-06 (2026-05-21): POST /sleep?level=1 RESTORED.
        # iter-05 attempted to skip this on the theory that Modal's
        # GPU snapshot could capture the live engine; it failed with
        # CudaCheckpointException because Modal's CRIU+CUDA checkpoint
        # API timed out enumerating 58 GiB of live KV cache + 3621
        # blocks. level=1 releases all KV blocks to CPU before snapshot
        # so only the ~10 GiB weights region remains — small enough for
        # the checkpointer to enumerate within its window. level=2
        # would also discard weights and require a full reload on wake;
        # we want fast restore.
        sleep_url = f"{self._VLLM_BASE_URL}/sleep?level=1"
        t0 = time.monotonic()
        try:
            r = httpx.post(sleep_url, timeout=self._SLEEP_WAKE_TIMEOUT_S)
            r.raise_for_status()
        except Exception as err:
            duration_ms = (time.monotonic() - t0) * 1000.0
            log.error(
                "vllm sleep failed for %s: %s",
                self.SERVED_MODEL_NAME,
                err,
                extra={
                    "event": VLLM_SLEEP_FAIL,
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "duration_ms": duration_ms,
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                },
            )
            raise
        duration_ms = (time.monotonic() - t0) * 1000.0
        log.info(
            "vllm engine sleeping (snapshot-ready) for %s",
            self.SERVED_MODEL_NAME,
            extra={
                "event": VLLM_SLEEP_OK,
                "served_model_name": self.SERVED_MODEL_NAME,
                "duration_ms": duration_ms,
            },
        )

        # Phase 3 iter-06 (2026-05-21): belt-and-braces GPU-allocator drain
        # BEFORE Modal's CRIU+CUDA checkpointer walks the process. vLLM's
        # /sleep?level=1 frees KV blocks into PyTorch's caching allocator,
        # not back to the CUDA driver — the allocator holds the pages for
        # reuse. CRIU still has to enumerate every cached allocation, which
        # was the root cause of iter-05's "Get state command timed out".
        # Calling ``torch.cuda.empty_cache()`` returns the cached pages to
        # the driver so CRIU sees only the weight buffers (~18 GiB) instead
        # of weights + the held-but-empty KV region (~58 GiB more).
        # ``gc.collect()`` clears any Python-side references the engine was
        # holding (request batches, sampler state). Both calls are idempotent
        # and quick (<1 s); they exist purely to reduce the live working
        # set CRIU sees, maximising snapshot success rate.
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # ipc_collect cleans up any cross-process CUDA IPC handles
                # vLLM spun up for shared-memory tensor passing; harmless
                # to call when no IPC is in use, critical when EngineCore_DP0
                # leaves a handle dangling.
                torch.cuda.ipc_collect()
                log.info(
                    "GPU allocator drained pre-snapshot",
                    extra={
                        "event": "vllm.snapshot.gpu_drained",
                        "served_model_name": self.SERVED_MODEL_NAME,
                    },
                )
        except Exception as exc:
            # Drain is advisory; if torch is unavailable or the call fails
            # we still let the snapshot proceed. The /sleep above is the
            # load-bearing protection.
            log.warning(
                "pre-snapshot GPU drain skipped: %s",
                exc,
                extra={
                    "event": "vllm.snapshot.gpu_drain_skipped",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "error_type": type(exc).__name__,
                },
            )

    @modal.enter(snap=False)
    def _wake(self) -> None:
        """Start cloudflared sidecar and wake the engine after snapshot restore.

        Runs POST-snapshot on every cold-start (including the first
        deploy after the snap=True branch). The cloudflared sidecar
        intentionally starts here, not at snap=True, so no live socket
        is frozen into the snapshot — see auto-memory
        ``modal_vllm_303_hang`` and the prior implementation's
        ``start_tunnel`` comment for the rationale.
        """
        import threading
        import time

        import httpx
        from obs.events import (
            VLLM_COLDSTART_COMPLETE,
            VLLM_WAKE_FAIL,
            VLLM_WAKE_OK,
        )
        from obs.logging import get_logger

        log = get_logger(f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}")

        # iter-12 (2026-05-22): cloudflared sidecar startup is now
        # PARALLELIZED with /wake_up + /health below. Previously
        # serialized, which added the sidecar's ~5-15 s tunnel-bootstrap
        # window to every cold-start. Because ``_start_qrng_tunnel`` is
        # soft-fail by design (returns ``None`` on any failure; the
        # request path falls through to ``SystemEntropySource`` via
        # ``FallbackEntropySource``), there is no correctness reason to
        # block ``_wake`` on it — at worst, the first few tokens after a
        # cold-start use system entropy and surface ``entropy.degraded``
        # events until the tunnel is up. We initialise the attribute to
        # ``None`` so any concurrent access during the bootstrap window
        # gets the fallback path cleanly. The thread is a daemon so
        # ``_stop`` need not join it on container shutdown.
        #
        # iter-09 background (retained for context): the sidecar starts
        # here in ``@modal.enter(snap=False)`` — POST-snapshot — so no
        # live cloudflared socket is captured into the snapshot (auto-
        # memory modal_vllm_303_hang).
        self._cloudflared = None

        def _spawn_cloudflared() -> None:
            try:
                self._cloudflared = _start_qrng_tunnel(self.SERVED_MODEL_NAME)
            except Exception as exc:
                # _start_qrng_tunnel already soft-fails internally; this
                # guard is belt-and-braces for an unhandled raise so the
                # daemon thread cannot kill the container.
                log.warning(
                    "cloudflared background spawn raised: %s",
                    exc,
                    extra={
                        "event": "cloudflared.background_spawn_raised",
                        "served_model_name": self.SERVED_MODEL_NAME,
                        "error_type": type(exc).__name__,
                    },
                )

        threading.Thread(
            target=_spawn_cloudflared,
            name=f"cloudflared-spawn-{self.SERVED_MODEL_NAME}",
            daemon=True,
        ).start()

        # Phase 3 iter-06 (2026-05-21): POST /wake_up RESTORED, paired
        # with sleep mode restore in _start_and_sleep (see iter-05
        # CudaCheckpointException analysis there). The belt-and-braces
        # /health poll added in iter-05 is RETAINED — it guarantees the
        # engine is actually serving requests on 127.0.0.1:8000 before
        # we return from snap=False (and thus before Modal's
        # web_server proxy accepts external traffic). The iter-04
        # "deploy-guard probes don't land" symptom — if it recurs — is
        # then provably NOT a wake-state issue, narrowing the diagnosis.
        wake_url = f"{self._VLLM_BASE_URL}/wake_up"
        t0 = time.monotonic()
        try:
            r = httpx.post(wake_url, timeout=self._SLEEP_WAKE_TIMEOUT_S)
            r.raise_for_status()
        except Exception as err:
            duration_ms = (time.monotonic() - t0) * 1000.0
            log.error(
                "vllm wake failed for %s: %s",
                self.SERVED_MODEL_NAME,
                err,
                extra={
                    "event": VLLM_WAKE_FAIL,
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "duration_ms": duration_ms,
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                },
            )
            raise
        wake_duration_ms = (time.monotonic() - t0) * 1000.0

        # Belt-and-braces /health poll: /wake_up returning 200 means
        # the engine has re-allocated GPU state, but vLLM has a brief
        # post-wake window where /v1/chat/completions may 5xx because
        # the scheduler is still binding KV blocks. Polling /health
        # before declaring the container ready closes that race.
        health_url = f"{self._VLLM_BASE_URL}/health"
        deadline = time.monotonic() + self._SLEEP_WAKE_TIMEOUT_S
        ready = False
        while time.monotonic() < deadline:
            try:
                r = httpx.get(health_url, timeout=5.0)
                if r.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
        total_duration_ms = (time.monotonic() - t0) * 1000.0
        if not ready:
            log.error(
                "vllm post-wake /health never returned 200 for %s within %ds",
                self.SERVED_MODEL_NAME,
                self._SLEEP_WAKE_TIMEOUT_S,
                extra={
                    "event": VLLM_WAKE_FAIL,
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "duration_ms": total_duration_ms,
                    "wake_post_duration_ms": wake_duration_ms,
                    "error_type": "PostWakeHealthTimeout",
                    "error_msg": (
                        f"/health did not return 200 within "
                        f"{self._SLEEP_WAKE_TIMEOUT_S}s after /wake_up"
                    ),
                },
            )
            raise RuntimeError(
                f"vllm post-wake /health did not return 200 within "
                f"{self._SLEEP_WAKE_TIMEOUT_S}s"
            )
        log.info(
            "vllm engine awake + healthy (snapshot-restored) for %s",
            self.SERVED_MODEL_NAME,
            extra={
                "event": VLLM_WAKE_OK,
                "served_model_name": self.SERVED_MODEL_NAME,
                "duration_ms": total_duration_ms,
                "wake_post_duration_ms": wake_duration_ms,
            },
        )

        # Phase 2 R6: emit a cold-start budget event covering the full
        # container-process lifetime (module import → engine awake). The
        # research targets <30s snapshot-warmed cold start; this event
        # is the one observable an operator can grep for to assess the
        # success criteria after each Phase 3 iteration.
        total_elapsed_ms = (time.monotonic() - _CONTAINER_START_MONOTONIC) * 1000.0
        log.info(
            "vllm cold-start complete for %s (total %.1f ms since container start)",
            self.SERVED_MODEL_NAME,
            total_elapsed_ms,
            extra={
                "event": VLLM_COLDSTART_COMPLETE,
                "served_model_name": self.SERVED_MODEL_NAME,
                "total_elapsed_ms": total_elapsed_ms,
            },
        )

        # Phase 2 R5: subprocess health monitor. vLLM issue #19849 — the
        # parent Modal container does not notice EngineCore_DP0 dying
        # inside the ``vllm serve`` subprocess (it survives as a zombie
        # of sorts: TCP listener up, /health 5xx, /v1/chat/completions
        # hangs or 5xxs). Without this thread we have no log signal
        # between requests, only "next user request 500s with no
        # actionable trace". The thread polls /health every 30s and
        # emits VLLM_HEALTH_DEGRADED on any non-2xx or connection error.
        # daemon=True ensures container shutdown is not blocked.
        self._health_stop_event = threading.Event()
        self._health_thread = threading.Thread(
            target=self._poll_vllm_health,
            args=(self._health_stop_event,),
            daemon=True,
            name=f"vllm-health-{self.SERVED_MODEL_NAME}",
        )
        self._health_thread.start()

    def _poll_vllm_health(self, stop_event: Any) -> None:
        """Background poll of vllm serve /health (Phase 2 R5).

        Runs every 30s until the stop_event is set (in ``_stop``).
        Emits VLLM_HEALTH_DEGRADED on any non-2xx HTTP response or
        connection error — the operator's only signal that the
        subprocess died between requests.

        Errors inside this thread are swallowed (a failing health-poll
        thread MUST NOT crash the container); the absence of regular
        health events in cold-start logs is itself the canary.
        """
        import httpx
        from obs.events import VLLM_HEALTH_DEGRADED
        from obs.logging import get_logger

        log = get_logger(f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}.health")
        health_url = f"{self._VLLM_BASE_URL}/health"
        poll_interval_s = 30.0

        while not stop_event.is_set():
            try:
                r = httpx.get(health_url, timeout=5.0)
                if r.status_code >= 300:
                    log.warning(
                        "vllm /health returned %s for %s",
                        r.status_code,
                        self.SERVED_MODEL_NAME,
                        extra={
                            "event": VLLM_HEALTH_DEGRADED,
                            "served_model_name": self.SERVED_MODEL_NAME,
                            "status_code": r.status_code,
                            "error_type": "HTTPStatus",
                            "error_msg": f"/health returned {r.status_code}",
                        },
                    )
            except Exception as err:
                log.warning(
                    "vllm /health poll failed for %s: %s",
                    self.SERVED_MODEL_NAME,
                    err,
                    extra={
                        "event": VLLM_HEALTH_DEGRADED,
                        "served_model_name": self.SERVED_MODEL_NAME,
                        "status_code": 0,
                        "error_type": type(err).__name__,
                        "error_msg": str(err),
                    },
                )
            stop_event.wait(poll_interval_s)

    @modal.exit()
    def _stop(self) -> None:
        """Tear down the sidecar and the vllm serve subprocess."""
        import subprocess

        # Phase 2 R5: stop the background /health poller before tearing
        # down the subprocess so the thread doesn't log spurious
        # connection-refused events during shutdown.
        stop_event = getattr(self, "_health_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        _stop_qrng_tunnel(getattr(self, "_cloudflared", None))
        proc = getattr(self, "_vllm_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    @modal.web_server(port=_VLLM_PORT, startup_timeout=_STARTUP_TIMEOUT_S)
    def serve(self) -> None:
        """Modal proxies inbound HTTP traffic to ``vllm serve`` on port 8000.

        The subprocess is already running by the time this function is
        invoked (started in ``_start_and_sleep``, woken in ``_wake``).
        @modal.web_server returns no app — Modal handles the proxy
        itself; this function body is only entered for Modal's
        port-readiness probe.
        """
        return None


# ----- QRNG cloudflared sidecar wiring -------------------------------------
#
# Both per-model classes share the same sidecar bootstrap. We import the
# sidecar module lazily inside the helpers so this file stays importable
# at deploy-host introspection time (where qr_sampler is on the path but
# the cloudflared binary is not).


def _start_qrng_tunnel(served_model_name: str) -> Any:
    """Spawn the cloudflared sidecar for one VllmQr* container — soft-fail.

    The sidecar listens on 127.0.0.1:50051 and forwards through Cloudflare
    Access to ``QRNG_TUNNEL_HOSTNAME``. The qr-sampler ``QuantumGrpcSource``
    dials the loopback address (set via ``QR_GRPC_SERVER_ADDRESS`` in the
    Dockerfile, overridable via the qr-sampler-prod Modal Secret).

    Failure is SOFT: when the Cloudflare Access service token env vars are
    missing/revoked, when cloudflared cannot reach the tunnel hostname, or
    when the binary is otherwise unavailable, we log a structured event
    and return ``None``. The caller stores ``None`` as ``self._cloudflared``;
    request-time entropy then falls back through
    ``qr_sampler.entropy.FallbackEntropySource`` (configured via the
    ``QR_FALLBACK_MODE=system`` env var baked into Dockerfile.vllm) to
    ``SystemEntropySource`` (``os.urandom``). The per-request
    ``entropy.degraded`` events surface the QRNG outage to operators.

    Prior design hard-failed here, which killed the container AFTER the
    engine had successfully built — wasting ~5 min of GPU per attempted
    cold-start AND producing a misleading
    ``Engine core proc EngineCore_DP0 died unexpectedly`` shutdown
    cascade (atexit join on the EngineCore subprocess receives
    KeyboardInterrupt during Python interpreter shutdown).

    Returns:
        ``CloudflaredSidecar`` on success.
        ``None`` when the sidecar could not start (any reason).
    """
    from obs.events import (
        CLOUDFLARED_CONFIG_MISSING,
        CLOUDFLARED_SIDECAR_FAILED,
        CLOUDFLARED_SIDECAR_SKIPPED,
    )
    from obs.logging import get_logger

    from qr_sampler.connectors.modal.cloudflared_sidecar import (
        CloudflaredConfig,
        CloudflaredConfigError,
        CloudflaredSidecar,
    )

    log = get_logger("qr_sampler.cloudflared")
    log.info(
        "Starting QRNG cloudflared sidecar for %s container",
        served_model_name,
        extra={"event": "cloudflared.container_start", "model": served_model_name},
    )

    try:
        config = CloudflaredConfig.from_env()
    except CloudflaredConfigError as err:
        # Parse the "unset or empty: A, B, C" tail from the error message
        # so the structured event surfaces actionable missing-var names.
        msg = str(err)
        missing: list[str] = []
        marker = "unset or empty: "
        if marker in msg:
            tail = msg.split(marker, 1)[1].split(".", 1)[0]
            missing = [v.strip() for v in tail.split(",") if v.strip()]
        log.warning(
            "QRNG cloudflared sidecar config missing for %s; "
            "container will serve with system-entropy fallback (%s)",
            served_model_name,
            ", ".join(missing) if missing else str(err),
            extra={
                "event": CLOUDFLARED_CONFIG_MISSING,
                "served_model_name": served_model_name,
                "missing_vars": missing,
                "error_msg": str(err),
            },
        )
        log.info(
            "QRNG cloudflared sidecar skipped for %s "
            "(falling back to system entropy via FallbackEntropySource)",
            served_model_name,
            extra={
                "event": CLOUDFLARED_SIDECAR_SKIPPED,
                "served_model_name": served_model_name,
                "reason": "config_missing",
            },
        )
        return None

    sidecar = CloudflaredSidecar(config)
    try:
        sidecar.start()
    except Exception as err:
        # Cloudflared binary missing from PATH, tunnel unreachable,
        # service-token revoked, etc. Same soft-fail policy.
        stderr_tail = None
        with contextlib.suppress(Exception):
            stderr_tail = "\n".join(list(sidecar._stderr_tail)[-30:])
        log.warning(
            "QRNG cloudflared sidecar failed to start for %s: %s: %s; "
            "container will serve with system-entropy fallback",
            served_model_name,
            type(err).__name__,
            err,
            extra={
                "event": CLOUDFLARED_SIDECAR_FAILED,
                "served_model_name": served_model_name,
                "error_type": type(err).__name__,
                "error_msg": str(err),
                "stderr_tail": stderr_tail,
            },
        )
        log.info(
            "QRNG cloudflared sidecar skipped for %s (falling back to system entropy)",
            served_model_name,
            extra={
                "event": CLOUDFLARED_SIDECAR_SKIPPED,
                "served_model_name": served_model_name,
                "reason": "startup_failed",
            },
        )
        # Best-effort cleanup of any partially-started sidecar process.
        with contextlib.suppress(Exception):
            sidecar.stop()
        return None

    return sidecar


def _stop_qrng_tunnel(sidecar: Any) -> None:
    """Tear down the cloudflared sidecar, tolerating an unset attribute.

    The attribute is ``None`` when ``start_tunnel`` raised before assigning —
    in that case there is nothing to stop. We log either branch so an
    operator reading ``modal app logs`` sees the container shutdown sequence.
    """
    import logging

    log = logging.getLogger("qr_sampler.cloudflared")
    if sidecar is None:
        log.info(
            "QRNG cloudflared sidecar was not running; nothing to stop",
            extra={"event": "cloudflared.stop_skipped"},
        )
        return
    sidecar.stop()


# ----- Open WebUI container ------------------------------------------------
#
# qr-llm-chat split (plan R1-R6, requirements §10, spec §11): the Open WebUI
# surface is now declared *here* in the same Modal app as the two vLLM
# classes, so `modal deploy -m qr_sampler.connectors.modal.app` brings up
# all three services as one unit. The OWUI-specific bootstrap (admin user,
# lifespan hooks, Function envelope import, valve writer) lives in the
# separate `qr-llm-chat` Python package and is invoked via a single thin
# entrypoint module — `qr_llm_chat.modal_entrypoint` — kept out of this
# file so qr-sampler tests do not depend on qr-llm-chat being installed.
#
# Image construction approach (impl decision, R2):
#   Option A — add_local_python_source("qr_llm_chat", copy=True). Resolves
#     the package at deploy time via Python's import machinery. Requires
#     `pip install -e <qr-llm-chat>` on the deploy host. ← chosen.
#   Option B — .run_commands("pip install qr-llm-chat==X.Y") off a wheel.
#     Useful once qr-llm-chat publishes wheels; not relevant during dev.
#   Option C — .add_local_dir(qr_llm_chat_root, "/repo").run_commands(
#     "pip install -e /repo"). More verbose but works without a sibling-
#     repo editable install on the deploy host.
# Option A is the lowest-friction choice that matches how the qr_sampler
# source itself is shipped (line above this block). R3 reorganises
# qr-llm-chat into the canonical src/qr_llm_chat/ layout so Option A
# resolves cleanly.

_OWUI_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libpq-dev", "build-essential", "curl")
    .pip_install(
        "open-webui==0.9.5",
        # OWUI 0.9.5 transitively pins bcrypt==5.0.0; widening from
        # <5 to <6 lets pip resolve the image build (4.x's Python API
        # is unchanged in 5.x, so qr_llm_chat.admin_bootstrap works
        # against either).
        "bcrypt>=4.1,<6",
        "psycopg[binary]>=3.1,<4",
        # OWUI imports `pgvector.sqlalchemy` at module load when
        # VECTOR_DB=pgvector (set in qr-llm-chat-prod). Pin matches
        # what OWUI 0.9.5's pgvector_client.py was developed against.
        "pgvector>=0.3,<0.4",
        # OWUI 0.9.5's pgvector client uses psycopg2 (v2 driver) even
        # though admin_bootstrap goes through SQLAlchemy. Without this
        # `import open_webui.main` fails on the pgvector branch with
        # ModuleNotFoundError("No module named 'psycopg2'").
        "psycopg2-binary>=2.9,<3",
    )
    .env(
        {
            # Block OWUI's import-time network probes — without these set,
            # `import open_webui.main` reaches out to localhost:11434 and
            # huggingface.co at module load, which would freeze open TCP
            # sockets into the memory snapshot. See qr-llm-chat plan
            # "Step: OWUI lifespan hooks and SvelteKit patch".
            "ENABLE_OLLAMA_API": "false",
            "OLLAMA_BASE_URLS": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # Disable OWUI's RAG embedding model loader. OWUI 0.9.5 defaults
            # to ``sentence-transformers/all-MiniLM-L6-v2`` and tries to
            # snapshot_download() it at startup -- which fails noisily under
            # ``HF_HUB_OFFLINE=1`` with a multi-page traceback. We do not
            # use RAG/document retrieval in this deploy, so an empty model
            # name takes the falsy short-circuit in
            # ``open_webui.routers.retrieval.get_ef`` and skips the loader.
            #
            # The env var alone is insufficient when OWUI's ``config`` row
            # already holds a DB-persisted ``rag.embedding_model`` from an
            # earlier deploy: ``PersistentConfig`` prefers the DB value over
            # env. qr-llm-chat's ``lifespan_hooks._strip_stale_rag_embedding_config``
            # strips that DB key in pre_snapshot before OWUI is imported so
            # this env value actually wins.
            "RAG_EMBEDDING_MODEL": "",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_python_source("qr_sampler", copy=True)
    .add_local_python_source("qr_llm_chat", copy=True)
    # `obs` is a top-level package living at qr-llm-chat's repo root
    # (not under src/). Five qr_llm_chat modules import from `obs.events`
    # / `obs.logging`; without shipping it the snapshot pre-import fails
    # with ModuleNotFoundError("No module named 'obs'").
    .add_local_python_source("obs", copy=True)
    # Function envelope JSON bundles -- `add_local_python_source` above
    # only ships `.py`/`.pyi` files, so the JSON envelopes in
    # `src/qr_llm_chat/functions/*.json` (consumed by
    # `qr_llm_chat.bootstrap_functions._load_bundles`) never reach the
    # container without this explicit dir add. Symptom of regression:
    # ``no Function bundles found under /root/qr_llm_chat/functions``
    # in `modal app logs` at restore, followed by "no models available"
    # in the OWUI UI.
    #
    # We resolve the on-disk location via `importlib.util.find_spec` so
    # the path stays correct whether qr-llm-chat is installed as an
    # editable (`pip install -e`) sibling repo or a published wheel --
    # `add_local_python_source` itself uses the same resolver, so
    # whatever path it ships .py files FROM is the path we ship the JSON
    # files from too. `remote_path` is pinned to match the resolved
    # location of `qr_llm_chat/functions/` inside the container --
    # `add_local_python_source(..., copy=True)` lands the package under
    # `/root/qr_llm_chat/` (visible in the log path quoted above), so
    # the JSON sibling goes there too.
    .add_local_dir(
        str(_qr_llm_chat_functions_dir()),
        remote_path="/root/qr_llm_chat/functions",
        copy=True,
    )
)

# OWUI-specific secret — separate from qr_sampler_prod_secret per spec §11.6.
# Holds OWUI admin seed, Neon DSN, OAuth client creds, SMTP creds, etc.
# Provisioned via `modal secret create qr-llm-chat-prod ...`; documented in
# qr-llm-chat/infra/modal_secrets.md.
qr_llm_chat_secret = modal.Secret.from_name("qr-llm-chat-prod")

# Persistent storage for OWUI's uploads / cached config under /data.
# OWUI 0.9.5 keeps SQLite at /data/webui.db by default; we override with
# a Neon DSN via OWUI_DATABASE_URL in the secret, but the Volume still
# holds chat-attachment uploads and any future on-disk state.
owui_data_volume = modal.Volume.from_name("qr-llm-chat-data", create_if_missing=True)


@app.cls(
    image=_OWUI_IMAGE,
    cpu=1.0,
    memory=2048,
    secrets=[qr_sampler_prod_secret, qr_llm_chat_secret],
    volumes={"/data": owui_data_volume},
    enable_memory_snapshot=True,
    scaledown_window=1800,  # 30 min idle — see qr-llm-chat spec §3.5 cost note.
    timeout=60 * 60,
    min_containers=0,  # NEVER set ≥1 — keep-warm cost is unacceptable (Q10).
    # Hard cap at 1 replica. OWUI keeps in-memory state (PersistentConfig
    # cache, session cookies, lazy admin-user check) that must be
    # authoritative; fan-out would create N caches racing against each
    # other and waste snapshot-restore cycles on every replica. The SPA
    # cold-load burst (~15-25 parallel requests) is absorbed by raising
    # ``max_inputs`` below, not by horizontal scaling.
    max_containers=1,
)
# ``max_inputs`` is the per-container in-flight request budget. Set high
# enough that a SvelteKit SPA cold-load (manifest + OAuth probes + model
# list + config + prompts + knowledge + tools, fired in parallel) fits
# inside one container's queue. Anything below ~32 will trigger
# autoscaling pressure even though we cap at one replica.
@modal.concurrent(max_inputs=64)
class OWUIService:
    """Open WebUI ASGI surface for the qr-llm-chat split.

    Lifecycle (mirrors the qr-llm-chat lifespan_hooks.py contract):

    * ``@modal.enter(snap=True)`` — ``pre_snapshot`` imports the OWUI
      package eagerly so its module tree is fully populated before Modal
      freezes the container memory image. No DB connections, no outbound
      sockets opened in this phase (see ``HF_HUB_OFFLINE`` / ``OLLAMA_*``
      env above).
    * ``@modal.enter(snap=False)`` — ``post_restore`` ensures the admin
      user exists (idempotent), imports the bundled Function envelopes
      (Filter + Pipe) via OWUI's Python API, and writes the comparison
      Pipe's valves so its ``vllm_base_url`` / ``model_base_urls`` point
      at the ``VllmQrQwen`` web URL above (the sibling ``VllmQrGemma``
      class is paused — see module docstring).
    * ``@modal.asgi_app() serve`` — returns the OWUI FastAPI app.

    All heavy lifting is in the ``qr_llm_chat`` package; this class is
    intentionally thin so the OWUI lifecycle and the Modal class lifecycle
    stay one-to-one.
    """

    @modal.enter(snap=True)
    def _pre_snapshot(self) -> None:
        from qr_llm_chat.modal_entrypoint import pre_snapshot

        pre_snapshot()

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        from qr_llm_chat.modal_entrypoint import build_owui_asgi_app

        self._asgi_app = build_owui_asgi_app()

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app
