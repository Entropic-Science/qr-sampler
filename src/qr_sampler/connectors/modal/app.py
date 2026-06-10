"""Modal app definition — vllm-qr (H200, per-model containers).

Layout (matches spec.md §5.5 / §4.1, with the labs-cutover per-model split):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQrQwen         — @app.cls (H100:1) running lovedheart/Qwen3.5-9B-FP8 alone

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

2. ``VllmQrQwen`` currently serves lovedheart/Qwen3.5-9B-FP8 (community
   FP8 e4m3 quant of Qwen/Qwen3.5-9B, ~9 GiB resident weights, served
   as ``qwen3.5-9b``). vLLM auto-detects the FP8 quantization config
   from the model's ``config.json`` so no explicit ``--quantization fp8``
   flag is needed on the ``vllm serve`` command line. iter-15 (2026-05-24)
   swapped here from Qwen/Qwen3.6-27B-FP8 (~27 GiB) to optimise
   cold-from-storage latency for a public-facing demo: ``/sleep level=1``
   keeps weights resident in CPU RAM, so the snapshot's cold-storage
   pull is bandwidth-bound by weight tensor size (~115 s for 27B, projected
   ~40 s for 9B). The bf16 build of the same 9B model (~18 GiB) is the
   official Qwen-org publication; no Qwen-org FP8 build exists for the
   3.5-9B size at deploy time, so the ``lovedheart`` community quant is
   the only path to a ≤10 GiB resident footprint. Both 9B and 27B Qwen3.* variants carry
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
from typing import Any, Final

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
MODEL_HF_REPO_ID = "lovedheart/Qwen3.5-9B-FP8"  # community FP8 e4m3 build of Qwen3.5-9B (~9 GiB resident, vs ~18 GiB for the official bf16 build, vs ~27 GiB for the prior Qwen3.6-27B-FP8). Demo-grade: 3x smaller cold-from-storage payload than the 27B pin → projected snapshot restore ~40 s vs 115-125 s. Architecture is Qwen3_5ForConditionalGeneration so the existing transformers==5.5.4 + MM-probe monkey-patch combo applies unchanged. No official Qwen-org FP8 build of 3.5-9B exists at deploy time (2026-05-24); ``lovedheart`` is the only published FP8 quant. If a future Qwen-org FP8 build appears, switch back to ``Qwen/<...>``.
MODEL_SERVED_NAME = "qwen3.5-9b"  # /v1/models id; lockstep with qr-llm-chat _QWEN_ID (precision is implementation detail, not picker-visible)

# iter-14: Snapshot identity hardening. The HF revision used to default to
# os.environ.get("QWEN_REVISION", "") which (a) silently resolves to "latest"
# (whatever main currently points at — changes without our deploys), and (b)
# makes an env var part of the snapshot input surface. Both are sources of
# the brittle cold-cold restore behaviour iter-09..iter-13 chased. We now
# hardcode the SHA. To bump: look up the current commit on
# https://huggingface.co/lovedheart/Qwen3.5-9B-FP8/commits/main, paste the full
# 40-char hex here, and bump SNAPSHOT_IDENTITY_VERSION below in lockstep.
# An empty string here still falls through (so dev iteration on environments
# without the SHA available is unblocked), but the predeploy.ps1 gate in
# qr-llm-chat refuses to deploy until this is populated.
MODEL_REVISION: str = "5d77dcb2e2c606bc261b5b8e946a67781f18d733"  # lovedheart/Qwen3.5-9B-FP8 main as of 2026-05-24 via HF API

# iter-14: Snapshot identity version. Modal computes the snapshot key from
# the image hash + the @modal.enter(snap=True) body. This constant is NOT
# read by Modal directly; it is logged on every cold-start so an operator
# can grep ``modal app logs`` and confirm "the cold-start I am looking at
# belongs to identity version X". Bump this string when an intentional
# snapshot-bust is needed (e.g. flipping a snap=True kwarg, or after
# upgrading vllm/transformers). When this bumps but the image digest does
# not, the cold-start log will show two events with the same
# image_digest + different identity_version — which is the operator's
# signal that the snapshot needs to be re-materialised.
#
# Format: "iter-NN-MMM" where NN is the iteration log number and MMM is a
# monotonic sub-iteration counter inside that iter.
SNAPSHOT_IDENTITY_VERSION: Final[str] = "iter-15-001-qwen3.5-9b-fp8-demo"

MODEL_GPU_MEMORY_UTILIZATION = "0.85"  # iter-15: bumped from 0.8 (which was tight for 27B-FP8) to 0.85 with the 9B-FP8 swap. 9 GiB weights leave plenty of headroom under H100's 79 GiB even at 0.85 utilization; the extra 5 % expands the KV cache budget so multi-turn chats stay coherent through a longer demo session without prefix caching (which is the one knob we can't re-enable per auto-memory iter14_snapshot_load_working).

# ---- iter-17 NEW PROFILE: PrismaQuant on vLLM v0.20.0 ---------------------
# Parallel constants block for the experimental PrismaQuant profile. The
# constants above (MODEL_*) drive VllmQrQwen on vLLM 0.17; these
# (PRISMAQUANT_*) drive VllmQrPrismaQuant on vLLM 0.20. Each class reads
# its own block, so an edit to one profile does NOT invalidate the
# other's snapshot identity. To swap the PrismaQuant build:
#   1. Edit PRISMAQUANT_MODEL_HF_REPO_ID / _REVISION below.
#   2. modal run -m qr_sampler.connectors.modal.app::download_weights_prismaquant
#   3. Bump PRISMAQUANT_SNAPSHOT_IDENTITY_VERSION in lockstep so the prior
#      snapshot identity is invalidated cleanly.
#   4. modal deploy.
#
# Why a separate profile (vs editing the MODEL_* block):
#   * vLLM 0.20 ships a strictly larger compressed-tensors scheme set
#     than 0.17 — required for PrismaQuant's mixed NVFP4 + MXFP8 weights.
#     vLLM 0.17 raised ``NotImplementedError: No compressed-tensors
#     compatible scheme was found`` on engine init (iter-16 traceback
#     captured in artifacts/iter-16-coldstart.log).
#   * vLLM major upgrades carry collateral risk: PyTorch baseline +
#     CUDA toolkit + the monkey-patches in vllm_serve.py / vllm_patches.py.
#     Isolating the upgrade in a SECOND class lets the iter-15 9B-FP8
#     stack stay a known-good fallback while we validate 0.20.
PRISMAQUANT_MODEL_HF_REPO_ID = "rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm"
PRISMAQUANT_MODEL_SERVED_NAME = "qwen3.6-27b-prismaquant"
PRISMAQUANT_MODEL_REVISION: Final[str] = "09de726107c7f9c6b44e34c28541579f0b73a719"
# iter-54: bumped for the commit-then-fetch entropy pipelining — the
# prefetch ticket machinery lives in the snapshotted EngineCore process
# (logits processor + QuantumGrpcSource), so the prior snapshot must be
# re-materialised rather than restored over the new code.
PRISMAQUANT_SNAPSHOT_IDENTITY_VERSION: Final[str] = "iter-54-001-prefetch-pipeline"
PRISMAQUANT_GPU_MEMORY_UTILIZATION = "0.8"  # operator override of recipe's 0.90; demo doesn't use MTP so no need to grow KV budget — keep headroom for cuBLAS + FlashInfer NVFP4 workspaces.

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


def _qr_llm_chat_static_assets_dir() -> Path:
    """Resolve the on-disk location of `qr_llm_chat/static_assets/` at deploy time.

    Twin of ``_qr_llm_chat_functions_dir`` — see that function for the
    full container-restore tolerance rationale. The static assets
    directory holds the splash overlay CSS / JS / SVG that
    ``qr_llm_chat.bootstrap_static_assets`` copies into OWUI's
    ``STATIC_DIR``. The pyproject.toml package-data entry
    (``qr_llm_chat.static_assets = ["*.css", "*.js", "*.svg"]``) makes
    importlib.resources see the files in an editable install, but
    Modal's ``.add_local_python_source("qr_llm_chat", copy=True)`` only
    ships ``.py`` / ``.pyi`` source files — so the non-Python assets
    never reach the container without this explicit ``.add_local_dir``
    layer. Symptom of regression: ``bootstrap_static_assets done:
    wrote=0 skipped=0 missing=3`` in OWUI logs at restore (iter-17c).
    """
    spec = importlib.util.find_spec("qr_llm_chat")
    in_modal_container = bool(os.environ.get("MODAL_TASK_ID"))
    if spec is None or spec.origin is None:
        if in_modal_container:
            return Path("/__qr_llm_chat_static_assets_unavailable_in_container__")
        raise RuntimeError(
            "qr_llm_chat is not importable in the deploy host's Python env. "
            "Run `pip install -e <path/to/qr-llm-chat>` in the venv you use "
            "for `modal deploy` and try again."
        )
    static_assets_dir = Path(spec.origin).resolve().parent / "static_assets"
    if not static_assets_dir.is_dir():
        if in_modal_container:
            return Path("/__qr_llm_chat_static_assets_unavailable_in_container__")
        raise RuntimeError(
            f"Expected `qr_llm_chat/static_assets/` at {static_assets_dir}; not found. "
            "If the package layout has changed, update _qr_llm_chat_static_assets_dir "
            "in qr_sampler/connectors/modal/app.py."
        )
    return static_assets_dir


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

# iter-14b finding (2026-05-23, reverted): An earlier iter-14 revision
# baked weights into the image at /opt/models/qwen3.6-27b-fp8 via a
# `.run_commands(["hf download ..."], secrets=[hf_token_secret])` layer.
# Empirically this BROKE memory-snapshot restore: the 27 GB of weights
# mmap'd into the container's address space inflated the GPU-state set
# the cuda-checkpoint enumerator has to walk during snapshot capture,
# pushing capture time over Modal's 180 s `CUDA_CHECKPOINT_TIMEOUT`.
# Both iter-14b cold-starts ran the full `_start_and_sleep` path (no
# snapshot reuse) and the second one's capture window was 201 s.
#
# Modal's official guidance — and their `lfm_snapshot.py` reference
# example for the exact same pattern (vLLM + sleep mode + GPU snapshot)
# — explicitly puts weights on a Volume mount:
#   > "GPU Memory Snapshots do not speed up model loading from storage."
#   > Snapshots should target "library initialization (imports) and JIT
#   > compilation."
# So weights belong on the `llm-weights` Volume (already populated by
# `download_weights`), and the snapshot captures only Python imports +
# torch.compile artefacts + CUDA graphs.
#
# This revert KEEPS all other iter-14 work:
#   * MODEL_REVISION pinned to a specific HF SHA (predeploy gate enforced)
#   * SNAPSHOT_IDENTITY_VERSION constant + `vllm.snapshot.identity` event
#   * `vllm.snapshot.restore_class` event (threshold raised to 900 s)
#   * predeploy.ps1 + deploy.ps1 + JSON-bundle regeneration discipline
#   * OWUI progressive status indicator
# The image-baked-weights idea is preserved in git history; do NOT
# resurrect without first reading Modal's snapshot docs again.
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

# iter-17: parallel image for the PrismaQuant profile, built from
# Dockerfile.vllm-prismaquant (FROM vllm/vllm-openai:v0.20.0). The two
# images are bytewise different — different vLLM major, different env
# block (PrismaQuant pins VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass) —
# so Modal hashes them to distinct digests and each profile gets its
# own snapshot identity. Modal Volume mounts (llm-weights, the
# cloudflared sidecar setup, the qr-sampler + obs source layers)
# mirror the iter-15 image.
vllm_prismaquant_image = (
    modal.Image.from_dockerfile(
        str(Path(__file__).parent / "Dockerfile.vllm-prismaquant"),
        context_dir=str(_REPO_ROOT),
    )
    .add_local_python_source("qr_sampler", copy=True)
    .add_local_python_source("obs", copy=True)
)

# iter-17: parallel torch.compile cache. vLLM 0.17 and 0.20 emit
# incompatible AOT artefacts (different vLLM internal IR + different
# PyTorch baseline) — sharing the cache between the two profiles would
# cause vLLM 0.20 to either miss the cache (best case, slow first-load)
# or crash trying to load 0.17-format binaries (worst case). Separate
# Volume isolates them cleanly. Same per-version cache-key discipline
# as the iter-14 vllm-cache Volume (see comment on vllm_cache_volume).
vllm_prismaquant_cache_volume = modal.Volume.from_name(
    "vllm-cache-prismaquant", create_if_missing=True
)

# --- App -------------------------------------------------------------------

app = modal.App(APP_NAME)


# ----- One-shot weights download -------------------------------------------


# 2026-05-24 (iter-15): lovedheart/Qwen3.5-9B-FP8 is the active build target. Both 9B and
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
    ``_QWEN_REVISION``). As of 2026-05-24 (iter-15) this targets lovedheart/Qwen3.5-9B-FP8
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


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": weights_volume},
    secrets=[hf_token_secret],
    timeout=60 * 60,  # 1 hour
)
def download_weights_prismaquant() -> dict[str, str]:
    """Populate the ``llm-weights`` Volume with the PrismaQuant model.

    iter-17 parallel to ``download_weights``. Shares the same
    ``llm-weights`` Volume (HF caches files by repo_id under
    ``~/.cache/huggingface/hub/`` so the two profiles' weights live in
    separate subdirectories — no collision). Run once per model bump:

        modal run -m qr_sampler.connectors.modal.app::download_weights_prismaquant

    Idempotent. Pins the SHA from ``PRISMAQUANT_MODEL_REVISION`` for
    snapshot-identity hygiene; ``snapshot_download`` validates the
    revision matches the cache and re-pulls only if the local files are
    stale.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    kwargs: dict[str, Any] = {"repo_id": PRISMAQUANT_MODEL_HF_REPO_ID}
    if PRISMAQUANT_MODEL_REVISION:
        kwargs["revision"] = PRISMAQUANT_MODEL_REVISION

    path = snapshot_download(**kwargs)
    weights_volume.commit()  # type: ignore[attr-defined]

    return {
        "prismaquant_path": path,
        "prismaquant_revision": PRISMAQUANT_MODEL_REVISION or "(latest)",
    }


# ----- Per-model GPU classes -----------------------------------------------
#
# Each class:
#   * Loads ONE model into ONE H200 container.
#   * Scales to zero independently via `scaledown_window=300` (5 min idle).
#     iter-39 bumped 180→300 after the operator caught GPU containers
#     incurring cost between back-to-back demo questions. Once the
#     scaledown fires Modal calls `@modal.exit()` and TERMINATES the
#     container (releasing the GPU back to the pool); the warm-restore
#     comes from the memory snapshot, not from a persisted container.
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
    # iter-14b revert: weights_volume mount RESTORED. Putting weights on
    # the Volume (not in the image) is Modal's recommended pattern for
    # vLLM + GPU memory snapshots — see `lfm_snapshot.py` example and the
    # docs quote: "GPU Memory Snapshots do not speed up model loading
    # from storage." Volume-mounted weights keep snapshot capture
    # focused on imports + torch.compile artefacts (small, fast to
    # enumerate via cuda-checkpoint), avoiding the 180s timeout that
    # image-baked weights triggered.
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
    # iter-39 (2026-05-25): 5 min idle -> shutdown (bumped from 3 min).
    # User caught a B200 PrismaQuant container running between demo
    # questions and accruing GPU cost; the prior 3-min window was
    # tight enough that quick back-to-back prompts kept the container
    # alive but slightly-longer pauses landed in the awkward 3-5 min
    # window where the container had JUST scaled down and the next
    # prompt paid another restore. 5 min gives the demo session a
    # comfortable "single train of thought" buffer.
    #
    # ENDING SEMANTICS (Modal documentation, §Lifecycle):
    # - When scaledown_window expires with no in-flight requests,
    #   Modal calls the class's ``@modal.exit()`` handler (our
    #   _stop method, which kills the cloudflared sidecar + vllm
    #   serve subprocess), then TERMINATES the container.
    # - Terminating the container RELEASES the GPU back to Modal's
    #   pool — billing stops. enable_memory_snapshot=True preserves
    #   the warm snapshot for fast next start, but the snapshot
    #   itself is not GPU-resident.
    # - max_containers=1 below caps the per-class replica count at
    #   one — Modal will NEVER stand up a second container while
    #   the first is idle. Combined with min_containers omitted
    #   (defaults to 0, no warm-pinned replicas), this guarantees
    #   that an idle period beyond 300 s always results in zero
    #   running containers for the class.
    "scaledown_window": 300,
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
#
# iter-45 (2026-05-25): VllmQrQwen PAUSED — the @app.cls + @modal.concurrent
# decorators are commented out so Modal does NOT spawn this function on
# deploy (saves the per-deploy GPU mount + image-pull cost). The class
# body is retained so a future re-introduction of Qwen 9B in the picker
# is a two-line uncomment. Mirrors the prior Gemma-paused pattern (see
# module docstring + the ``VllmQrGemma`` reference). The qr-llm-chat
# OWUI side (``qr_llm_chat.bootstrap_connections``) no longer reads
# ``VLLM_QR_QWEN_URL``, no longer seeds a Connection for ``qwen3.5-9b``,
# and no longer registers the dual-lane pseudo-model — the only picker
# entries after iter-45 are the two Qwen3.6-27B-PrismaQuant variants.
#
# @app.cls(**_CLS_KWARGS)            # PAUSED iter-45
# @modal.concurrent(max_inputs=8)    # PAUSED iter-45
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
    # ("qwen3.5-9b (quantum-random)") is set via an OWUI ``model`` table
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

        # iter-53 (2026-06-09): raise the adaptive-timeout floor for the
        # tunnel reality. The library default (5 ms) is tuned for a
        # localhost QRNG; here every fetch crosses the cloudflared
        # tunnel, where the observed steady-state P99 is ~96 ms with
        # tail spikes past 200 ms. With the old floor the adaptive
        # ceiling converged to ~145 ms and killed every tail fetch
        # (all observed flap timeouts were 144-229 ms — a 300 ms floor
        # prevents the entire class). setdefault so the qr-sampler-prod
        # Modal Secret can still override per-deploy.
        env.setdefault("QR_CB_MIN_TIMEOUT_MS", "300")
        # iter-53b: shorten the breaker's half-open cadence. The library
        # default (10 s) assumes remote-outage recovery is slow; here the
        # common cause is the post-wake stale channel, which the
        # half-open path now fixes in one cycle — at ~2.6 tok/s a 10 s
        # window still costs ~26 PRNG tokens, a 3 s window costs ~8.
        env.setdefault("QR_CB_RECOVERY_WINDOW_S", "3")

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
            # iter-14b revert: pass HF_REPO_ID. vLLM resolves this against
            # the local HF cache mounted at /root/.cache/huggingface
            # (Volume-backed, pre-populated by `download_weights`), so
            # there is NO HF Hub network round-trip on cold-start when
            # the Volume is warm. The MODEL_REVISION pin (see top of
            # file) ensures vLLM loads the exact same commit each time.
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
            # Precision: weights are FP8 e4m3 (the model is
            # ``lovedheart/Qwen3.5-9B-FP8``, a community FP8 quant of
            # Qwen/Qwen3.5-9B; vLLM auto-detects the FP8
            # ``quantization_config`` in the model's ``config.json``, so no
            # explicit ``--quantization fp8`` flag is needed on the serve
            # command line). KV cache stays at vLLM's default (bf16), which
            # keeps the ``init_fp8_kv_scales`` patch in ``vllm_patches.py``
            # dormant — its own ``cache_dtype.startswith("fp8")`` gate
            # short-circuits when the cmd does not request ``--kv-cache-dtype
            # fp8``. FP8 weights (~9 GiB) + bf16 KV cache (bounded by
            # max-num-seqs=4 × max-model-len=32768) + activation fits
            # comfortably under the 73 GiB usable budget on the H100:1 with
            # the gpu-memory-utilization=0.85 ceiling below.
            # iter-14d (2026-05-23): --enable-prefix-caching DROPPED.
            # The 2026-05-22 A/B test that previously kept it on was
            # against the IMAGE-BAKED weights path, which had its own
            # bigger snapshot-state problem (now reverted in iter-14c).
            # With the Volume-mounted-weights baseline restored, the
            # documented unreliable triad (custom logits processors +
            # FP8 + prefix caching, auto-memory ``modal_snapshot_tuning``)
            # is back in play and prefix caching is the easiest leg to
            # disable without losing the QR sampling pipeline. iter-14c
            # confirmed every cold-cold runs snap=True (no snapshot
            # reuse across containers); dropping prefix caching is the
            # next variable to test. UX cost: multi-turn chats redo
            # the system-prompt prefill each turn (~hundreds of ms on
            # this hardware). The user has explicitly accepted slower
            # normal-start in exchange for reliable snapshot load.
            # "--enable-prefix-caching",  # DISABLED iter-14d
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
            # active model: iter-15 (2026-05-24) uses 0.85 for
            # lovedheart/Qwen3.5-9B-FP8 (FP8 e4m3 ~9 GiB weights).
            # The prior pin (Qwen3.6-27B-FP8 ~27 GiB) ran at 0.8 to
            # leave ~7 GiB headroom on the 79 GiB H100 for cuBLAS
            # workspaces + CUDA contexts + paged-attention overflow
            # to CPU. With ~18 GiB freed up by the 9B FP8 swap, we can
            # comfortably afford the extra 5% utilization to expand
            # the KV cache budget for multi-turn demo sessions. Drop
            # further (0.8, 0.75) if /wake_up surfaces OOM after a
            # future model swap.
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

        # iter-49a (2026-05-25): conditionally add the /health/entropy
        # middleware to the vllm serve cmd, gated on the import working
        # cleanly in this container's Python. Two reasons for the gate:
        #
        # 1. vLLM's ``--middleware`` uses ``module.callable`` (rsplit on
        #    the rightmost ``.``), NOT the ``module:callable`` form that
        #    ``--logits-processors`` uses. A wrong format causes vLLM's
        #    ``build_app`` to crash AFTER engine init has already burned
        #    ~3 min of weight load + dynamo compile — disastrous for
        #    iteration cadence. Validating the value imports HERE,
        #    on-host, fails fast in <1 s.
        # 2. The /health/entropy endpoint powers iter-49's regenerate-
        #    banner — a cosmetic UX feature, not a load-bearing
        #    dependency. If the middleware ever fails to import (renamed
        #    fastapi dep, transitive ImportError on a future vLLM bump),
        #    the container MUST still boot. The qr-llm-chat side
        #    gracefully no-ops the banner when the endpoint returns 404.
        try:
            from qr_sampler.connectors.modal.health_entropy_middleware import (
                health_entropy_middleware as _qr_he_probe,
            )

            if not callable(_qr_he_probe):
                raise TypeError("health_entropy_middleware is not callable")
            cmd.extend(
                [
                    "--middleware",
                    "qr_sampler.connectors.modal.health_entropy_middleware.health_entropy_middleware",
                ]
            )
            log.info(
                "iter-49 /health/entropy middleware enabled",
                extra={"event": "qr.health_entropy.middleware_enabled"},
            )
        except Exception as err:
            log.warning(
                "iter-49 /health/entropy middleware NOT importable; skipping "
                "--middleware (regenerate-banner will silently no-op): %r",
                err,
                extra={
                    "event": "qr.health_entropy.middleware_skipped",
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
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

        # iter-18 (2026-05-24): defensive mirror of the VllmQrPrismaQuant
        # commit. The `vllm-cache` Volume here is already pre-warmed
        # from iter-14/15 (which is why VllmQrQwen never hit the
        # iter-17c CRIU restore failure that motivated this fix). But
        # if a future operator resets the Volume — or the cache key
        # rotates on a vLLM upgrade — the next snapshot capture would
        # silently include filesystem references to a `torch_compile_cache/<hash>`
        # directory that isn't on the persistent mount yet, and restore
        # would fail with "Runner failed with exit code: 128". This
        # commit is idempotent on an already-populated Volume (no-op
        # when there's nothing new to flush), so the cost is ~0 in the
        # common case and the protection is real in the regression case.
        try:
            vllm_cache_volume.commit()
            log.info(
                "vllm-cache Volume committed pre-snapshot",
                extra={
                    "event": "vllm.volume.commit_ok",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "volume": "vllm-cache",
                },
            )
        except Exception as exc:  # pragma: no cover -- belt-and-braces
            log.warning(
                "vllm-cache Volume commit failed: %s",
                exc,
                extra={
                    "event": "vllm.volume.commit_fail",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "volume": "vllm-cache",
                    "error_type": type(exc).__name__,
                },
            )

        # iter-14f (2026-05-23): REVERTED /sleep?level=2 back to level=1.
        # Level=2 produced fast cold-from-storage restore (10-14 s
        # observed in iter-14e logs with `vllm.snapshot.restore_class=
        # snapshot_hit`) BUT the model emitted gibberish on actual
        # inference (CJK + Cyrillic + emoji noise like "laut jewe拍下
        # Gui开机คณะกรรมการ ..."). Diagnosis: vLLM 0.17 + custom logits
        # processor + /sleep level=2 + memory snapshot together don't
        # properly restore weight tensors on wake — /wake_up returned in
        # 530 ms (far too fast for genuine 27 GB weight reload from
        # disk, which should be 2-5 s at PCIe Gen4) and the GPU memory
        # for weights ended up in undefined state. The fast wake was a
        # red flag we should have caught before declaring victory.
        #
        # Reverted to level=1: snapshot keeps weights in CPU RAM,
        # cold-from-storage retrieval is bandwidth-bound at 115-125 s,
        # but inference outputs are correct. iter-14d already proved
        # level=1 is reliable for snapshot CAPTURE under 180 s (when
        # paired with --enable-prefix-caching off — that fix stays).
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

        # iter-14: real-wall-clock timestamp used by _wake to classify the
        # next restore as snapshot_hit vs snapshot_miss. Real wall clock
        # (not monotonic) because CRIU's snapshot/restore preserves
        # time.time() but not necessarily time.monotonic() across the
        # container boundary. Stored on self so it gets captured into the
        # snapshot — _wake on a TRUE restore reads this as a stale (much
        # older) value than time.time() returns in the restored container,
        # which is exactly the signal we need.
        self._snap_built_at_wallclock = time.time()

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
            VLLM_SNAPSHOT_IDENTITY,
            VLLM_SNAPSHOT_RESTORE_CLASS,
            VLLM_WAKE_FAIL,
            VLLM_WAKE_OK,
        )
        from obs.logging import get_logger

        log = get_logger(f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}")

        # iter-14: emit the snapshot identity FIRST so even if anything
        # below in _wake fails, the operator can correlate the failed
        # cold-start to a specific deploy. Read MODAL_IMAGE_ID from env
        # (set by Modal) so the image hash is visible alongside the
        # version constant. ``modal_image_id`` may be unset on the deploy
        # host (e.g. local smoke tests) — log "<unset>" rather than
        # raising, since this event is purely diagnostic.
        log.info(
            "vllm cold-start: snapshot identity",
            extra={
                "event": VLLM_SNAPSHOT_IDENTITY,
                "served_model_name": self.SERVED_MODEL_NAME,
                "identity_version": SNAPSHOT_IDENTITY_VERSION,
                "model_revision": MODEL_REVISION or "<unpinned>",
                "image_digest": os.environ.get("MODAL_IMAGE_ID", "<unset>"),
            },
        )

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

        # iter-14: classify this restore as snapshot_hit vs snapshot_miss
        # using the wall-clock gap between _start_and_sleep finishing and
        # _wake running. ``_snap_built_at_wallclock`` is captured at the
        # end of _start_and_sleep and frozen into the snapshot; on a TRUE
        # restore the stored value is much older than the current
        # ``time.time()`` (because the container that took the snapshot
        # has long since died), whereas on the snap-create-then-wake-
        # immediately path it is only a few seconds old. 60 s is a safe
        # threshold: snapshot creation + restore happen within seconds;
        # scaledown_window=300 s (iter-39) guarantees real restores see
        # at minimum ~5 minutes of gap.
        # iter-14e tuning: with /sleep level=2 the snapshot capture
        # window collapsed from 150-200 s to ~31 s (weights are freed
        # before snapshot rather than CPU-cached, so cuda-checkpoint has
        # almost nothing to enumerate). The classification threshold
        # has to drop in step:
        #   * MISS path (just captured): gap ≈ 31-60 s on level-2.
        #   * HIT path (restored from cold storage): gap ≥ some idle +
        #     capture; iter-14e observed 259 s on a force-stop-and-ping
        #     test (gap = pre-warm-end → /health-during-restore).
        # 90 s is a comfortable margin above any capture window we
        # observe at level=2, and well below the smallest restore gap.
        snap_built_at = getattr(self, "_snap_built_at_wallclock", None)
        if snap_built_at is None:
            restore_class = "unknown"
            inference_signal = "snap_built_at_wallclock=missing"
        else:
            gap_s = time.time() - snap_built_at
            if gap_s < 90.0:
                restore_class = "snapshot_miss"
                inference_signal = f"gap_to_snap_built_s={gap_s:.1f}<90"
            else:
                restore_class = "snapshot_hit"
                inference_signal = f"gap_to_snap_built_s={gap_s:.1f}>=90"
        log.info(
            "vllm cold-start restore class: %s for %s",
            restore_class,
            self.SERVED_MODEL_NAME,
            extra={
                "event": VLLM_SNAPSHOT_RESTORE_CLASS,
                "served_model_name": self.SERVED_MODEL_NAME,
                "restore_class": restore_class,
                "total_elapsed_ms": total_elapsed_ms,
                "inference_signal": inference_signal,
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


# ----- iter-17 NEW PROFILE: VllmQrPrismaQuant ------------------------------
#
# Parallel to VllmQrQwen above. Uses ``vllm_prismaquant_image`` (vLLM
# v0.20.0) + ``vllm_prismaquant_cache_volume`` (separate torch.compile
# cache) so the iter-15 stack stays untouched. Shares the ``llm-weights``
# Volume because the HF cache layout is keyed by repo_id — no collision.
# Otherwise mirrors ``_CLS_KWARGS`` for the GPU pin, scaledown window,
# experimental_options, etc. so cold-start lifecycle behaviour matches.

_PRISMAQUANT_CLS_KWARGS: dict[str, Any] = {
    **{k: v for k, v in _CLS_KWARGS.items() if k not in {"image", "volumes", "gpu"}},
    "image": vllm_prismaquant_image,
    "volumes": {
        "/root/.cache/huggingface": weights_volume,
        "/root/.cache/vllm": vllm_prismaquant_cache_volume,
    },
    # iter-18b (2026-05-24): REVERTED RTX-PRO-6000 -> B200+ after
    # iter-18a snapshot-restore failures. iter-18a deployed cleanly
    # to RTX PRO 6000, built the snapshot successfully, and the
    # same-container wake succeeded (734 ms vllm.wake.ok, 5.94 s
    # end-to-end /health). But every cross-container restore failed
    # with the EXACT 180s NVIDIA cuda-checkpoint timeout documented
    # in the iter14 auto-memory:
    #   modal._runtime.gpu_memory_snapshot.CudaCheckpointException:
    #     Failed to restore 1 processes: PID: 43 Get state command
    #     timed out
    #   Runner failed with exit code: 1
    # Modal retried every ~1-2 min for 13+ min, all failing the same
    # way. RTX PRO 6000 is sm_120 (consumer Blackwell); the datacenter
    # B200 is sm_100, with more mature cuda-checkpoint support. Same
    # NVFP4 + MXFP8 kernel suite works on both (we confirmed
    # FlashInferCutlassNvFp4LinearKernel loaded cleanly on
    # RTX PRO 6000), so reverting only affects the snapshot-restore
    # path — the model + runtime combination remains untouched.
    # Cost trade: $6.25/h instead of $3.03/h, but containers only
    # spin up on demand for the demo, and snapshot-hit makes the
    # cold-start <30 s instead of 400 s+, so the per-user-experience
    # economics favour the more expensive SKU on this workload.
    #
    # iter-17c (kept for context): the original Hopper -> Blackwell
    # swap was driven by iter-17b's deterministic crash at vLLM 0.20's
    # ``init_nvfp4_linear_kernel`` with:
    #   ValueError: Forced NVFP4 kernel FlashInferCutlassNvFp4LinearKernel
    #     is not supported: FlashInfer + >=sm_100 required
    # The "FlashInfer-cutlass" backend in vLLM 0.20 is a FlashInfer
    # *kernel*, not a Hopper *emulation* layer — it requires Blackwell
    # silicon. PrismaQuant's recipe was developed on DGX Spark (Grace
    # Blackwell GB10) so the model card's "vLLM 0.11+ required" omitted
    # the implicit Blackwell prereq.
    #
    # The iter-15 VllmQrQwen sibling stays on H100:1 — Qwen3.5-9B-FP8
    # is pure FP8 + bf16, no NVFP4 layers, so Hopper is plenty.
    "gpu": "B200+",
}


@app.cls(**_PRISMAQUANT_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrPrismaQuant:
    """vLLM v0.20.0 + rdtand/Qwen3.6-27B-PrismaQuant-5.5bit-vllm.

    iter-17 new-engine-type profile. Parallel to ``VllmQrQwen`` (which
    serves Qwen3.5-9B-FP8 on vLLM 0.17); see that class for full
    lifecycle docs — every architectural decision (snap=True for
    snapshot build, snap=False for wake, cloudflared post-snapshot,
    background /health poller, /sleep level=1) is mirrored here.

    Notable diffs vs VllmQrQwen:

      * vLLM v0.20.0 base image (Dockerfile.vllm-prismaquant) —
        compressed-tensors loader now recognises NVFP4 (W4A16Fp4) +
        MXFP8 (W8A8Mxfp8) schemes which PrismaQuant needs.
      * ``--trust-remote-code`` (per the PrismaQuant recipe).
      * ``--hf-overrides '{"architectures":["Qwen3_5ForCausalLM"]}'`` so
        vLLM constructs the text-only model class. The PrismaQuant
        config reports ``Qwen3_5ForConditionalGeneration`` which would
        build the vision tower; that path failed in iter-16 at vLLM
        0.17 and is unverified at 0.20. Forcing text-only sidesteps
        the question for the text-chat demo; re-enable vision when a
        future iter ships a config that loads cleanly.
      * ``VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`` env (Dockerfile)
        pins the NVFP4 GEMM backend per the PrismaQuant recipe.
      * MTP speculative decoding NOT enabled — qr-sampler's per-token
        QRNG entropy accounting assumes one logits call per token.
      * Snapshot identity references ``PRISMAQUANT_SNAPSHOT_IDENTITY_VERSION``
        + ``PRISMAQUANT_MODEL_REVISION`` so this class's snapshot is
        independent from VllmQrQwen's.

    The unchanged lifecycle methods (``_poll_vllm_health``, ``_stop``,
    ``serve``) are borrowed from ``VllmQrQwen`` via class-dict
    assignment — Modal's decoration lives on the function object, so
    sharing the function preserves the hook wiring. If a future
    refactor breaks that pattern, the methods can be fully duplicated
    here at the cost of code bulk.
    """

    SERVED_MODEL_NAME = PRISMAQUANT_MODEL_SERVED_NAME
    HF_REPO_ID = PRISMAQUANT_MODEL_HF_REPO_ID

    _VLLM_PORT = 8000
    _VLLM_HOST = "127.0.0.1"
    _VLLM_BASE_URL = f"http://{_VLLM_HOST}:{_VLLM_PORT}"
    _STARTUP_TIMEOUT_S = 1200
    _SLEEP_WAKE_TIMEOUT_S = 300

    @modal.enter(snap=True)
    def _start_and_sleep(self) -> None:
        """Spawn ``vllm serve`` with PrismaQuant args, warm, /sleep.

        Near-copy of VllmQrQwen._start_and_sleep with the cmd list
        updated for vLLM 0.20 + PrismaQuant. See VllmQrQwen for full
        rationale on each unchanged step (argv validation, warmup
        before sleep, GPU drain pre-snapshot, wall-clock anchor).
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

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # iter-53/53b: tunnel-aware adaptive-timeout floor + fast
        # half-open cadence — see the twin comments in the Qwen class's
        # _start_and_sleep above.
        env.setdefault("QR_CB_MIN_TIMEOUT_MS", "300")
        env.setdefault("QR_CB_RECOVERY_WINDOW_S", "3")

        cmd = [
            "vllm",
            "serve",
            self.HF_REPO_ID,
            "--served-model-name",
            self.SERVED_MODEL_NAME,
            "--host",
            "0.0.0.0",
            "--port",
            str(self._VLLM_PORT),
            "--enable-sleep-mode",
            # iter-17: --trust-remote-code per PrismaQuant recipe. vLLM's
            # compressed-tensors loader needs to import the model's own
            # modeling code to resolve the per-layer NVFP4 / MXFP8 / BF16
            # scheme assignments.
            "--trust-remote-code",
            # iter-17: force text-only model class. PrismaQuant's
            # config.json reports ``Qwen3_5ForConditionalGeneration``
            # which builds the vision tower; that path was the iter-16
            # crash site at vLLM 0.17 and stays unverified at 0.20.
            # Forcing ``Qwen3_5ForCausalLM`` (sibling class in
            # vllm/model_executor/models/qwen3_5.py with no
            # ``self.visual``) sidesteps the question for our text-chat
            # demo. Vision-input support can be re-enabled in a future
            # iter once vLLM ships a release that loads the PrismaQuant
            # vision-QKV scheme cleanly.
            "--hf-overrides",
            '{"architectures":["Qwen3_5ForCausalLM"]}',
            "--max-model-len",
            "32768",
            "--max-num-seqs",
            "4",
            "--max-cudagraph-capture-size",
            "4",
            "--max-num-batched-tokens",
            "8192",
            # iter-17b (2026-05-24): --swap-space DROPPED for vLLM 0.20.
            # vLLM 0.20 removed/renamed the --swap-space flag from
            # AsyncEngineArgs CLI (caught by the in-process argv
            # validator: vllm.argv.unrecognized event, container
            # crashed in 3 s rather than running blind). The 0.17
            # VllmQrQwen sibling still uses --swap-space=16 — kept
            # there because that profile's argv set is validated and
            # known-working. Dropping it here means PrismaQuant uses
            # vLLM 0.20's default CPU-RAM swap budget for KV cache
            # eviction; with --max-num-seqs=4 + --max-model-len=32768
            # we will not saturate the KV cache during demo workloads
            # so the default is fine. If a future deploy needs a
            # bigger swap, look up the new flag name in vllm 0.20+
            # docs (likely renamed to --cpu-offload-gb).
            "--gpu-memory-utilization",
            PRISMAQUANT_GPU_MEMORY_UTILIZATION,
            "--logits-processors",
            "qr_sampler.engines.vllm:VLLMAdapter",
            "--reasoning-parser",
            "qwen3",
            # iter-17: --speculative-config (MTP n=3) deliberately OMITTED.
            # Recipe-recommended for raw throughput, but speculative bursts
            # generate multiple draft tokens per forward pass — collides
            # with qr-sampler's per-token QRNG entropy accounting (one
            # fetch per logits call assumed). See bootstrap_connections
            # _QWEN_ID block for the full rationale.
        ]

        # argv gate — same fail-fast pattern as VllmQrQwen.
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

        # iter-49a (2026-05-25): conditionally add the /health/entropy
        # middleware to the vllm serve cmd, gated on the import working
        # cleanly in this container's Python. Two reasons for the gate:
        #
        # 1. vLLM's ``--middleware`` uses ``module.callable`` (rsplit on
        #    the rightmost ``.``), NOT the ``module:callable`` form that
        #    ``--logits-processors`` uses. A wrong format causes vLLM's
        #    ``build_app`` to crash AFTER engine init has already burned
        #    ~3 min of weight load + dynamo compile — disastrous for
        #    iteration cadence. Validating the value imports HERE,
        #    on-host, fails fast in <1 s.
        # 2. The /health/entropy endpoint powers iter-49's regenerate-
        #    banner — a cosmetic UX feature, not a load-bearing
        #    dependency. If the middleware ever fails to import (renamed
        #    fastapi dep, transitive ImportError on a future vLLM bump),
        #    the container MUST still boot. The qr-llm-chat side
        #    gracefully no-ops the banner when the endpoint returns 404.
        try:
            from qr_sampler.connectors.modal.health_entropy_middleware import (
                health_entropy_middleware as _qr_he_probe,
            )

            if not callable(_qr_he_probe):
                raise TypeError("health_entropy_middleware is not callable")
            cmd.extend(
                [
                    "--middleware",
                    "qr_sampler.connectors.modal.health_entropy_middleware.health_entropy_middleware",
                ]
            )
            log.info(
                "iter-49 /health/entropy middleware enabled",
                extra={"event": "qr.health_entropy.middleware_enabled"},
            )
        except Exception as err:
            log.warning(
                "iter-49 /health/entropy middleware NOT importable; skipping "
                "--middleware (regenerate-banner will silently no-op): %r",
                err,
                extra={
                    "event": "qr.health_entropy.middleware_skipped",
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
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

        self._vllm_proc = subprocess.Popen(cmd, env=env)

        # Poll /health until ready.
        deadline = time.monotonic() + self._STARTUP_TIMEOUT_S
        health_url = f"{self._VLLM_BASE_URL}/health"
        ready = False
        while time.monotonic() < deadline:
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
                        "phase": "VllmQrPrismaQuant._start_and_sleep",
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
                    "phase": "VllmQrPrismaQuant._start_and_sleep",
                },
            )
            raise RuntimeError(
                f"vllm serve /health did not return 200 within {self._STARTUP_TIMEOUT_S}s"
            )

        # Warmup before /sleep — bakes torch.compile + cudagraph state.
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

        # iter-18 (2026-05-24): Persist torch.compile artefacts to the
        # `vllm-cache-prismaquant` Volume BEFORE Modal's CRIU snapshot.
        # The Volume is brand-new (created in iter-17) and vLLM writes
        # `/root/.cache/vllm/torch_compile_cache/<hash>/...` during
        # engine init. Those writes land in the ephemeral container
        # layer until `commit()` flushes them to the persistent 9p
        # mount; without this call, CRIU captures filesystem references
        # to directories that don't exist on the restored Volume, and
        # restore fails with:
        #   failed to walk "vo-.../torch_compile_cache" of type 4000:
        #   no such file or directory; Runner failed with exit code: 128
        # (the iter-17c symptom). VllmQrQwen has been getting away
        # without this because its `vllm-cache` Volume is pre-warmed
        # from iter-14/15 — see the defensive mirror at the bottom of
        # VllmQrQwen._start_and_sleep for that case.
        try:
            vllm_prismaquant_cache_volume.commit()
            log.info(
                "vllm-cache-prismaquant Volume committed pre-snapshot",
                extra={
                    "event": "vllm.volume.commit_ok",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "volume": "vllm-cache-prismaquant",
                },
            )
        except Exception as exc:  # pragma: no cover -- belt-and-braces
            log.warning(
                "vllm-cache-prismaquant Volume commit failed: %s",
                exc,
                extra={
                    "event": "vllm.volume.commit_fail",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "volume": "vllm-cache-prismaquant",
                    "error_type": type(exc).__name__,
                },
            )

        # /sleep level=1 — same rationale as VllmQrQwen (level=2 produces
        # gibberish per auto-memory iter14_snapshot_load_working).
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

        # GPU drain pre-snapshot.
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                log.info(
                    "GPU allocator drained pre-snapshot",
                    extra={
                        "event": "vllm.snapshot.gpu_drained",
                        "served_model_name": self.SERVED_MODEL_NAME,
                    },
                )
        except Exception as exc:
            log.warning(
                "pre-snapshot GPU drain skipped: %s",
                exc,
                extra={
                    "event": "vllm.snapshot.gpu_drain_skipped",
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "error_type": type(exc).__name__,
                },
            )

        # Wall-clock anchor for the iter-14 snapshot_hit/miss classifier.
        self._snap_built_at_wallclock = time.time()

    @modal.enter(snap=False)
    def _wake(self) -> None:
        """Wake the engine + emit PrismaQuant snapshot-identity events.

        Near-copy of VllmQrQwen._wake. The only meaningful diff is
        substituting PRISMAQUANT_SNAPSHOT_IDENTITY_VERSION /
        PRISMAQUANT_MODEL_REVISION for the iter-15 constants in the
        ``vllm.snapshot.identity`` event so the operator can correlate
        cold-starts to this profile specifically. The rest (cloudflared
        background spawn, /wake_up + /health, coldstart-complete event,
        snapshot-restore-class classifier, background health poller)
        is structurally identical.
        """
        import threading
        import time

        import httpx
        from obs.events import (
            VLLM_COLDSTART_COMPLETE,
            VLLM_SNAPSHOT_IDENTITY,
            VLLM_SNAPSHOT_RESTORE_CLASS,
            VLLM_WAKE_FAIL,
            VLLM_WAKE_OK,
        )
        from obs.logging import get_logger

        log = get_logger(f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}")

        # iter-17: identity references PRISMAQUANT_* constants so this
        # profile's snapshot history is greppable independently of the
        # iter-15 VllmQrQwen identity stream.
        log.info(
            "vllm cold-start: snapshot identity",
            extra={
                "event": VLLM_SNAPSHOT_IDENTITY,
                "served_model_name": self.SERVED_MODEL_NAME,
                "identity_version": PRISMAQUANT_SNAPSHOT_IDENTITY_VERSION,
                "model_revision": PRISMAQUANT_MODEL_REVISION or "<unpinned>",
                "image_digest": os.environ.get("MODAL_IMAGE_ID", "<unset>"),
            },
        )

        # cloudflared sidecar background spawn — same pattern as VllmQrQwen.
        self._cloudflared = None

        def _spawn_cloudflared() -> None:
            try:
                self._cloudflared = _start_qrng_tunnel(self.SERVED_MODEL_NAME)
            except Exception as exc:
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

        # /wake_up + post-wake /health poll.
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

        # coldstart.complete + restore-class events.
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

        snap_built_at = getattr(self, "_snap_built_at_wallclock", None)
        if snap_built_at is None:
            restore_class = "unknown"
            inference_signal = "snap_built_at_wallclock=missing"
        else:
            gap_s = time.time() - snap_built_at
            if gap_s < 90.0:
                restore_class = "snapshot_miss"
                inference_signal = f"gap_to_snap_built_s={gap_s:.1f}<90"
            else:
                restore_class = "snapshot_hit"
                inference_signal = f"gap_to_snap_built_s={gap_s:.1f}>=90"
        log.info(
            "vllm cold-start restore class: %s for %s",
            restore_class,
            self.SERVED_MODEL_NAME,
            extra={
                "event": VLLM_SNAPSHOT_RESTORE_CLASS,
                "served_model_name": self.SERVED_MODEL_NAME,
                "restore_class": restore_class,
                "total_elapsed_ms": total_elapsed_ms,
                "inference_signal": inference_signal,
            },
        )

        # Background /health poller — same as VllmQrQwen.
        self._health_stop_event = threading.Event()
        self._health_thread = threading.Thread(
            target=self._poll_vllm_health,
            args=(self._health_stop_event,),
            daemon=True,
            name=f"vllm-health-{self.SERVED_MODEL_NAME}",
        )
        self._health_thread.start()

    # ``_poll_vllm_health`` is a regular method (no @modal.* decoration),
    # so borrowing the function object via class-dict assignment is fine
    # — it gets bound to ``self`` at call time and reads its own class
    # attributes. The Modal-decorated lifecycle methods (_stop with
    # @modal.exit, serve with @modal.web_server) CANNOT be borrowed: the
    # decorator registers the function into a per-class data structure
    # at definition time, so the registration is class-scoped even
    # though the function object's marker is global. iter-17 first pass
    # (commit 1650fd0) tried borrowing all three and Modal silently
    # skipped the @modal.web_server registration for VllmQrPrismaQuant
    # — the deploy output showed "Created function VllmQrPrismaQuant.*"
    # but no matching "Created web endpoint for VllmQrPrismaQuant.serve",
    # leaving the class with no public URL. iter-17a fix: define _stop
    # and serve explicitly inside the class so their decorators run in
    # VllmQrPrismaQuant's class context.
    _poll_vllm_health = VllmQrQwen._poll_vllm_health

    @modal.exit()
    def _stop(self) -> None:
        """Tear down the sidecar and the vllm serve subprocess.

        Functionally identical to ``VllmQrQwen._stop`` — see that
        method for full rationale. Defined explicitly here because
        @modal.exit() class-scope registration prevents borrowing the
        decorated function via class-dict assignment (see comment
        above).
        """
        import subprocess

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

        Same rationale as ``VllmQrQwen.serve`` — the subprocess is
        already running by the time this function is invoked
        (_start_and_sleep + _wake have run), so this body is only
        entered for Modal's port-readiness probe.
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
    # iter-18 (2026-05-24): same `add_local_python_source` filter as the
    # `functions/*.json` workaround above — the splash overlay's
    # `static_assets/{custom.css, loader.js, entropic-logo.svg}` never
    # reached the container despite the pyproject.toml package-data entry,
    # because Modal's add_local_python_source ships .py / .pyi only.
    # Without this layer, `qr_llm_chat.bootstrap_static_assets` logs
    # `bootstrap_static_assets done: wrote=0 ... missing=3` at restore
    # and the OWUI app loads bare (no splash overlay, no scanline FX,
    # no rotating taglines — the entire iter-15 demo UX is invisible).
    .add_local_dir(
        str(_qr_llm_chat_static_assets_dir()),
        remote_path="/root/qr_llm_chat/static_assets",
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
    # iter-42 (2026-05-25): OWUI is kept perpetually warm via
    # min_containers=1 — Modal guarantees AT LEAST one OWUI replica
    # is alive at all times. scaledown_window=300 still applies to
    # any EXTRA replicas Modal spawns under burst (capped at
    # max_containers=1 below, so in practice there are never extras),
    # but the floor of 1 means the donor demo's first click never
    # pays the 30-60 s OWUI cold-restore.
    #
    # Cost: 1 vCPU + 2 GB RAM continuous = ~$43/month for OWUI's
    # CPU container. Acceptable for the demo posture; the operator
    # can drop this back to 0 (and pair with a scheduled cron from
    # outside the qr_sampler package) if cost ever becomes a concern.
    # iter-39/40/41 explored the cron approach — the in-app cron
    # crash-looped on the qr_sampler/__init__.py pydantic import,
    # and a standalone-app split added more deploy complexity than
    # min_containers=1 saves in cost. Direct floor is simpler.
    #
    # ENDING SEMANTICS (Modal docs, §lifecycle-functions): if the
    # operator runs ``modal app stop qr-llm-chat`` the floor is
    # released and the container terminates gracefully via
    # @modal.exit() (30 s grace) + ASGI lifespan shutdown.
    # max_containers=1 below caps the per-class replica count;
    # min_containers=1 sets the floor. The two together pin OWUI
    # at exactly one replica.
    scaledown_window=300,  # 5 min — only relevant for transient extras.
    timeout=60 * 60,
    min_containers=1,  # iter-42: keep OWUI warm 24/7 — see comment above.
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
