"""Modal app definition — vllm-qr (H200, per-model containers).

Layout (matches spec.md §5.5 / §4.1, with the labs-cutover per-model split):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQrQwen         — @app.cls (H200) running Qwen/Qwen3.6-27B alone

Each model is its own scale-to-zero @app.cls so OWUI's model picker wakes
only the requested container. Open WebUI itself is provided by the
`OWUIService` class defined below; the OWUI-specific lifecycle code
(admin bootstrap, SvelteKit base-path patch, Function bundle import)
lives in the downstream `qr-llm-chat` package, imported lazily inside
the @modal.enter hooks so this module stays usable without that
dependency installed.

Gemma 4 31B pause + Qwen 3.6 27B MM-probe monkey-patch (2026-05-20)
-------------------------------------------------------------------
1. ``VllmQrGemma`` (google/gemma-4-31B) is paused while the vLLM/
   transformers ecosystem stabilises around the gemma-4 GDN architecture.
   vLLM 0.17.0 does not register ``Gemma4ForConditionalGeneration``.
   Restore Gemma when a vLLM release ships gemma-4 GDN support.

2. ``VllmQrQwen`` serves Qwen/Qwen3.6-27B. The 27B variant has a populated
   HF ``vision_config`` so vLLM V1's ``profile_run`` would otherwise run
   an unconditional MM dummy probe that crashes in
   ``transformers.processing_utils.get_text_with_replacements`` with
   ``StopIteration``. The load-bearing fix is in
   ``qr_sampler.connectors.modal.vllm_serve._install_mm_probe_skip_patch``,
   which monkey-patches ``GPUModelRunner.profile_run`` to set
   ``mm_config.skip_mm_profiling=True`` at entry — vLLM's own supported
   short-circuit at gpu_model_runner.py:5226 in v0.17.0. The patch fires
   ONE event (``vllm.mm.probe_skipped``) on every cold-start; absence of
   that event on a future cold-start means the patch lost its hook.

Both prior model directories remain on the ``llm-weights`` volume; the
restore path is a code-only change.

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
31B/27B models at native bf16 with the default max_model_len=65536)
widens the schedulable pool while keeping the us-east-1 region pin in
place. If H200 still queues in us-east-1, the next knob is to relax
that region pin (see comment in `_CLS_KWARGS`).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "qr-llm-chat"

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

# Currently only Qwen 3.6 27B is actively served; the volume also retains
# prior model directories (Qwen 3.5 9B, Gemma 4 31B) for warm-cache
# resume if those classes return.
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
    # pydantic + pydantic-settings are needed because Modal's harness imports
    # this entire module (``qr_sampler.connectors.modal.app``) to introspect
    # ``download_weights`` before launching it, and the import transitively
    # hits ``qr_sampler.__init__:19`` → ``qr_sampler.config`` which imports
    # pydantic. Without these deps the function fails to start with
    # ``ModuleNotFoundError: No module named 'pydantic'``.
    .pip_install(
        "huggingface_hub>=0.24",
        "pydantic>=2,<3",
        "pydantic-settings>=2,<3",
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


# Restored 2026-05-20 to Qwen/Qwen3.6-27B (was briefly Qwen/Qwen3.5-9B).
# Both 27B and 9B variants have a populated HF ``vision_config`` and route
# through ``*ForConditionalGeneration`` — so the 9B swap did not actually
# dodge vLLM's V1 ``profile_run`` MM dummy probe (it crashed the same way,
# just on a smaller model). The load-bearing fix is in
# ``qr_sampler.connectors.modal.vllm_serve._install_mm_probe_skip_patch``:
# we monkey-patch ``GPUModelRunner.profile_run`` to flip
# ``mm_config.skip_mm_profiling=True`` at entry, taking the vendor-
# supported short-circuit at gpu_model_runner.py:5226.
_QWEN_REPO = "Qwen/Qwen3.6-27B"
# Pinned revisions are recorded here at deploy time. Empty string means
# "latest at download time" — pin once you know the SHA you want to lock to.
_QWEN_REVISION = os.environ.get("QWEN_REVISION", "")


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

    Idempotent — re-running just re-validates the cache. As of 2026-05-20
    this targets Qwen/Qwen3.6-27B; the MM-probe monkey-patch in
    ``vllm_serve._install_mm_probe_skip_patch`` makes this variant
    cold-startable. Prior weight directories (Qwen3.5-9B, google/gemma-4-31B)
    remain on the volume so resuming either is a code-only change with a
    warm cache hit.
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
    # H200 (141 GB HBM3e) sized for Qwen 3.6 27B at bf16 (~54 GB weights
    # plus KV cache for max_model_len=65536). H200 also keeps a wider
    # schedulable pool than B200 / A100, which mattered for the
    # 2026-05-19 capacity crunch.
    "gpu": "H200",
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
    # coast H200 would add ~30–50 ms RTT to every sampled token — at
    # 50 tok/s that is 1.5–2.5 s of added wall-clock per second of
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
    # 30 min idle -> shutdown. Modal's GPU memory snapshot CANNOT capture
    # CUDA state (see ``_VLLM_CLS_KWARGS`` comment), so every cold-start
    # rebuilds the engine from scratch (~3 min with torch.compile cache,
    # ~5 min cold). The realistic latency win comes from keeping the
    # container warm long enough that a user's next message arrives while
    # the engine is still resident — at 180 s any "let me think" pause
    # past 3 min meant a full reload. 1800 s gives a much better UX for
    # interactive chat while still scaling to zero on real idle.
    "scaledown_window": 1800,
    "max_containers": 1,  # Pre-flight §11.8 cost ceiling, per model
    "timeout": 60 * 60,
}


# ``enable_memory_snapshot=False`` on the vLLM class (only): the @modal.enter
# methods on VllmQrQwen are all ``snap=False`` anyway (the vLLM ModelConfig
# import chain needs a CUDA-attached GPU; snap=True's CUDA_VISIBLE_DEVICES=
# "none" crashes ``device_id_to_physical_device_id``), so the snapshot path
# never actually engages. Carrying ``enable_memory_snapshot=True`` from the
# shared ``_CLS_KWARGS`` is therefore inert AND creates a confusing
# failure mode: if Modal ever restores a STALE snapshot from a previous
# image's pre-restore state, the container boots against the old
# transformers/vLLM versions and produces traceback lines that cannot exist
# in the new image (e.g. the 2026-05-20 "configuration_auto.py:1040" crash,
# which referenced a file that only has 438 lines in transformers @ main).
# Disabling per-class removes that surface entirely. ``OWUIService`` keeps
# its snapshot (different image, no GPU requirement, fast cold-start win).
_VLLM_CLS_KWARGS: dict[str, Any] = {
    **_CLS_KWARGS,
    "enable_memory_snapshot": False,
}


@app.cls(**_VLLM_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrQwen:
    """One ``AsyncLLMEngine`` serving ``Qwen/Qwen3.6-27B`` at full precision.

    The 27B's HF config carries a populated ``vision_config`` which would
    otherwise crash vLLM V1's MM dummy probe. The load-bearing fix is the
    ``_install_mm_probe_skip_patch`` monkey-patch in
    ``qr_sampler.connectors.modal.vllm_serve`` — see that helper's
    docstring for the patch's risk surface.

    Lifecycle:

    * ``@modal.enter(snap=False) load`` — builds the engine and pre-initialises
      both entropy pipelines (per ``QR_PREINIT_ENTROPY_SOURCES``). Runs in
      the snap=False phase because vLLM's ``ModelConfig`` validation spawns
      a subprocess that imports the model's quantization stack, which
      transitively calls ``current_platform.get_device_capability()`` —
      requires a CUDA-attached GPU. Modal's snap=True phase has no GPU
      (``CUDA_VISIBLE_DEVICES=none``), so the import chain crashes with
      ``ValueError: invalid literal for int() with base 10: 'none'`` inside
      ``vllm/platforms/interface.py::device_id_to_physical_device_id``.
      Running at snap=False means the engine is rebuilt on every cold-start
      (no snapshot benefit) but it actually works.
    * ``@modal.enter(snap=False) start_tunnel`` — spawns the per-container
      ``cloudflared access tcp`` sidecar that fronts the QRNG gRPC service.
      Already snap=False so no live socket is frozen into the snapshot.
    * ``@modal.exit() stop_tunnel`` — terminates the sidecar on container
      shutdown.
    """

    # Machine-friendly ID echoed by vLLM's /v1/models endpoint and used as
    # the routing key throughout OWUI + the comparison Pipe. No spaces or
    # parens here -- the human-readable display label
    # ("qwen-3.6-27b (quantum-random)") is set via an OWUI ``model`` table
    # row override seeded by ``qr_llm_chat.bootstrap_connections``. The id
    # MUST stay in lockstep with ``_QWEN_ID`` in
    # ``qr_llm_chat/bootstrap_connections.py`` and the Pipe's
    # ``base_models`` default.
    SERVED_MODEL_NAME = "qwen-3.6-27b"
    # 2026-05-21: switched to Qwen/Qwen3.5-9B as a smaller, faster
    # cold-start variant (createmp-evalsuite uses this model
    # successfully). 9B has the same Qwen3_5ForConditionalGeneration
    # architecture as the 27B but ~3x smaller (17 GiB vs 51 GiB) so
    # cold-start drops to ~90 s (vs ~115 s). The OWUI-facing
    # SERVED_MODEL_NAME stays "qwen-3.6-27b" so bootstrap_connections
    # doesn't need updating to swap the displayed model.
    # Note: a pending EngineCore-500 issue (see scripts/dump_modal_secret.py
    # and Phase K notes) is independent of model choice — switching
    # back to "Qwen/Qwen3.6-27B" did not fix or worsen the symptom.
    HF_REPO_ID = "Qwen/Qwen3.5-9B"

    @modal.enter(snap=False)
    def load(self) -> None:
        import asyncio

        from qr_sampler.connectors.modal.vllm_serve import build_dispatcher_for

        # Wrap the dispatcher build so any unhandled failure surfaces as
        # a structured ``vllm.engine.build_failed`` event BEFORE Modal's
        # runtime traceback printing kicks in. Without this wrap the
        # event from ``build_engine`` itself still fires (we emit before
        # ``raise``), but a failure *outside* ``build_engine`` (e.g. in
        # ``build_app`` or in the event-loop scaffolding itself) would
        # be lost to Modal's generic exit-1 path. Emitting one more
        # JSON-per-line event with the served_model_name tag means
        # ``modal app logs | grep vllm.engine.build_failed`` always
        # turns up the failed container even when stacks interleave.
        #
        # NOTE on asyncio + vLLM V1 engine cold-start flakiness (2026-05-20):
        #
        # Some cold-starts on Modal end with ``Engine core proc EngineCore_DP0
        # died unexpectedly`` + exit code 0 + PyTorch NCCL atexit warning
        # ``destroy_process_group() was not called before program exit``.
        # The hypothesis was that ``asyncio.run()`` installs SIGINT/SIGTERM
        # handlers on the loop (via ``asyncio.Runner._install_signal_handler``)
        # which the forked EngineCore inherits, then dies on signal during
        # the multi-minute model load.
        #
        # Iteration 6 tried ``asyncio.new_event_loop()`` + ``run_until_complete``
        # to bypass the signal-handler install — but ``close()`` on exit then
        # killed vLLM's background output_handler task with ``Event loop is
        # closed`` errors. And the EngineCore subprocess STILL died with
        # exit code 0 on a separate cold-start, ruling out signal propagation
        # as the sole cause. The crashes appear intermittent (Modal container
        # resource pressure + multiple concurrent cold-start attempts
        # triggered by OWUI's deploy_guard probes).
        #
        # Canonical fix would be Modal's ``@modal.web_server`` pattern running
        # ``vllm serve`` as a subprocess (see modal-examples/06_gpu_and_ml/
        # llm-serving/vllm_inference.py) — that sidesteps the asyncio +
        # multiprocessing signal interaction entirely. Significant rewrite,
        # tracked as a follow-up.
        try:
            self._asgi_app = asyncio.run(
                build_dispatcher_for(self.SERVED_MODEL_NAME, self.HF_REPO_ID)
            )
        except BaseException as err:  # noqa: BLE001 -- must surface every failure
            import traceback as _tb

            from obs.events import VLLM_ENGINE_BUILD_FAILED
            from obs.logging import get_logger

            tb_text = "".join(
                _tb.format_exception(type(err), err, err.__traceback__)
            )
            tb_tail = "\n".join(tb_text.splitlines()[-30:])
            log = get_logger(
                f"qr_sampler.modal.app.{self.SERVED_MODEL_NAME}"
            )
            log.error(
                "VllmQrQwen.load FAILED for %s: %s: %s",
                self.SERVED_MODEL_NAME,
                type(err).__name__,
                err,
                extra={
                    "event": VLLM_ENGINE_BUILD_FAILED,
                    "served_model_name": self.SERVED_MODEL_NAME,
                    "hf_repo_id": self.HF_REPO_ID,
                    "error_type": type(err).__name__,
                    "error_msg": str(err),
                    "traceback_tail": tb_tail,
                    "phase": "VllmQrQwen.load",
                },
            )
            raise

    @modal.enter(snap=False)
    def start_tunnel(self) -> None:
        # Soft-fail: when the cloudflared sidecar cannot start (missing CF
        # Access creds, cloudflared binary not on PATH, tunnel unreachable),
        # log a structured event and continue. ``self._cloudflared = None``
        # marks the sidecar as not running; the container still serves the
        # model. qr-sampler's ``QuantumGrpcSource`` will fail to dial
        # 127.0.0.1:50051, then ``FallbackEntropySource`` (configured via
        # ``QR_FALLBACK_MODE=system`` in Dockerfile.vllm) transparently
        # degrades to ``SystemEntropySource``. The per-request
        # ``entropy.degraded`` events surface the QRNG outage to operators.
        #
        # Prior design hard-failed here, which killed the container AFTER
        # the engine had successfully built (~5 min of GPU time wasted per
        # cold-start) AND produced a misleading
        # ``Engine core proc EngineCore_DP0 died unexpectedly`` shutdown
        # cascade (atexit join → KeyboardInterrupt → subprocess exit 0).
        # See ``obs.events.CLOUDFLARED_*`` for the post-mortem.
        self._cloudflared = _start_qrng_tunnel(self.SERVED_MODEL_NAME)

    @modal.exit()
    def stop_tunnel(self) -> None:
        _stop_qrng_tunnel(getattr(self, "_cloudflared", None))

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app


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
    except Exception as err:  # noqa: BLE001 -- sidecar failure must NOT kill the container
        # Cloudflared binary missing from PATH, tunnel unreachable,
        # service-token revoked, etc. Same soft-fail policy.
        stderr_tail = None
        try:
            stderr_tail = "\n".join(list(sidecar._stderr_tail)[-30:])
        except Exception:  # noqa: BLE001
            pass
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
            "QRNG cloudflared sidecar skipped for %s "
            "(falling back to system entropy)",
            served_model_name,
            extra={
                "event": CLOUDFLARED_SIDECAR_SKIPPED,
                "served_model_name": served_model_name,
                "reason": "startup_failed",
            },
        )
        # Best-effort cleanup of any partially-started sidecar process.
        try:
            sidecar.stop()
        except Exception:  # noqa: BLE001
            pass
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
# qr-llm-chat split (plan R1–R6, requirements §10, spec §11): the Open WebUI
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
owui_data_volume = modal.Volume.from_name(
    "qr-llm-chat-data", create_if_missing=True
)


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
