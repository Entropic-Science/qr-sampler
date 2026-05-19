"""Modal app definition — vllm-qr (H200, per-model containers).

Layout (matches spec.md §5.5 / §4.1, with the labs-cutover per-model split):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQrGemma        — @app.cls (H200) running google/gemma-4-31B alone
    VllmQrQwen         — @app.cls (H200) running Qwen/Qwen3.6-27B alone

Each model is its own scale-to-zero @app.cls so OWUI's model picker wakes
only the requested container. The two classes share `vllm_image` (same
Dockerfile, same qr-sampler install) — the split is purely runtime, not
build-time. Open WebUI itself is provided by the `OWUIService` class
defined below; the OWUI-specific lifecycle code (admin bootstrap,
SvelteKit base-path patch, Function bundle import) lives in the
downstream `qr-llm-chat` package, imported lazily inside the @modal.enter
hooks so this module stays usable without that dependency installed.

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

# Both Gemma 4 31B and Qwen 3.6 27B directories live in this volume.
# Populated by `download_weights`; each class mounts it read-only and reads
# only its own subdirectory at engine init.
weights_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

# --- Secrets ---------------------------------------------------------------

# Provisioned via `modal secret create` — see modal_secrets.md (co-located).
qr_sampler_prod_secret = modal.Secret.from_name("qr-sampler-prod")
hf_token_secret = modal.Secret.from_name("huggingface-secret")

# --- Images ----------------------------------------------------------------

# Lightweight image for the one-shot weights downloader.
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub>=0.24")
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
# This image is shared by both VllmQrGemma and VllmQrQwen — the per-model
# split is at the class/container level, not the image level.
vllm_image = (
    modal.Image.from_dockerfile(
        str(Path(__file__).parent / "Dockerfile.vllm"),
        context_dir=str(_REPO_ROOT),
    )
    .add_local_python_source("qr_sampler", copy=True)
)

# --- App -------------------------------------------------------------------

app = modal.App(APP_NAME)


# ----- One-shot weights download -------------------------------------------


_GEMMA_REPO = "google/gemma-4-31B"
_QWEN_REPO = "Qwen/Qwen3.6-27B"
# Pinned revisions are recorded here at deploy time. Empty string means
# "latest at download time" — pin once you know the SHA you want to lock to.
_GEMMA_REVISION = os.environ.get("GEMMA_REVISION", "")
_QWEN_REVISION = os.environ.get("QWEN_REVISION", "")


@app.function(
    image=download_image,
    volumes={"/root/.cache/huggingface": weights_volume},
    secrets=[hf_token_secret],
    timeout=60 * 60,  # 1 hour — both downloads typically finish in 10-20 min
)
def download_weights() -> dict[str, str]:
    """Populate the ``llm-weights`` Volume with both model directories.

    Run once per model-version bump:
        modal run deployments/modal/app.py::download_weights

    Idempotent — re-running just re-validates the cache.
    """
    from huggingface_hub import snapshot_download  # type: ignore[import-untyped]

    gemma_kwargs: dict[str, Any] = {"repo_id": _GEMMA_REPO}
    if _GEMMA_REVISION:
        gemma_kwargs["revision"] = _GEMMA_REVISION
    qwen_kwargs: dict[str, Any] = {"repo_id": _QWEN_REPO}
    if _QWEN_REVISION:
        qwen_kwargs["revision"] = _QWEN_REVISION

    gemma_path = snapshot_download(**gemma_kwargs)
    qwen_path = snapshot_download(**qwen_kwargs)
    weights_volume.commit()  # type: ignore[attr-defined]

    return {
        "gemma_path": gemma_path,
        "qwen_path": qwen_path,
        "gemma_revision": _GEMMA_REVISION or "(latest)",
        "qwen_revision": _QWEN_REVISION or "(latest)",
    }


# ----- Per-model GPU classes -----------------------------------------------
#
# Each class:
#   * Loads ONE model into ONE B200 container.
#   * Scales to zero independently via `scaledown_window=180` (3 min idle).
#   * Exposes its own public Modal URL via `@modal.asgi_app()`.
#
# OWUI's bundled qr_comparison_pipe routes per-request to the right
# Modal URL using its `valves.model_base_urls` map (written at boot by
# `qr_llm_chat.bootstrap_connections`); the model the user picks in the
# OWUI dropdown determines which container wakes.
#
# Sharing the @app.cls config: the two classes only differ in their
# class-level `SERVED_MODEL_NAME` and `HF_REPO_ID`. Everything else
# (image, secrets, volume mount, GPU type, scale-to-zero window, max
# concurrent inputs) is identical.


_CLS_KWARGS: dict[str, Any] = {
    "image": vllm_image,
    # H200 (141 GB HBM3e) fits both Gemma 4 31B and Qwen 3.6 27B at
    # native bf16 with max_model_len=65536 + gpu_memory_utilization=0.90,
    # and has a wider schedulable pool than B200.
    "gpu": "H200",
    # Region pool: widened to the full US triple after the QRNG-adjacent
    # pair ("us-central", "us-east") still produced capacity-pending
    # warnings on H200 (2026-05-19). Operational priority is scheduling
    # success first — Modal's H200 supply across a single region group
    # is bursty enough that even us-central + us-east left both Gemma
    # and Qwen queued, surfacing as HTTP 303 + ``__modal_function_call_id``
    # + hang on the curl-side probe. Modal does NOT accept a wildcard
    # "us" string; only the explicit region-cluster list is valid.
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
    # We accept that penalty for the workloads that land on us-west
    # because the alternative is workloads that do not land at all.
    # If steady-state QRNG latency becomes the dominant pain point and
    # H200 capacity in us-central + us-east stabilises, narrow this
    # back to ``["us-central", "us-east"]`` — the prior comment block
    # in git history explains the latency math in detail.
    "region": ["us-east", "us-central", "us-west"],
    "volumes": {"/root/.cache/huggingface": weights_volume},
    "secrets": [qr_sampler_prod_secret, hf_token_secret],
    "enable_memory_snapshot": True,
    "scaledown_window": 180,  # 3 min idle -> shutdown
    "max_containers": 1,  # Pre-flight §11.8 cost ceiling, per model
    "timeout": 60 * 60,
}


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrGemma:
    """One ``AsyncLLMEngine`` serving ``google/gemma-4-31B`` at full precision.

    Lifecycle:

    * ``@modal.enter(snap=True) load`` — builds the engine and pre-initialises
      both entropy pipelines (per ``QR_PREINIT_ENTROPY_SOURCES``). Modal
      captures the post-init state; cold starts restore from the snapshot.
    * ``@modal.enter(snap=False) start_tunnel`` — spawns the per-container
      ``cloudflared access tcp`` sidecar that fronts the QRNG gRPC service.
      Runs in the snap=False phase so no live socket is frozen into the
      snapshot.
    * ``@modal.exit() stop_tunnel`` — terminates the sidecar on container
      shutdown.
    """

    # Machine-friendly ID echoed by vLLM's /v1/models endpoint and used as
    # the routing key throughout OWUI + the comparison Pipe. No spaces or
    # parens here -- the human-readable display label
    # ("gemma-4-31b (quantum-random)") is set via an OWUI ``model`` table
    # row override seeded by ``qr_llm_chat.bootstrap_connections``.
    SERVED_MODEL_NAME = "gemma-4-31b"
    HF_REPO_ID = "google/gemma-4-31B"

    @modal.enter(snap=True)
    def load(self) -> None:
        import asyncio

        from qr_sampler.connectors.modal.vllm_serve import build_dispatcher_for

        self._asgi_app = asyncio.run(
            build_dispatcher_for(self.SERVED_MODEL_NAME, self.HF_REPO_ID)
        )

    @modal.enter(snap=False)
    def start_tunnel(self) -> None:
        self._cloudflared = _start_qrng_tunnel(self.SERVED_MODEL_NAME)

    @modal.exit()
    def stop_tunnel(self) -> None:
        _stop_qrng_tunnel(getattr(self, "_cloudflared", None))

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrQwen:
    """One ``AsyncLLMEngine`` serving ``Qwen/Qwen3.6-27B`` at full precision.

    See ``VllmQrGemma`` for the lifecycle design — identical here, only the
    model identity differs.
    """

    # See VllmQrGemma.SERVED_MODEL_NAME for the naming contract.
    SERVED_MODEL_NAME = "qwen-3.6-27b"
    HF_REPO_ID = "Qwen/Qwen3.6-27B"

    @modal.enter(snap=True)
    def load(self) -> None:
        import asyncio

        from qr_sampler.connectors.modal.vllm_serve import build_dispatcher_for

        self._asgi_app = asyncio.run(
            build_dispatcher_for(self.SERVED_MODEL_NAME, self.HF_REPO_ID)
        )

    @modal.enter(snap=False)
    def start_tunnel(self) -> None:
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
    """Spawn the cloudflared sidecar for one VllmQr* container.

    The sidecar listens on 127.0.0.1:50051 and forwards through Cloudflare
    Access to ``QRNG_TUNNEL_HOSTNAME``. The qr-sampler ``QuantumGrpcSource``
    dials the loopback address (set via ``QR_GRPC_SERVER_ADDRESS`` in the
    Dockerfile, overridable via the qr-sampler-prod Modal Secret).

    Failure is hard: if the Cloudflare Access service token is missing,
    revoked, or the tunnel hostname is wrong, the container fails to enter
    with a structured error rather than silently degrading to urandom.
    Operators see the cloudflared stderr tail in ``modal app logs``.
    """
    import logging

    from qr_sampler.connectors.modal.cloudflared_sidecar import (
        CloudflaredConfig,
        CloudflaredSidecar,
    )

    log = logging.getLogger("qr_sampler.cloudflared")
    log.info(
        "Starting QRNG cloudflared sidecar for %s container",
        served_model_name,
        extra={"event": "cloudflared.container_start", "model": served_model_name},
    )
    sidecar = CloudflaredSidecar(CloudflaredConfig.from_env())
    sidecar.start()
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
      at the two ``VllmQrGemma`` / ``VllmQrQwen`` web URLs above.
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
