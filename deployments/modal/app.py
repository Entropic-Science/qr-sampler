"""Modal app definition — vllm-qr (B200, dual-model) + owui-edge + owui.

Layout (matches spec.md §5.5):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQr             — @app.cls (B200) running both vLLM engines in one process
    owui_edge          — @app.function (CPU) FastAPI auth-proxy
    owui               — @app.function (CPU) stock Open WebUI in trusted-header mode

Deploy:
    modal deploy deployments/modal/app.py

One-shot weights download (before first deploy / on model upgrade):
    modal run deployments/modal/app.py::download_weights

Custom domain (binds public chat.entropic.science -> owui_edge):
    modal domain create chat.entropic.science --function owui_edge

Snapshot-failure fallback (Pre-flight §11.7): if memory-snapshot restore
fails on B200, set enable_memory_snapshot=False below and redeploy. Cold
start becomes ~30-45s instead of ~10-15s; pre-baked weights still cut the
majority of init time. Do NOT add keep_warm=1 on VllmQr — always-on B200
cost is unacceptable per Pre-flight §11.7.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "qr-sampler-entropic"

# Repo root, computed relative to this file (deployments/modal/app.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Volumes ---------------------------------------------------------------

# Both Gemma 4 31B Reasoning FP8 and Qwen 3.6 27B Reasoning FP8 directories
# live in this volume. Populated by `download_weights`.
weights_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)
owui_data_volume = modal.Volume.from_name("owui-data", create_if_missing=True)

# --- Secrets ---------------------------------------------------------------

# Provisioned via `modal secret create` — see deployments/modal/modal_secrets.md.
qr_sampler_prod_secret = modal.Secret.from_name("qr-sampler-prod")
hf_token_secret = modal.Secret.from_name("hf-token")

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
vllm_image = modal.Image.from_dockerfile(
    str(Path(__file__).parent / "Dockerfile.vllm"),
    context_dir=str(_REPO_ROOT),
).add_local_python_source("deployments", copy=True)

# OWUI image built from Dockerfile.owui (stock OWUI + httpx).
owui_image = modal.Image.from_dockerfile(
    str(Path(__file__).parent / "Dockerfile.owui"),
    context_dir=str(Path(__file__).parent),
)

# Edge proxy image — CPU-only, FastAPI + httpx.
owui_edge_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi>=0.110", "httpx>=0.27", "uvicorn>=0.30")
    .add_local_python_source("deployments", copy=True)
)

# --- App -------------------------------------------------------------------

app = modal.App(APP_NAME)


# ----- One-shot weights download -------------------------------------------


_GEMMA_REPO = "google/gemma-4-31b-reasoning"
_QWEN_REPO = "Qwen/Qwen-3.6-27B-Reasoning"
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


# ----- VllmQr GPU class ----------------------------------------------------


@app.cls(
    image=vllm_image,
    gpu="B200",
    region="us-east-1",
    volumes={"/root/.cache/huggingface": weights_volume},
    secrets=[qr_sampler_prod_secret, hf_token_secret],
    enable_memory_snapshot=True,
    container_idle_timeout=180,  # 3 min idle -> shutdown
    allow_concurrent_inputs=8,
    max_containers=1,  # Pre-flight §11.8 cost ceiling
    timeout=60 * 60,
)
class VllmQr:
    """Two ``AsyncLLMEngine`` siblings on one B200, plus an ASGI dispatcher.

    Memory-snapshot phase (``@modal.enter(snap=True)``) builds both engines
    and pre-initialises both entropy pipelines per engine. Modal captures
    the post-init state; subsequent cold starts restore from the snapshot.
    """

    @modal.enter(snap=True)
    def load(self) -> None:
        import asyncio

        from deployments.modal.vllm_serve import build_dispatcher

        # build_dispatcher() instantiates AsyncLLMEngines (which warm
        # qr-sampler's VLLMAdapter and its pre-init entropy pipelines)
        # and composes the OpenAI-protocol dispatcher in front of them.
        self._asgi_app = asyncio.run(build_dispatcher())

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app


# ----- owui_edge (CPU auth-proxy) ------------------------------------------


@app.function(
    image=owui_edge_image,
    region="us-east-1",
    secrets=[qr_sampler_prod_secret],
    keep_warm=1,  # CPU-only, cheap; keeps the public-edge latency low
    timeout=60 * 5,
)
@modal.asgi_app()
def owui_edge() -> Any:
    """FastAPI auth-proxy in front of OWUI.

    Bound to the public custom domain ``chat.entropic.science`` via
    ``modal domain create chat.entropic.science --function owui_edge``.
    """
    from deployments.modal.owui_edge import build_app

    return build_app()


# ----- owui (CPU web app) --------------------------------------------------


@app.function(
    image=owui_image,
    region="us-east-1",
    volumes={"/app/backend/data": owui_data_volume},
    secrets=[qr_sampler_prod_secret],
    keep_warm=1,  # CPU-only; instant per-user response on the chat surface
    timeout=60 * 30,
)
@modal.web_server(port=8080, startup_timeout=120)
def owui() -> None:
    """Stock Open WebUI, trusted-header auth mode.

    Trusted headers are populated by ``owui_edge``. OWUI never sees the
    entropic.science session cookie directly.

    The two Global Functions (``qr_sampler_filter.py`` and
    ``qr_comparison_pipe.py``) are imported manually via OWUI admin UI on
    first deploy — see deployments/modal/README.md.

    The ``OPENAI_API_BASE_URL`` env var is set from the Modal Secret so OWUI
    forwards every chat request to ``VllmQr.serve``'s internal endpoint.
    The base image's entrypoint launches the OWUI server on port 8080.
    """
    # Modal's @web_server decorator runs the image's CMD/ENTRYPOINT. We add
    # an env precheck so the container fails loudly if a required setting
    # is missing instead of producing a silently misconfigured OWUI.
    required = (
        "WEBUI_AUTH",
        "ENABLE_SIGNUP",
        "WEBUI_AUTH_TRUSTED_EMAIL_HEADER",
        "WEBUI_AUTH_TRUSTED_NAME_HEADER",
        "OPENAI_API_BASE_URL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"owui container is missing required env vars: {missing}. "
            "Populate them via the qr-sampler-prod Modal Secret."
        )
