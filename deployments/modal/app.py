"""Modal app definition — vllm-qr (B200, per-model containers).

Layout (matches spec.md §5.5 / §4.1, with the labs-cutover per-model split):

    weights_volume     — Volume "llm-weights", mounted at /root/.cache/huggingface
    download_weights   — one-shot @app.function to populate weights_volume
    VllmQrGemma        — @app.cls (B200) running google/gemma-4-31B alone
    VllmQrQwen         — @app.cls (B200) running Qwen/Qwen3.6-27B alone

Each model is its own scale-to-zero @app.cls so OWUI's model picker wakes
only the requested container. The two classes share `vllm_image` (same
Dockerfile, same qr-sampler install) — the split is purely runtime, not
build-time. Open WebUI itself, and its auth bridge, live on the
entropic.science Replit deployment alongside the api-server (see
entropic.science/spec.md §3.2 — the `artifacts/open-webui/` artifact and
`middlewares/owuiAuthBridge.ts`).

Deploy:
    modal deploy deployments/modal/app.py

One-shot weights download (before first deploy / on model upgrade):
    modal run deployments/modal/app.py::download_weights

Snapshot-failure fallback (Pre-flight §11.7): if memory-snapshot restore
fails on B200, set `enable_memory_snapshot=False` on the affected class
and redeploy. Cold start becomes ~30-45s instead of ~10-15s; pre-baked
weights still cut the majority of init time. Do NOT add `keep_warm=1`
on either class — always-on B200 cost is unacceptable per Pre-flight §11.7.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import modal

APP_NAME = "qr-sampler-entropic"

# Repo root, computed relative to this file (deployments/modal/app.py).
# Only meaningful at image-build time (locally). In a Modal container Modal
# mounts app.py at /root/app.py, which has no parents[2] — we fall back to a
# placeholder because _REPO_ROOT is only read by `Image.from_dockerfile`'s
# `context_dir` arg at build time.
try:
    _REPO_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    _REPO_ROOT = Path("/")

# --- Volumes ---------------------------------------------------------------

# Both Gemma 4 31B and Qwen 3.6 27B directories live in this volume.
# Populated by `download_weights`; each class mounts it read-only and reads
# only its own subdirectory at engine init.
weights_volume = modal.Volume.from_name("llm-weights", create_if_missing=True)

# --- Secrets ---------------------------------------------------------------

# Provisioned via `modal secret create` — see deployments/modal/modal_secrets.md.
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
# `add_python="3.12"` tells Modal which Python version `add_local_python_source`
# should target. The vllm/vllm-openai:v0.6.6 base ships Python 3.12 (visible
# in the build log as /usr/local/lib/python3.12/dist-packages), but Modal's
# from_dockerfile() introspection can't see that through Docker layers, so we
# declare it. (Modal SDK 1.1.1: this is the documented kwarg name —
# `python_version` is not accepted on from_dockerfile.)
#
# This image is shared by both VllmQrGemma and VllmQrQwen — the per-model
# split is at the class/container level, not the image level.
vllm_image = modal.Image.from_dockerfile(
    str(Path(__file__).parent / "Dockerfile.vllm"),
    context_dir=str(_REPO_ROOT),
    add_python="3.12",
).add_local_python_source("deployments", copy=True)

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
# OWUI is configured with one Connection per Modal URL (see
# entropic.science/.zenflow/tasks/qr-sampler-app-ui-ad88/labs-cutover-handoff.md
# §3.5); the model the user picks in the OWUI dropdown determines which
# container wakes.
#
# Sharing the @app.cls config: the two classes only differ in their
# class-level `SERVED_MODEL_NAME` and `HF_REPO_ID`. Everything else
# (image, secrets, volume mount, GPU type, scale-to-zero window, max
# concurrent inputs) is identical.


_CLS_KWARGS: dict[str, Any] = {
    "image": vllm_image,
    "gpu": "B200",
    "region": "us-east-1",
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

    Memory-snapshot phase (``@modal.enter(snap=True)``) builds the engine
    and pre-initialises both entropy pipelines (per
    ``QR_PREINIT_ENTROPY_SOURCES``). Modal captures the post-init state;
    subsequent cold starts restore from the snapshot.
    """

    SERVED_MODEL_NAME = "gemma-4-31b-reasoning"
    HF_REPO_ID = "google/gemma-4-31B"

    @modal.enter(snap=True)
    def load(self) -> None:
        import asyncio

        from deployments.modal.vllm_serve import build_dispatcher_for

        self._asgi_app = asyncio.run(
            build_dispatcher_for(self.SERVED_MODEL_NAME, self.HF_REPO_ID)
        )

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app


@app.cls(**_CLS_KWARGS)
@modal.concurrent(max_inputs=8)
class VllmQrQwen:
    """One ``AsyncLLMEngine`` serving ``Qwen/Qwen3.6-27B`` at full precision.

    See ``VllmQrGemma`` for the snapshot/scale-to-zero design — identical
    here, only the model identity differs.
    """

    SERVED_MODEL_NAME = "qwen-3.6-27b-reasoning"
    HF_REPO_ID = "Qwen/Qwen3.6-27B"

    @modal.enter(snap=True)
    def load(self) -> None:
        import asyncio

        from deployments.modal.vllm_serve import build_dispatcher_for

        self._asgi_app = asyncio.run(
            build_dispatcher_for(self.SERVED_MODEL_NAME, self.HF_REPO_ID)
        )

    @modal.asgi_app()
    def serve(self) -> Any:
        return self._asgi_app
