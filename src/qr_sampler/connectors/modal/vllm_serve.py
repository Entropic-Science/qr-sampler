"""Modal-hosted vLLM helpers imported by ``qr_sampler.connectors.modal.app``.

The original incarnation of this module owned the engine construction +
ASGI app lifecycle behind a ``@modal.asgi_app()`` deployment shape
(``build_engine`` / ``build_app`` / ``build_dispatcher_for``). That
architecture was replaced in Phase 2 by the ``@modal.web_server(port=8000)``
+ ``vllm serve`` subprocess pattern; the lifecycle now lives entirely in
``app.py`` and Modal proxies inbound traffic to vLLM's own OpenAI server.

What survives here are two helpers that are still load-bearing in the
subprocess architecture, plus the bearer-auth utilities that remain
test-pinned for future restoration of an in-container auth gate:

* ``_install_mm_probe_skip_patch`` — monkey-patches vLLM V1's
  ``GPUModelRunner.profile_run`` to flip ``mm_config.skip_mm_profiling``
  before vLLM's unconditional MM dummy probe crashes on a text-only
  Qwen3.5-9B (HF config carries a populated ``vision_config``). Imported
  + invoked from ``qr_sampler.engines.vllm`` at module import time so
  the patch lands inside the ``vllm serve`` subprocess before V1
  EngineCore startup.
* ``_accepted_bearer_secrets`` / ``_verify_bearer`` / ``_check_vllm_api_key``
  — rolling-secret bearer verification helpers, pinned by
  ``tests/connectors/modal/test_vllm_serve_bearer.py``. The current
  ``vllm serve`` subprocess deployment does NOT route through these
  yet (vLLM owns the FastAPI app); when an in-container auth gate is
  added back it should reuse these helpers rather than re-deriving the
  rolling-secret semantics.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Request


def _install_mm_probe_skip_patch() -> None:
    """Force ``mm_config.skip_mm_profiling=True`` before vLLM V1's profile_run.

    vLLM V1's ``profile_run`` runs an MM dummy probe unconditionally for any
    HF model whose ``architectures`` ends in ``*ForConditionalGeneration``,
    even when no MM input will ever flow. The probe crashes for Qwen3.5-9B
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


def _accepted_bearer_secrets() -> list[str]:
    """Return the rolling-secret vector from ``SERVICE_TOKEN_SECRETS``.

    Comma-separated. Empty entries (extra commas, surrounding whitespace) are
    dropped. The signer side (Open WebUI's ``OPENAI_API_KEY``) uses the first
    entry; any entry in the vector is accepted by ``_verify_bearer`` so a
    new secret never breaks live traffic.
    """
    raw = os.environ.get("SERVICE_TOKEN_SECRETS", "") or ""
    return [token for token in (entry.strip() for entry in raw.split(",")) if token]


def _verify_bearer(token: str) -> bool:
    """Constant-time match against every accepted bearer secret.

    Uses ``hmac.compare_digest`` per entry so timing analysis cannot leak the
    accepted value. Returns ``True`` if any entry matches, ``False`` otherwise
    (including the empty-token case).
    """
    if not token:
        return False
    accepted_secrets = _accepted_bearer_secrets()
    matched = False
    # Iterate the full vector even after a hit so the wall-clock cost is
    # constant w.r.t. which entry actually matched.
    for accepted in accepted_secrets:
        if hmac.compare_digest(token, accepted):
            matched = True
    return matched


def _check_vllm_api_key(request: Request) -> None:
    """Reject the request unless it carries a valid bearer token.

    Ordering invariants pinned by ``test_vllm_serve_bearer.py``:

    1. If ``SERVICE_TOKEN_SECRETS`` is unset and
       ``ALLOW_UNAUTHENTICATED_INFERENCE=1`` is set, the request passes
       through. This is the smoke/dev escape hatch.
    2. If ``SERVICE_TOKEN_SECRETS`` is unset and the opt-in is anything
       other than the literal ``"1"`` (including ``"0"``, ``"true"``,
       empty, or absent), fail-closed with **503** — a misconfigured
       container should be loud, not open.
    3. If ``SERVICE_TOKEN_SECRETS`` is set, the opt-in is IGNORED and a
       valid bearer must be supplied (missing → 401; invalid → 401).
       Defense-in-depth: a stale ``ALLOW_UNAUTHENTICATED_INFERENCE=1``
       left over from a smoke test cannot silently disable auth in prod.
    """
    accepted_secrets = _accepted_bearer_secrets()
    if not accepted_secrets:
        if os.environ.get("ALLOW_UNAUTHENTICATED_INFERENCE") == "1":
            return
        raise HTTPException(
            status_code=503,
            detail=(
                "SERVICE_TOKEN_SECRETS is unset; bearer auth cannot be evaluated. "
                "Set ALLOW_UNAUTHENTICATED_INFERENCE=1 only for smoke/dev."
            ),
        )
    header = request.headers.get("authorization", "")
    if not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if not _verify_bearer(header[len("Bearer ") :]):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
