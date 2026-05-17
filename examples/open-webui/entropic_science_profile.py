"""Opt-in profile that names entropic.science-specific Valves defaults.

Activated by setting the environment variable `QR_INTEGRATION_PROFILE` to
`entropic.science`. When the env var is absent (or set to anything else),
the filter and Pipe behave exactly as they did pre-cold-start: no probe, no
indicator, no entropic.science-flavoured copy.

The module exposes a single `apply()` function that mutates a Valves
instance in place. Both the filter and the Pipe call this from their
`__init__` so an operator who imports either file gets the right defaults
without touching the source.

Per-model Modal URLs (`model_base_urls` valve)
----------------------------------------------
Since the Modal cutover splits inference into per-model containers
(`VllmQrGemma`, `VllmQrQwen`), the cold-start probe and the comparison
Pipe's streaming side need to target the specific Modal endpoint for the
requested model. This profile populates `model_base_urls` from the two
env vars `OPENAI_BASE_URL_GEMMA` and `OPENAI_BASE_URL_QWEN` so the
operator sets them once on the Replit deployment and both the filter and
the Pipe pick them up.
"""

from __future__ import annotations

import os
from typing import Any

ENV_FLAG = "QR_INTEGRATION_PROFILE"
ENV_VALUE = "entropic.science"

_OPENAI_BASE_URL_ENV = "OPENAI_API_BASE_URL"
_GEMMA_BASE_URL_ENV = "OPENAI_BASE_URL_GEMMA"
_QWEN_BASE_URL_ENV = "OPENAI_BASE_URL_QWEN"


def is_active() -> bool:
    """True when the integration profile is selected via env var."""
    return os.environ.get(ENV_FLAG) == ENV_VALUE


def apply(valves: Any) -> None:
    """Overlay entropic.science cold-start defaults onto a Valves instance.

    No-op when the profile is not active, so callers can invoke
    unconditionally from `__init__`. Only fields that exist on the Valves
    instance are touched, which keeps the helper safe to call from both the
    filter and the Pipe (their Valves shapes differ).
    """
    if not is_active():
        return

    overrides: dict[str, Any] = {
        "cold_start_enabled": True,
        "cold_start_probe_timeout_s": 1.0,
        "cold_start_warm_threshold_s": 0.5,
        "cold_start_message": (
            "Spinning up the quantum sampler \u2014 this happens the first time "
            "you ask after a quiet period. Usually under 15 seconds."
        ),
        "cold_start_first_token_timeout_s": 60.0,
    }

    probe_base = os.environ.get(_OPENAI_BASE_URL_ENV, "").strip()
    if probe_base:
        overrides["cold_start_probe_base_url"] = probe_base

    # Per-model URL map for the split Modal deployment. The base entries
    # cover the OWUI model picker's plain names; the `--qr-vs-prng` entries
    # cover the comparison-Pipe pseudo-models (both columns route to the
    # same base model's container).
    gemma = os.environ.get(_GEMMA_BASE_URL_ENV, "").strip()
    qwen = os.environ.get(_QWEN_BASE_URL_ENV, "").strip()
    model_base_urls: dict[str, str] = {}
    if gemma:
        model_base_urls["gemma-4-31b-reasoning"] = gemma
        model_base_urls["gemma-4-31b-reasoning--qr-vs-prng"] = gemma
    if qwen:
        model_base_urls["qwen-3.6-27b-reasoning"] = qwen
        model_base_urls["qwen-3.6-27b-reasoning--qr-vs-prng"] = qwen
    if model_base_urls:
        overrides["model_base_urls"] = model_base_urls

    for name, value in overrides.items():
        if hasattr(valves, name):
            setattr(valves, name, value)
