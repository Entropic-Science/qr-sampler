"""Reusable Modal warm-probe helper for OWUI plugins.

Generally useful for any deployment that fronts a scale-to-zero Modal
function — not entropic.science-specific. Returned states map cleanly to a
UX decision tree:

    "warm"     → no indicator needed; the upstream is already serving.
    "cold"     → emit a "spinning up" indicator before the first token.
    "unknown"  → upstream is unreachable; let the underlying request raise.

The probe is a single `HEAD /models` call with a hard cap (`timeout_s`) and a
fast-path cutoff (`warm_threshold_s`). Latencies above the cutoff but below
the cap are still treated as `"cold"` because Modal's snapshot-restore curve
is bimodal; there is no meaningful "warm-ish" middle ground.
"""

from __future__ import annotations

import time
from typing import Literal

import httpx

WarmthState = Literal["warm", "cold", "unknown"]


async def probe_warmth(
    base_url: str,
    *,
    timeout_s: float,
    warm_threshold_s: float,
) -> WarmthState:
    """Probe an OpenAI-compatible upstream for warmth.

    Args:
        base_url: Upstream base URL (e.g. `https://…modal.run/v1`). The probe
            issues `HEAD <base_url>/models`.
        timeout_s: Hard cap on the HTTP request. Anything past this point is
            treated as `"cold"` (the upstream is responding but slowly).
        warm_threshold_s: Latency cutoff. Responses arriving within this
            budget are `"warm"`; slower-but-still-within-`timeout_s` are
            `"cold"`.

    Returns:
        `"warm"`, `"cold"`, or `"unknown"` (connection refused or transport
        error — caller may want to surface this distinctly from a slow cold
        start because the upstream may be genuinely down).
    """
    url = base_url.rstrip("/") + "/models"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.head(url)
    except httpx.TimeoutException:
        return "cold"
    except httpx.HTTPError:
        return "unknown"

    elapsed = time.monotonic() - start
    # 4xx/5xx still counts as a response — the container is alive enough to
    # respond. A scale-to-zero cold start manifests as a slow response or
    # timeout, not a non-2xx status.
    if response.status_code >= 500:
        return "unknown"
    if elapsed <= warm_threshold_s:
        return "warm"
    return "cold"
