"""
title: QR-Sampler Parameters
author: qr-sampler
author_url: https://github.com/alchemystack/qr-sampler
version: 0.2.0
license: MIT
description: qr-sampler params + entropic.science allowance metering.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_WAITLIST_URL = "https://entropic.science/account/waitlist?from=allowance"
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEBIT_PATH = "/allowance/debit"
_PREFLIGHT_PATH = "/allowance/preflight"
_UPSERT_PATH = "/conversations/upsert"

_log = logging.getLogger("qr_sampler.open_webui_filter")


class FilterError(RuntimeError):
    """Raised by `_call_api` and helpers when an API call cannot proceed.

    Subclassing `RuntimeError` keeps Open WebUI's default error rendering
    intact while still letting tests assert against `FilterError` specifically.
    """


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_out_of_allowance_markdown(resp: dict[str, Any]) -> str:
    """Render the user-facing rejection body when preflight reports `insufficient`.

    Spec §5.3 + PRD R-3.6: surface the next-refill time and a waitlist CTA.
    The timestamp is rendered as UTC plus a humanised relative window so the
    user does not need to mentally convert timezones inside the chat.
    """
    next_refill_raw = resp.get("nextRefillAt", "")
    pretty_utc = _format_utc(next_refill_raw)
    relative = _humanise_until(next_refill_raw)
    refill_line = f"**{pretty_utc} UTC**"
    if relative:
        refill_line += f" (in about {relative})"
    return (
        "## Daily allowance used up\n\n"
        "You've used today's free quantum-sampling allowance. "
        f"Your allowance refills at {refill_line}.\n\n"
        f"Want priority access? [Register interest →]({_WAITLIST_URL})"
    )


def _format_utc(iso_ts: str) -> str:
    """Format an ISO-8601 UTC timestamp as `YYYY-MM-DD HH:MM`.

    Falls back to the raw string on parse failure so the message remains
    readable even if the API ever returns an unexpected shape.
    """
    parsed = _parse_iso_utc(iso_ts)
    if parsed is None:
        return iso_ts or "soon"
    return parsed.strftime("%Y-%m-%d %H:%M")


def _humanise_until(iso_ts: str) -> str:
    """Return a coarse-grained "x hours" / "x minutes" string, or empty if past."""
    parsed = _parse_iso_utc(iso_ts)
    if parsed is None:
        return ""
    delta = parsed - datetime.now(timezone.utc)
    total = int(delta.total_seconds())
    if total <= 0:
        return ""
    hours, remainder = divmod(total, 3600)
    minutes = remainder // 60
    if hours >= 1:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m" if minutes else "<1m"


def _parse_iso_utc(iso_ts: str) -> datetime | None:
    """Best-effort ISO-8601 → aware datetime parser. Accepts trailing `Z`."""
    if not iso_ts:
        return None
    candidate = iso_ts.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now_iso() -> str:
    """UTC `now()` as ISO-8601 with a `Z` suffix (matches the API contract)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Prompt-token estimator
# ---------------------------------------------------------------------------


def _estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Rough char-count → token estimate for the preflight gate.

    The preflight gate is intentionally soft — the binding charge is the
    outlet debit against actual vLLM usage. We use ~4 chars/token (the
    OpenAI rule of thumb for English) on every message's text content.
    Multi-part content lists are flattened to their `text` parts; non-text
    parts (images, audio) are ignored for the estimate.
    """
    total_chars = 0
    for msg in body.get("messages") or []:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
    return max(1, total_chars // _CHARS_PER_TOKEN_ESTIMATE)


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


def _split_secrets(raw: str) -> list[str]:
    """Split a comma-separated rolling-secret vector and trim whitespace."""
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _sign_service_token(path: str, secret: str, unix_ts: int | None = None) -> str:
    """Mint a service token matching `serviceToken.ts` on the entropic.science side.

    Wire format: `<unix_ts>.<hmac>` where `hmac = HMAC-SHA256(secret, unix_ts + path)`.
    The path must include the `/api` mount prefix (e.g. `/api/allowance/preflight`)
    because the server reconstructs it via `req.baseUrl + req.path`.
    """
    ts = unix_ts if unix_ts is not None else int(time.time())
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{ts}{path}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{ts}.{digest}"


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter:
    """Open WebUI filter — `qr_*` injection + entropic.science allowance metering.

    Inlet flow:
        OWUI → preflight (rejects below gate cost) → inject qr_* params → vLLM

    Outlet flow:
        vLLM (or final SSE chunk) → debit weighted tokens → upsert chat shadow row

    Both flows degrade safely: a missing email, missing service-token secret,
    or unreachable API raises a `FilterError` from `inlet()` (so OWUI surfaces
    the failure to the user) but never from `outlet()` (debit/upsert errors
    are logged and swallowed — a failed metering call must not corrupt the
    user's response).
    """

    class Valves(BaseModel):
        """Admin-configurable parameters surfaced in OWUI's Valves UI."""

        # --- Filter control ---
        priority: int = Field(
            default=0,
            description="Filter execution priority (lower runs first).",
        )
        enable_qr_sampling: bool = Field(
            default=True,
            description=(
                "Master switch. When False, requests pass through unmodified "
                "and the entropic.science allowance is NOT consulted."
            ),
        )

        # --- entropic.science integration ---
        api_base_url: str = Field(
            default="https://entropic.science/api",
            description=(
                "Base URL of the entropic.science API. The filter signs the "
                "URL path (including `/api/...`) with the service-token secret."
            ),
        )
        service_token_secret: str = Field(
            default_factory=lambda: os.environ.get("SERVICE_TOKEN_SECRETS", ""),
            description=(
                "Service-token secret. Accepts a comma-separated rolling-secret "
                "vector — the filter signs with the FIRST entry. Rotation = "
                "prepend the new secret on both ends, redeploy at leisure, "
                "remove the old one on the next routine deploy."
            ),
        )
        min_reserved_output_tokens: int = Field(
            default=128,
            description=(
                "Output tokens reserved in the preflight cost estimate. The "
                "API gate cost is `prompt + 3 * min_reserved_output_tokens`, "
                "doubled for comparison mode."
            ),
        )
        request_timeout_s: float = Field(
            default=5.0,
            description="Per-call HTTP timeout for the entropic.science API. No retries.",
        )

        # --- Token selection ---
        top_k: int = Field(
            default=0,
            description="Top-k filtering: keep only the k most probable tokens (0 disables).",
        )
        top_p: float = Field(
            default=1.0,
            description="Nucleus sampling threshold (1.0 disables).",
        )

        # --- Temperature ---
        temperature_strategy: str = Field(
            default="fixed",
            description="Temperature strategy: 'fixed' or 'edt' (entropy-dependent).",
        )
        fixed_temperature: float = Field(
            default=0.7,
            description="Constant temperature when strategy is 'fixed'.",
        )
        edt_base_temp: float = Field(
            default=0.8,
            description="Base coefficient for EDT strategy.",
        )
        edt_exponent: float = Field(
            default=0.5,
            description="Power-law exponent for EDT strategy.",
        )
        edt_min_temp: float = Field(
            default=0.1,
            description="EDT temperature floor.",
        )
        edt_max_temp: float = Field(
            default=2.0,
            description="EDT temperature ceiling.",
        )

        # --- Signal amplification ---
        signal_amplifier_type: str = Field(
            default="zscore_mean",
            description="Signal amplification algorithm.",
        )
        sample_count: int = Field(
            default=20480,
            description="Number of entropy bytes to fetch per token.",
        )

        # --- Logging ---
        log_level: str = Field(
            default="summary",
            description="Logging verbosity: 'none', 'summary', or 'full'.",
        )
        diagnostic_mode: bool = Field(
            default=False,
            description="Store all token records in memory for analysis.",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # Fields that map to `qr_*` extra args on the vLLM request.
    _QR_FIELDS: frozenset[str] = frozenset(
        {
            "signal_amplifier_type",
            "sample_count",
            "temperature_strategy",
            "fixed_temperature",
            "edt_base_temp",
            "edt_exponent",
            "edt_min_temp",
            "edt_max_temp",
            "top_k",
            "top_p",
            "log_level",
            "diagnostic_mode",
        }
    )

    # -- Public hooks -----------------------------------------------------

    async def inlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Preflight the allowance, then inject `qr_*` params.

        Raises:
            FilterError: when preflight rejects or the email is missing.
        """
        if not self.valves.enable_qr_sampling:
            return body

        email = (__user__ or {}).get("email")
        if not isinstance(email, str) or not email:
            raise FilterError("This chatbot requires a signed-in account.")

        comparison = _read_comparison_flag(body)
        estimated_prompt = _estimate_prompt_tokens(body)

        resp = await self._call_api(
            "POST",
            _PREFLIGHT_PATH,
            json={
                "accountEmail": email,
                "estimatedPromptTokens": estimated_prompt,
                "comparisonMode": comparison,
            },
        )

        if not resp.get("ok"):
            reason = str(resp.get("reason") or "unknown")
            if reason == "insufficient":
                raise FilterError(_render_out_of_allowance_markdown(resp))
            raise FilterError(f"Unable to start generation: {reason}.")

        valve_dict = self.valves.model_dump()
        for field_name in self._QR_FIELDS:
            body[f"qr_{field_name}"] = valve_dict[field_name]

        # `stream: True` (or `False`) set by the caller is preserved as-is.
        return body

    async def outlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Debit the actual usage and upsert the conversation shadow row.

        Best-effort: any debit/upsert error is logged and swallowed. The
        response body is returned unmodified either way.
        """
        if not self.valves.enable_qr_sampling:
            return body

        email = (__user__ or {}).get("email")
        if not isinstance(email, str) or not email:
            return body

        usage = body.get("usage") or {}
        prompt_t = _coerce_nonneg_int(usage.get("prompt_tokens"))
        completion_t = _coerce_nonneg_int(usage.get("completion_tokens"))
        if prompt_t == 0 and completion_t == 0:
            return body

        metadata = body.get("metadata") or {}
        comparison = bool(metadata.get("qr_comparison_mode"))
        chat_id = metadata.get("chat_id")
        title = metadata.get("chat_title") or "Untitled"

        try:
            await self._call_api(
                "POST",
                _DEBIT_PATH,
                json={
                    "accountEmail": email,
                    "promptTokens": prompt_t,
                    "completionTokens": completion_t,
                    "comparisonMode": comparison,
                    "conversationId": chat_id if isinstance(chat_id, str) else None,
                },
            )
        except FilterError as exc:
            _log.warning("allowance debit failed: %s", exc)

        if isinstance(chat_id, str) and chat_id:
            try:
                await self._call_api(
                    "POST",
                    _UPSERT_PATH,
                    json={
                        "accountEmail": email,
                        "owuiChatId": chat_id,
                        "title": title,
                        "lastMessageAt": _now_iso(),
                        "comparisonModeUsed": comparison,
                        "weightedTokensTotal": prompt_t + 3 * completion_t,
                    },
                )
            except FilterError as exc:
                _log.warning("conversations upsert failed: %s", exc)

        return body

    # -- Internal helpers -------------------------------------------------

    async def _call_api(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any],
    ) -> dict[str, Any]:
        """Sign an `X-Service-Token` header and POST `json` to `<api_base_url><path>`.

        Returns the decoded JSON envelope on 2xx. Raises `FilterError` on any
        transport failure, non-2xx response, or undecodable body.
        """
        secrets = _split_secrets(self.valves.service_token_secret)
        if not secrets:
            raise FilterError(
                "SERVICE_TOKEN_SECRETS is not configured; allowance metering is offline.",
            )
        signing_secret = secrets[0]

        url = httpx.URL(self.valves.api_base_url.rstrip("/") + path)
        signed_path = url.path  # includes the `/api` mount prefix
        token = _sign_service_token(signed_path, signing_secret)

        timeout = httpx.Timeout(self.valves.request_timeout_s)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method,
                    str(url),
                    json=json,
                    headers={
                        "x-service-token": token,
                        "content-type": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            raise FilterError(f"entropic.science API unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise FilterError(
                f"entropic.science API returned {response.status_code} for {path}",
            )

        try:
            decoded: Any = response.json()
        except ValueError as exc:
            raise FilterError(f"entropic.science API returned non-JSON for {path}") from exc

        if not isinstance(decoded, dict):
            raise FilterError(f"entropic.science API returned non-object for {path}")
        return decoded


def _read_comparison_flag(body: dict[str, Any]) -> bool:
    """Read `qr_comparison_mode` from either request metadata or extra_args.

    The OWUI comparison Pipe sets `body["metadata"]["qr_comparison_mode"]`
    before the filter runs. A direct API caller might instead set the
    `qr_comparison_mode` field on the top-level body (where it would land
    in vLLM's `SamplingParams.extra_args`). Accept either location so the
    preflight gate cost matches what the Pipe will actually consume.
    """
    metadata = body.get("metadata") or {}
    if metadata.get("qr_comparison_mode"):
        return True
    return bool(body.get("qr_comparison_mode"))


def _coerce_nonneg_int(value: Any) -> int:
    """Best-effort non-negative-int coercion. Returns 0 on any failure."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)
