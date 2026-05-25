# ruff: noqa: E501  -- OWUI metadata header (line 7 `description:`) must be a single line per OWUI's manifest parser
"""
title: QR-Sampler Parameters
author: qr-sampler
author_url: https://github.com/alchemystack/qr-sampler
version: 0.6.0
license: MIT
description: qr-sampler params + entropic.science allowance metering + cold-start indicator + per-user creative-vs-T=1 sampling preset toggle + Qwen3.6 thinking-mode disable + QRNG-fallback warning banner.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

try:
    from . import (
        _modal_warmth,  # type: ignore[import-not-found]
        entropic_science_profile,  # type: ignore[import-not-found]
    )
except ImportError:
    # OWUI loads the filter as a standalone module (not as a package), so the
    # relative-import path above only works in tests. Fall back to file-relative
    # discovery so the same file works in both shapes.
    import importlib.util
    from pathlib import Path

    _here = Path(__file__).resolve().parent

    def _load_sibling(name: str) -> Any:
        spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _modal_warmth = _load_sibling("_modal_warmth")
    entropic_science_profile = _load_sibling("entropic_science_profile")

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_WAITLIST_URL = "https://entropic.science/account/waitlist?from=allowance"
_CHARS_PER_TOKEN_ESTIMATE = 4
_DEBIT_PATH = "/allowance/debit"
_PREFLIGHT_PATH = "/allowance/preflight"
_UPSERT_PATH = "/conversations/upsert"
_DEFAULT_REQUEST_KEY = "__default__"

# Fallback-visibility hook (plan R2, spec §11.5): when the configured entropy
# primary is the quantum source but the response metadata reports it actually
# resolved to ``system`` (urandom), surface that to the user as a warning.
# Otherwise users can't tell quantum-sampled tokens from PRNG-sampled ones in
# the rendered output. ``QR_ENTROPY_SOURCE_TYPE`` is the canonical name of the
# configured primary, set by the operator in the qr-sampler Modal Secret.
_FALLBACK_WARNING_MSG = (
    "Quantum entropy source unavailable; this response used local "
    "pseudo-random entropy. Operator: see DEPLOY.md."
)

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
# First-token timeout helper (reusable by the Pipe and by callers wrapping
# their own streams; the filter itself does not iterate OWUI's stream).
# ---------------------------------------------------------------------------


async def iter_with_first_token_timeout(
    source: AsyncIterator[Any],
    timeout_s: float,
) -> AsyncIterator[Any]:
    """Forward `source`, raising `asyncio.TimeoutError` if the first item is slow.

    Subsequent items have no per-item timeout — they are forwarded as fast as
    they arrive. The caller is expected to handle the `TimeoutError` (e.g.
    surface a system-message error chunk and skip the debit).
    """
    iterator = source.__aiter__()
    first = await asyncio.wait_for(iterator.__anext__(), timeout_s)
    yield first
    async for chunk in iterator:
        yield chunk


# ---------------------------------------------------------------------------
# Cold-start request state (per OWUI request)
# ---------------------------------------------------------------------------


class _ColdStartState:
    """Tracks per-request cold-start UI state shared across hook calls.

    The filter's `inlet`, `stream`, and `outlet` are separate Python calls;
    we key state by `chat_id` (from `body["metadata"]`) so a single Filter
    instance can serve many concurrent OWUI requests without crossed wires.
    Requests without a chat_id (rare — e.g. direct API tests) use a single
    sentinel key.
    """

    __slots__ = ("first_token_seen", "indicator_emitted", "started_at", "timed_out")

    def __init__(self) -> None:
        self.indicator_emitted: bool = False
        self.first_token_seen: bool = False
        self.timed_out: bool = False
        self.started_at: float = time.monotonic()


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class Filter:
    """Open WebUI filter — `qr_*` injection + entropic.science allowance metering.

    Inlet flow:
        OWUI → preflight (rejects below gate cost) → cold-start probe
             → emit indicator if cold → inject qr_* params → vLLM

    Stream flow (per SSE chunk):
        OWUI → filter.stream(event) → clear cold-start indicator on first
        non-empty assistant token.

    Outlet flow:
        vLLM (or final SSE chunk) → debit weighted tokens → upsert chat shadow row.

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

        # --- Cold-start indicator (off by default; entropic.science profile
        #     turns it on via QR_INTEGRATION_PROFILE=entropic.science) ---
        cold_start_enabled: bool = Field(
            default=False,
            description=(
                "Probe the upstream for warmth on each inlet and emit a status "
                "indicator above the assistant message when the upstream is "
                "cold. Cleared on first assistant token. Off by default."
            ),
        )
        cold_start_probe_base_url: str = Field(
            default="",
            description=(
                "Base URL probed for warmth (typically the OpenAI-compatible "
                "upstream, e.g. `https://…modal.run/v1`). Used as the fallback "
                "when `model_base_urls` has no entry for the current request's "
                "model. When blank too, the OPENAI_API_BASE_URL env var is read "
                "at probe time."
            ),
        )
        model_base_urls: dict[str, str] = Field(
            default_factory=dict,
            description=(
                "Per-model probe URL overrides: `{model_id: base_url}` (e.g. "
                "`{'gemma-4-31b-reasoning': 'https://…vllmqrgemma-serve.modal.run/v1'}`)."
                " The cold-start probe looks up `body['model']` here first; "
                "when missing or empty, falls back to `cold_start_probe_base_url`. "
                "Set this when a single OWUI instance fronts multiple per-model "
                "Modal endpoints so the probe wakes only the model in use."
            ),
        )
        cold_start_probe_timeout_s: float = Field(
            default=1.0,
            description="Hard cap on the warmth probe HTTP request.",
        )
        cold_start_warm_threshold_s: float = Field(
            default=0.5,
            description=(
                "Latency cutoff. Responses faster than this are warm; "
                "slower-but-within-`probe_timeout_s` are cold."
            ),
        )
        cold_start_message: str = Field(
            default="Spinning up the model — first request after a quiet period.",
            description="Markdown copy rendered above the assistant message during a cold start.",
        )

        entropy_degraded_enabled: bool = Field(
            default=True,
            description=(
                "Probe the upstream's /health/entropy on each inlet and emit "
                "a visible warning above the assistant message when the QRNG "
                "is unreachable (sampling falls back to system PRNG)."
            ),
        )
        entropy_degraded_probe_timeout_s: float = Field(
            default=1.5,
            description="Hard cap on the QRNG-health probe HTTP request.",
        )
        entropy_degraded_message: str = Field(
            default=(
                "⚠️ Quantum entropy source unavailable — "
                "this response is being sampled from the system PRNG fallback. "
                "Quantum-driven sampling will resume automatically once the "
                "QRNG endpoint becomes reachable again."
            ),
            description="Markdown copy rendered above the assistant message when QRNG is degraded.",
        )
        cold_start_first_token_timeout_s: float = Field(
            default=60.0,
            description=(
                "First-token deadline. Tracked via `iter_with_first_token_timeout`; "
                "if no token arrives in this window the outlet skips the debit "
                "(PRD R-3.5: no allowance consumed when no tokens generated)."
            ),
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
            default=1.0,
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
            # iter-48 (2026-05-25, qr-llm-chat shared-core mirror):
            # halved 20480 → 10000. Mirrors ``qr_sampler/config.py``.
            default=10000,
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

    class UserValves(BaseModel):
        """Per-user qr-sampler preset selector.

        Open WebUI renders this as a dropdown in each user's per-filter
        settings (it introspects pydantic ``Literal`` fields). Distinct
        from ``Valves``: ``UserValves`` is editable by every user, while
        ``Valves`` is admin-only.

        Default is ``creative_sampling`` -- the V6_HVD_R01_01 winner from
        createmp-evalsuite (UC=8.668 in the V6 explore phase). Users can
        opt back to the vanilla T=1 baseline.
        """

        preset: Literal["creative_sampling", "normal_t1"] = Field(
            default="creative_sampling",
            description=(
                "Token sampling preset. 'creative_sampling' (DEFAULT, "
                "EXPERIMENTAL) uses HVH-Drift dynamic temperature -- the "
                "V6_HVD_R01_01 winner from createmp-evalsuite (per-sequence "
                "EMA state + dynamic min-p). 'normal_t1' is the vanilla T=1 "
                "baseline (quantum entropy still drives selection)."
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Per-request cold-start state, keyed by chat_id (or a sentinel).
        self._cold_state: dict[str, _ColdStartState] = {}
        # Per-chat dedup set for the fallback-visibility warning (plan R2).
        # An entry means "the user has already been told their entropy source
        # fell back to urandom in this chat session". Cleared on filter
        # restart; OWUI's Filter instance is process-lived so this matches
        # the user's session lifetime.
        self._fallback_warned: set[str] = set()
        # Apply integration profile overrides if env switch is set.
        entropic_science_profile.apply(self.valves)

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
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Preflight the allowance, then inject `qr_*` params.

        When the cold-start probe is enabled and reports a cold upstream, a
        `status` event is emitted via `__event_emitter__` before the body is
        forwarded to vLLM. The corresponding "done" event is sent from
        `stream()` on the first non-empty assistant token.

        Raises:
            FilterError: when preflight rejects or the email is missing.
        """
        if not self.valves.enable_qr_sampling:
            return body

        # Disable Qwen3.6's default reasoning/thinking output. Qwen3.6
        # removed the ``/no_think`` soft switch from prior generations; the
        # only documented disable is the per-request ``enable_thinking``
        # template kwarg consumed by the Jinja chat template. This is a
        # top-level OpenAI-compat field (NOT a qr-sampler logits-processor
        # input), so it goes on ``body`` directly rather than under
        # ``vllm_xargs``. vLLM 0.17.0's ChatCompletionRequest schema
        # documents this field; it does not trigger the "fields ignored"
        # warning that motivated the ``vllm_xargs`` routing for ``qr_*``
        # keys. Defense in depth: even if a request bypasses this filter,
        # ``--reasoning-parser qwen3`` in vllm serve still extracts any
        # ``<think>`` block into ``reasoning_content`` so it lands in
        # OWUI's collapsible panel rather than leaking into chat content.
        body["chat_template_kwargs"] = {"enable_thinking": False}

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

        # Read the per-user preset selection. OWUI may pass ``__user__`` as a
        # dict (production) or as a pydantic-like object (tests); the nested
        # ``valves`` may itself be a ``UserValves`` instance, a plain dict,
        # or absent. Handle all three shapes defensively.
        user_valves = None
        if __user__ is not None:
            user_valves = (
                __user__.get("valves")
                if isinstance(__user__, dict)
                else getattr(__user__, "valves", None)
            )
        if user_valves is not None:
            preset_name = getattr(user_valves, "preset", None)
            if preset_name is None and isinstance(user_valves, dict):
                preset_name = user_valves.get("preset")
            if isinstance(preset_name, str) and preset_name:
                # vLLM 0.17.0's ChatCompletionRequest model logs a warning
                # for any top-level field outside its schema and DROPS it
                # before SamplingParams is built. qr-sampler reads
                # ``qr_preset`` from ``SamplingParams.extra_args``, which is
                # populated from vLLM's ``vllm_xargs`` request field. So we
                # must nest qr_* kwargs under ``vllm_xargs`` here rather
                # than setting them at the top level — otherwise the
                # logits-processor never sees them and falls back to
                # default sampling.
                xargs = body.setdefault("vllm_xargs", {})
                if isinstance(xargs, dict):
                    xargs["qr_preset"] = preset_name

        # When the user has selected a preset, qr-sampler's resolve_preset()
        # bundles the full hyperparameter set, and the caller's qr_* keys
        # WIN over preset overrides (presets.py FR-10). Skipping the admin
        # _QR_FIELDS injection keeps preset semantics coherent -- otherwise
        # admin defaults like fixed_temperature=0.7 would clobber the
        # preset's temperature_strategy=hvh_drift.
        xargs = body.setdefault("vllm_xargs", {})
        has_preset = isinstance(xargs, dict) and bool(xargs.get("qr_preset"))
        if not has_preset and isinstance(xargs, dict):
            valve_dict = self.valves.model_dump()
            for field_name in self._QR_FIELDS:
                xargs[f"qr_{field_name}"] = valve_dict[field_name]

        # iter-47 (2026-05-25, qr-llm-chat shared-core mirror): when the
        # request is routed through the qr-comparison Pipe (model id
        # prefixed ``qr_comparison_pipe.`` and/or suffixed ``-vs-prng``),
        # the Pipe owns the OWUI status slot end-to-end via its own
        # live-status updater. The Filter's status emits race the Pipe's
        # at the same ~1.5 s cadence, producing visible "alternating
        # message" flicker. Gate the Filter's cold-start probe on the
        # standalone path; the entropy-degraded probe still runs because
        # the Pipe doesn't replicate that surface.
        model_id = body.get("model")
        is_pipe_pseudo = isinstance(model_id, str) and (
            model_id.startswith("qr_comparison_pipe.")
            or model_id.endswith("-vs-prng")
        )
        if is_pipe_pseudo:
            if self.valves.entropy_degraded_enabled:
                await self._maybe_emit_entropy_degraded(body, __event_emitter__)
            return body

        if self.valves.cold_start_enabled:
            await self._maybe_emit_cold_start(body, __event_emitter__)

        if self.valves.entropy_degraded_enabled:
            await self._maybe_emit_entropy_degraded(body, __event_emitter__)

        # `stream: True` (or `False`) set by the caller is preserved as-is.
        return body

    async def stream(
        self,
        event: dict[str, Any],
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Per-chunk hook — clear the cold-start indicator on first token.

        OWUI calls this for every SSE chunk arriving from the upstream. We
        watch for the first non-empty `choices[0].delta.content` and emit a
        `{type: "status", data: {done: True}}` event to dismiss the indicator
        we set in `inlet`. The event itself is forwarded unmodified.

        iter-46 (2026-05-25, qr-llm-chat shared-core mirror): before any
        downstream extraction, inline ``reasoning_content`` into
        ``content`` for models that emit their entire response as
        reasoning (Qwen 3.6 with ``--reasoning-parser qwen3``). Runs
        BEFORE the cold-start gate so the mutation reaches subsequent
        handlers in the OWUI chain regardless of cold-start state.
        """
        event = _inline_reasoning_into_content(event)

        if not self.valves.cold_start_enabled or __event_emitter__ is None:
            return event

        delta_text = _extract_delta_text(event)
        if not delta_text:
            return event

        key = _state_key_from_event(event)
        state = self._cold_state.get(key)
        if state is None or state.first_token_seen:
            return event

        state.first_token_seen = True
        if state.indicator_emitted:
            try:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "", "done": True},
                    }
                )
            except Exception as exc:
                _log.warning("cold-start clear event failed: %s", exc)

        return event

    async def outlet(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Debit the actual usage and upsert the conversation shadow row.

        Best-effort: any debit/upsert error is logged and swallowed. The
        response body is returned unmodified either way. When the cold-start
        first-token timeout was hit, no debit is sent (PRD R-3.5).

        Also surfaces the fallback-visibility warning (plan R2) when the
        response's ``qr_metadata.last_source_used`` differs from the
        configured primary (``QR_ENTROPY_SOURCE_TYPE``). Runs before the
        allowance branches so the warning fires even on the OWUI-only
        deploy profile that does not gate on email.
        """
        if not self.valves.enable_qr_sampling:
            return body

        await self._maybe_warn_fallback_visibility(body, __event_emitter__)

        email = (__user__ or {}).get("email")
        if not isinstance(email, str) or not email:
            return body

        metadata = body.get("metadata") or {}
        chat_id = metadata.get("chat_id")
        state_key = chat_id if isinstance(chat_id, str) and chat_id else _DEFAULT_REQUEST_KEY
        state = self._cold_state.pop(state_key, None)
        if state is not None and state.timed_out and not state.first_token_seen:
            return body

        usage = body.get("usage") or {}
        prompt_t = _coerce_nonneg_int(usage.get("prompt_tokens"))
        completion_t = _coerce_nonneg_int(usage.get("completion_tokens"))
        if prompt_t == 0 and completion_t == 0:
            return body

        comparison = bool(metadata.get("qr_comparison_mode"))
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

    def mark_first_token_timeout(self, chat_id: str | None) -> None:
        """Mark a request as having missed the first-token deadline.

        Callers that wrap their own stream (e.g. the comparison Pipe, or a
        higher-level orchestrator using `iter_with_first_token_timeout`) call
        this when `asyncio.TimeoutError` fires. The next `outlet()` for the
        same `chat_id` will then skip the debit.
        """
        key = chat_id if isinstance(chat_id, str) and chat_id else _DEFAULT_REQUEST_KEY
        state = self._cold_state.setdefault(key, _ColdStartState())
        state.timed_out = True

    # -- Internal helpers -------------------------------------------------

    async def _maybe_warn_fallback_visibility(
        self,
        body: dict[str, Any],
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        """Emit a one-shot fallback warning when entropy primary != actual.

        The vLLM serve layer attaches ``body["qr_metadata"]["last_source_used"]``
        (the name of the entropy source that actually produced bytes for this
        response). The configured primary is read from
        ``QR_ENTROPY_SOURCE_TYPE`` at call time so an operator flipping the
        Modal Secret takes effect on the next response.

        Silently no-ops in three benign cases:
          * the response carries no ``qr_metadata`` (legacy / pre-R2 vLLM
            serve layer that does not yet attach it),
          * the configured primary env var is unset (no primary to compare),
          * OWUI did not pass an event emitter to outlet (no UI channel).

        Dedup is per ``chat_id`` (or the default-key sentinel for non-chat
        requests) so a single fallback session does not spam the user with
        one warning per turn.
        """
        if emitter is None:
            return

        qr_metadata = body.get("qr_metadata")
        if not isinstance(qr_metadata, dict):
            return
        last_source = qr_metadata.get("last_source_used")
        if not isinstance(last_source, str) or not last_source:
            return

        configured = os.environ.get("QR_ENTROPY_SOURCE_TYPE", "").strip()
        if not configured or configured == last_source:
            return

        metadata = body.get("metadata") or {}
        chat_id = metadata.get("chat_id")
        dedup_key = chat_id if isinstance(chat_id, str) and chat_id else _DEFAULT_REQUEST_KEY
        if dedup_key in self._fallback_warned:
            return

        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "level": "warning",
                        "description": _FALLBACK_WARNING_MSG,
                        "done": True,
                    },
                }
            )
        except Exception as exc:
            _log.warning("fallback-visibility emit failed: %s", exc)
            return

        self._fallback_warned.add(dedup_key)

    async def _maybe_emit_entropy_degraded(
        self,
        body: dict[str, Any],
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        """Probe ``/health/entropy``; emit a visible warning if QRNG is in fallback.

        The qr-sampler exposes ``/health/entropy`` on the same base URL
        as ``/v1/chat/completions``. When the upstream QRNG is unreachable
        (network failure, Cipherstone outage, provider-side rate limit),
        the qr-sampler falls back to the system PRNG and the response is
        no longer quantum-driven — which users care about. We probe the
        endpoint here, on every inlet, and emit a prominent ``status``
        event so the fallback is immediately visible above the assistant
        bubble. A ``None`` result (probe couldn't reach the endpoint at
        all) is treated as "unknown" and emits nothing — that case is
        already covered by the cold-start indicator.
        """
        if emitter is None:
            return
        requested_model = body.get("model")
        per_model_url = ""
        if isinstance(requested_model, str) and requested_model:
            per_model_url = (self.valves.model_base_urls or {}).get(requested_model, "").strip()
        probe_base = (
            per_model_url
            or self.valves.cold_start_probe_base_url
            or os.environ.get("OPENAI_API_BASE_URL", "")
        ).strip()
        if not probe_base:
            return
        url = probe_base.rstrip("/") + "/health/entropy"
        try:
            async with httpx.AsyncClient(timeout=self.valves.entropy_degraded_probe_timeout_s) as client:
                response = await client.get(url)
        except httpx.HTTPError:
            return
        if response.status_code != 200:
            return
        try:
            payload = response.json()
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        rpc_ok = payload.get("rpc_ok")
        if rpc_ok is True or not isinstance(rpc_ok, bool):
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": self.valves.entropy_degraded_message,
                        "done": True,
                    },
                }
            )
        except Exception as exc:
            _log.warning("entropy-degraded emit failed: %s", exc)

    async def _maybe_emit_cold_start(
        self,
        body: dict[str, Any],
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> None:
        """Probe the upstream and emit the cold-start status when slow.

        Resolution order for the probe URL:
          1. ``valves.model_base_urls[body['model']]`` — per-model override.
          2. ``valves.cold_start_probe_base_url`` — single fallback.
          3. ``OPENAI_API_BASE_URL`` env var — legacy single-endpoint fallback.

        The per-model branch is the one that matters when OWUI fronts
        multiple Modal endpoints (one per model) — only the endpoint serving
        the requested model gets woken, leaving the others at zero.
        """
        requested_model = body.get("model")
        per_model_url = ""
        if isinstance(requested_model, str) and requested_model:
            per_model_url = (self.valves.model_base_urls or {}).get(requested_model, "").strip()
        probe_base = (
            per_model_url
            or self.valves.cold_start_probe_base_url
            or os.environ.get("OPENAI_API_BASE_URL", "")
        ).strip()
        if not probe_base:
            # No probe target configured — silently skip rather than fail.
            return

        metadata = body.get("metadata") or {}
        chat_id = metadata.get("chat_id")
        key = chat_id if isinstance(chat_id, str) and chat_id else _DEFAULT_REQUEST_KEY
        state = self._cold_state.setdefault(key, _ColdStartState())

        warmth = await _modal_warmth.probe_warmth(
            probe_base,
            timeout_s=self.valves.cold_start_probe_timeout_s,
            warm_threshold_s=self.valves.cold_start_warm_threshold_s,
        )
        if warmth != "cold" or emitter is None:
            return

        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": self.valves.cold_start_message,
                        "done": False,
                    },
                }
            )
            state.indicator_emitted = True
        except Exception as exc:
            _log.warning("cold-start emit failed: %s", exc)

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


def _inline_reasoning_into_content(event: dict[str, Any]) -> dict[str, Any]:
    """Mutate-and-return an SSE event so OWUI sees content (not reasoning).

    iter-46 (2026-05-25, qr-llm-chat shared-core mirror). Qwen 3.6 with
    ``--reasoning-parser qwen3`` emits its entire response as
    ``reasoning_content``; OWUI's native chat renderer treats that as a
    collapsible "Thought" panel and shows an empty bubble. Moving the
    reasoning text into ``content`` lets OWUI render the response as
    plain assistant text. Only fires when ``content`` is empty AND
    reasoning has text — preserves collapsible-above-answer rendering
    for models that emit both fields.
    """
    if not isinstance(event, dict):
        return event
    data = event.get("data")
    if isinstance(data, dict):
        content = data.get("content")
        reasoning = data.get("reasoning") or data.get("reasoning_content")
        if (not content) and isinstance(reasoning, str) and reasoning:
            data["content"] = reasoning
            data["reasoning"] = ""
            data["reasoning_content"] = ""
    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            reasoning = delta.get("reasoning") or delta.get("reasoning_content")
            if (not content) and isinstance(reasoning, str) and reasoning:
                delta["content"] = reasoning
                delta["reasoning"] = ""
                delta["reasoning_content"] = ""
    return event


def _extract_delta_text(event: dict[str, Any]) -> str:
    """Pull the assistant-text delta out of one OpenAI-style SSE chunk.

    Accepts both the raw chunk shape (`choices[0].delta.content`) and the
    OWUI-event envelope shape (`{type: "message", data: {content: ...}}`)
    so the same helper can serve both `stream()` callbacks.
    """
    if not isinstance(event, dict):
        return ""
    # OWUI envelope first.
    data = event.get("data")
    if isinstance(data, dict):
        content = data.get("content")
        if isinstance(content, str) and content:
            return content
    # OpenAI-style chunk fallback.
    choices = event.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str):
                    return content
    return ""


def _state_key_from_event(event: dict[str, Any]) -> str:
    """Pull `chat_id` out of an OWUI stream event, or fall back to the default key."""
    if isinstance(event, dict):
        chat_id = event.get("chat_id")
        if isinstance(chat_id, str) and chat_id:
            return chat_id
    return _DEFAULT_REQUEST_KEY
