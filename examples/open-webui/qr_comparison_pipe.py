"""
title: QR vs PRNG Comparison
author: qr-sampler
author_url: https://github.com/alchemystack/qr-sampler
version: 0.3.1
license: MIT
description: Dual-column quantum vs pseudo-random comparison + per-lane history (no header echo).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

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
_PSEUDO_MODEL_SUFFIX = "--qr-vs-prng"
_DEBIT_PATH = "/allowance/debit"
_PREFLIGHT_PATH = "/allowance/preflight"
_UPSERT_PATH = "/conversations/upsert"
_QUANTUM_LABEL = "Quantum"
_PRNG_LABEL = "Pseudo-random"
# Compact status caption surfaced via the OWUI ``status`` event once at the
# top of the stream \u2014 replaces the prior verbose blockquote preamble which
# was being re-rendered on every streaming tick.
_COMPARISON_STATUS_DESCRIPTION = (
    f"Comparison mode \u00b7 {_QUANTUM_LABEL} vs {_PRNG_LABEL} \u00b7 ~2\u00d7 usage"
)

_log = logging.getLogger("qr_sampler.open_webui_comparison_pipe")


class PipeError(RuntimeError):
    """Raised by `_call_api` when an entropic.science API call cannot proceed.

    Subclassing `RuntimeError` keeps Open WebUI's default error rendering
    intact while still letting tests assert against `PipeError` specifically.
    """


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _render_dual_column_markdown(
    left: str,
    right: str,
    usage_footer: str | None = None,
) -> str:
    """Render the dual-column markdown table.

    The buffer state is pushed via OWUI ``replace`` events (see ``_run``)
    rather than re-yielded into the message body, so this no longer
    includes the verbose preamble blockquote — the table header row
    labels the columns directly and the comparison-mode caption lives in
    the status indicator above the bubble.
    """
    safe_left = _escape_for_table_cell(left)
    safe_right = _escape_for_table_cell(right)
    body = (
        f"| {_QUANTUM_LABEL} | {_PRNG_LABEL} |\n"
        "|---|---|\n"
        f"| {safe_left} | {safe_right} |"
    )
    if usage_footer:
        body = f"{body}\n\n{usage_footer}"
    return body


def _render_usage_footer(prompt_total: int, completion_total: int) -> str:
    """Per spec section 5.4: the final yield surfaces summed usage to the user."""
    return (
        f"_Tokens: {prompt_total} prompt + {completion_total} completion "
        "(billed at 2\u00d7 under comparison mode)._"
    )


def _escape_for_table_cell(text: str) -> str:
    """Escape `|` and convert newlines so a string can live inside a table cell."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


# Per-lane conversation-history sanitisation (mirrored from
# qr-llm-chat ``qr_comparison_pipe`` iter-58, 2026-06-28). The Pipe
# persists the *combined* dual-column table as the assistant turn; on a
# follow-up turn OWUI replays it as the prior assistant message and
# ``_build_side_body`` previously forwarded it verbatim to BOTH
# single-lane upstreams. Each model then saw a prior assistant turn
# beginning with ``| Quantum | Pseudo-random |`` and echoed that header
# at the top of its next reply. The helpers below invert
# ``_escape_for_table_cell`` / ``_render_dual_column_markdown`` so each
# lane is conditioned only on ITS OWN prior answer.
_LANE_QUANTUM = "quantum"
_LANE_PSEUDO = "pseudo"
_COMPARISON_HEADER_LINE = f"| {_QUANTUM_LABEL} | {_PRNG_LABEL} |"


def _split_table_cells(row: str) -> list[str]:
    """Split a markdown table row on UNescaped ``|`` delimiters.

    ``_escape_for_table_cell`` renders a literal pipe as ``\\|``, so a
    naive ``split("|")`` over-splits. We treat ``\\<char>`` as a two-char
    escape unit belonging to the current cell. For ``"| a | b |"`` this
    yields ``["", " a ", " b ", ""]``; callers take ``cells[1:-1]``.
    """
    cells: list[str] = []
    cur: list[str] = []
    i = 0
    n = len(row)
    while i < n:
        ch = row[i]
        if ch == "\\" and i + 1 < n:
            cur.append(ch)
            cur.append(row[i + 1])
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    cells.append("".join(cur))
    return cells


def _unescape_table_cell(text: str) -> str:
    """Invert ``_escape_for_table_cell`` for one cell's raw text."""
    if not text:
        return ""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt in ("\\", "|"):
                out.append(nxt)
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out).replace("<br>", "\n")


def _extract_prior_lane_text(content: str, lane: str) -> str | None:
    """Return one lane's answer from a rendered comparison table, else ``None``."""
    lines = content.split("\n")
    if len(lines) < 3:
        return None
    if lines[0].strip() != _COMPARISON_HEADER_LINE:
        return None
    if lines[1].strip().replace(" ", "") != "|---|---|":
        return None
    cells = [c.strip() for c in _split_table_cells(lines[2])[1:-1]]
    if len(cells) < 2:
        return None
    raw = cells[0] if lane == _LANE_QUANTUM else cells[1]
    return _unescape_table_cell(raw)


def _sanitize_history_for_lane(messages: Any, lane: str) -> Any:
    """Rewrite prior comparison-table assistant turns down to ``lane``'s cell.

    Returns a NEW list (new dicts only for rewritten messages) so the two
    lanes — which share the inbound ``body`` — never see each other's
    mutations. Non-assistant / non-string / non-table turns pass through.
    """
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, str):
            out.append(msg)
            continue
        lane_text = _extract_prior_lane_text(content, lane)
        if lane_text is None:
            out.append(msg)
            continue
        new_msg = dict(msg)
        new_msg["content"] = lane_text
        out.append(new_msg)
    return out


def _render_out_of_allowance_markdown(resp: dict[str, Any]) -> str:
    """Mirror of the filter's rejection markdown so the Pipe surfaces the same UX."""
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
    parsed = _parse_iso_utc(iso_ts)
    if parsed is None:
        return iso_ts or "soon"
    return parsed.strftime("%Y-%m-%d %H:%M")


def _humanise_until(iso_ts: str) -> str:
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
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Prompt-token estimator
# ---------------------------------------------------------------------------


def _estimate_prompt_tokens(body: dict[str, Any]) -> int:
    """Char-count → token estimate for the preflight gate (mirrors the filter)."""
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
# HMAC signing — kept identical to the filter so server-side verification works
# ---------------------------------------------------------------------------


def _split_secrets(raw: str) -> list[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _sign_service_token(path: str, secret: str, unix_ts: int | None = None) -> str:
    ts = unix_ts if unix_ts is not None else int(time.time())
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{ts}{path}".encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{ts}.{digest}"


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


def _coerce_nonneg_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, result)


_OWUI_PIPE_FUNCTION_PREFIX = "qr_comparison_pipe."


def _strip_pseudo_suffix(model_id: str) -> str:
    """Reduce an OWUI-routed pipe model id to its base served-model id.

    OWUI 0.9.5 dispatches to a Pipe function with ``body["model"]`` set to
    the fully-qualified id ``<function_id>.<pipe_entry_id>`` (e.g.
    ``qr_comparison_pipe.gemma-4-31b-reasoning--qr-vs-prng``). vLLM's
    ``/v1/models`` only knows the bare served name, so we strip both the
    OWUI function-id prefix and the ``--qr-vs-prng`` pseudo-model suffix
    before forwarding upstream.
    """
    model_id = model_id.removeprefix(_OWUI_PIPE_FUNCTION_PREFIX)
    if model_id.endswith(_PSEUDO_MODEL_SUFFIX):
        return model_id[: -len(_PSEUDO_MODEL_SUFFIX)]
    return model_id


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------


class Pipe:
    """OWUI manifold Pipe — quantum vs pseudo-random side-by-side comparison.

    Registers one pseudo-model per entry in `Valves.base_models`. The user
    picks `<base-model>--qr-vs-prng` from OWUI's model selector and gets a
    streaming dual-column comparison in one message.
    """

    class Valves(BaseModel):
        """Admin-configurable parameters surfaced in OWUI's Valves UI."""

        # --- entropic.science integration ---
        api_base_url: str = Field(
            default="https://entropic.science/api",
            description="Base URL of the entropic.science API.",
        )
        service_token_secret: str = Field(
            default_factory=lambda: os.environ.get("SERVICE_TOKEN_SECRETS", ""),
            description=(
                "Service-token secret. Comma-separated rolling-secret vector — "
                "the Pipe signs with the FIRST entry. Identical contract to the filter."
            ),
        )
        request_timeout_s: float = Field(
            default=5.0,
            description="Per-call HTTP timeout for the entropic.science API. No retries.",
        )

        # --- vLLM integration ---
        vllm_base_url: str = Field(
            default="http://vllm:8000/v1",
            description="Base URL of the vLLM OpenAI-compatible endpoint.",
        )
        vllm_api_key: str = Field(
            default_factory=lambda: os.environ.get("VLLM_API_KEY", ""),
            description="Bearer token forwarded to vLLM (matches the OWUI ↔ vLLM contract).",
        )
        vllm_stream_timeout_s: float = Field(
            default=120.0,
            description="Per-side streaming timeout. Excess hangs end the stream cleanly.",
        )

        # --- Pseudo-model registry ---
        base_models: str = Field(
            default="gemma-4-31b-reasoning,qwen3.5-9b-reasoning",
            description=(
                "Comma-separated list of real base models. The Pipe registers one "
                "pseudo-model per entry, suffixed `--qr-vs-prng`."
            ),
        )

        # --- Preflight ---
        min_reserved_output_tokens: int = Field(
            default=128,
            description=(
                "Output tokens reserved in the preflight cost estimate. Identical "
                "to the filter — the API doubles for `comparisonMode=true`."
            ),
        )

        # --- Cold-start indicator (off by default; entropic.science profile
        #     turns it on via QR_INTEGRATION_PROFILE=entropic.science) ---
        cold_start_enabled: bool = Field(
            default=False,
            description=(
                "Probe the upstream for warmth on each prompt and emit a single "
                "status indicator above the dual-column message. Cleared on the "
                "first delta from either side. Off by default."
            ),
        )
        cold_start_probe_base_url: str = Field(
            default="",
            description=(
                "Base URL probed for warmth (e.g. `https://…modal.run/v1`). Used "
                "as the fallback when `model_base_urls` has no entry for the "
                "current request's base model. When blank too, falls back to "
                "`vllm_base_url`."
            ),
        )
        model_base_urls: dict[str, str] = Field(
            default_factory=dict,
            description=(
                "Per-model upstream + probe URL overrides: `{base_model_id: base_url}` "
                "(e.g. `{'gemma-4-31b-reasoning': 'https://…vllmqrgemma-serve.modal.run/v1'}`)."
                " The Pipe looks up the resolved `base_model` (pseudo-model with "
                "`--qr-vs-prng` suffix stripped) here for both the cold-start probe "
                "and the streaming chat-completion call. Falls back to "
                "`cold_start_probe_base_url` / `vllm_base_url` when missing. Set "
                "this when a single OWUI instance fronts multiple per-model Modal "
                "endpoints so each prompt wakes only the model in use."
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
            description=(
                "Markdown copy rendered as a status above the comparison table during a cold start."
            ),
        )
        cold_start_first_token_timeout_s: float = Field(
            default=60.0,
            description=(
                "First-token deadline per side. On timeout the side surfaces a "
                "stream-error chunk and the debit is skipped (PRD R-3.5)."
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Manifold Pipe: OWUI groups entries under this name in the selector.
        self.type = "manifold"
        self.name = "QR vs PRNG · "
        self.id = "qr_vs_prng"
        entropic_science_profile.apply(self.valves)

    # -- Public hooks -----------------------------------------------------

    def pipes(self) -> list[dict[str, str]]:
        """Return one pseudo-model entry per registered base model.

        The returned `id` is what OWUI uses as the request body's `model` field
        when the user picks this entry. We strip the suffix back off in `pipe()`
        when forwarding to vLLM so only entropy differs across the two columns.
        """
        entries: list[dict[str, str]] = []
        for base in _split_secrets(self.valves.base_models):
            entries.append(
                {
                    "id": f"{base}{_PSEUDO_MODEL_SUFFIX}",
                    "name": f"{base} (Quantum vs Pseudo-random)",
                }
            )
        return entries

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the dual-column comparison.

        Yields raw markdown chunks. OWUI's manifold-Pipe contract treats each
        yielded string as a delta to render into the assistant message. We
        overwrite the full message content on every tick so the table grows
        live in both columns.
        """
        async for chunk in self._run(body, __user__, __event_emitter__):
            yield chunk

    # -- Orchestration ----------------------------------------------------

    async def _run(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None,
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
    ) -> AsyncIterator[str]:
        email = (__user__ or {}).get("email")
        if not isinstance(email, str) or not email:
            yield "## Sign-in required\n\nThis chatbot requires a signed-in account."
            return

        secrets = _split_secrets(self.valves.service_token_secret)
        if not secrets:
            yield (
                "## Service offline\n\n"
                "SERVICE_TOKEN_SECRETS is not configured; allowance metering is offline."
            )
            return

        try:
            preflight = await self._call_api(
                _PREFLIGHT_PATH,
                {
                    "accountEmail": email,
                    "estimatedPromptTokens": _estimate_prompt_tokens(body),
                    "comparisonMode": True,
                },
            )
        except PipeError as exc:
            yield f"## Allowance check failed\n\n{exc}"
            return

        if not preflight.get("ok"):
            reason = str(preflight.get("reason") or "unknown")
            if reason == "insufficient":
                yield _render_out_of_allowance_markdown(preflight)
            else:
                yield f"## Unable to start generation\n\n{reason}"
            return

        # Resolve the real base model from the pseudo-model id OWUI routed us to.
        pseudo_model = str(body.get("model") or "")
        base_model = _strip_pseudo_suffix(pseudo_model)
        if not base_model:
            yield "## Configuration error\n\nNo model id supplied by OWUI."
            return

        # One probe per prompt — both columns share the wake. The probe targets
        # the Modal endpoint serving `base_model`, so only that container wakes.
        cold_indicator_emitted = await self._maybe_emit_cold_start(emitter, base_model)

        # Compact status caption emitted ONCE so OWUI surfaces
        # "Comparison mode · Quantum vs Pseudo-random" above the
        # assistant bubble instead of re-rendering a preamble inside the
        # message body on every streaming tick. Skipped when the
        # cold-start indicator already claims the status slot.
        if emitter is not None and not cold_indicator_emitted:
            try:
                await emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": _COMPARISON_STATUS_DESCRIPTION,
                            "done": False,
                        },
                    }
                )
            except Exception as exc:
                _log.warning("comparison status emit failed: %s", exc)

        l_buf: list[str] = []
        r_buf: list[str] = []
        l_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        r_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        delta_event = asyncio.Event()
        first_token_seen = False
        side_timed_out = False

        async def clear_indicator() -> None:
            if not cold_indicator_emitted or emitter is None:
                return
            try:
                await emitter(
                    {
                        "type": "status",
                        "data": {"description": "", "done": True},
                    }
                )
            except Exception as exc:
                _log.warning("cold-start clear event failed: %s", exc)

        # Run both sides concurrently. Each side appends to its buffer and
        # signals `delta_event` so the emitter loop wakes up and re-renders.
        async def consume_side(
            entropy_source: str,
            buf: list[str],
            usage_out: dict[str, int],
        ) -> None:
            nonlocal first_token_seen, side_timed_out
            try:
                source = self._stream_completion(
                    base_model=base_model,
                    body=body,
                    entropy_source_type=entropy_source,
                )
                if self.valves.cold_start_enabled:
                    source = self._wrap_first_token_timeout(
                        source, self.valves.cold_start_first_token_timeout_s
                    )
                async for delta_text, usage in source:
                    if delta_text:
                        buf.append(delta_text)
                        if not first_token_seen:
                            first_token_seen = True
                            await clear_indicator()
                    if usage:
                        usage_out["prompt_tokens"] = _coerce_nonneg_int(usage.get("prompt_tokens"))
                        usage_out["completion_tokens"] = _coerce_nonneg_int(
                            usage.get("completion_tokens")
                        )
                    delta_event.set()
            except asyncio.TimeoutError:
                side_timed_out = True
                buf.append("_The service did not respond in time. Please retry._")
                delta_event.set()
            except Exception as exc:
                # Surface in-column, keep the other side running.
                # Include the exception TYPE alongside its str. Some httpx
                # exceptions (e.g. ``RemoteProtocolError``, ``ReadError``)
                # serialise with an empty message when the underlying socket
                # is closed mid-stream — without the type prefix the user
                # would see ``[stream error: ]`` and have nothing to act on.
                exc_label = type(exc).__name__
                exc_msg = str(exc).strip()
                rendered_exc = f"{exc_label}: {exc_msg}" if exc_msg else exc_label
                _log.warning(
                    "comparison side %s errored: %s", entropy_source, rendered_exc
                )
                buf.append(f"[stream error: {rendered_exc}]")
                delta_event.set()

        left_task = asyncio.create_task(
            consume_side("quantum_grpc", l_buf, l_usage),
        )
        right_task = asyncio.create_task(
            consume_side("system", r_buf, r_usage),
        )

        # OWUI's pipe contract appends every ``yield``-ed string to the
        # message body, which is wrong for a dual-column markdown table
        # that needs to be re-rendered on each token tick. We push live
        # updates via the ``replace`` event_emitter type (see
        # open_webui/socket/main.py:948-957: ``replace`` upserts the
        # message ``content`` field) and yield exactly once at the end so
        # the standard SSE/get_message_content persistence path stores
        # the same final markdown.
        last_emitted = ""
        try:
            while not (left_task.done() and right_task.done()):
                # Wait for either a delta or both tasks to finish.
                wait_event = asyncio.create_task(delta_event.wait())
                done, _pending = await asyncio.wait(
                    {left_task, right_task, wait_event},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                wait_event.cancel()
                delta_event.clear()
                rendered = _render_dual_column_markdown("".join(l_buf), "".join(r_buf))
                if rendered != last_emitted and emitter is not None:
                    try:
                        await emitter(
                            {"type": "replace", "data": {"content": rendered}}
                        )
                    except Exception as exc:
                        _log.warning("comparison replace emit failed: %s", exc)
                    last_emitted = rendered
                # Re-check loop condition after each wakeup.
                if left_task in done and right_task in done:
                    break
        finally:
            # Make sure neither task is leaked. Awaiting completed tasks is a no-op.
            for t in (left_task, right_task):
                if not t.done():
                    t.cancel()
            await asyncio.gather(left_task, right_task, return_exceptions=True)

        # If the indicator was never cleared (e.g. both sides errored), clear
        # it now so the UI does not show a spinner indefinitely.
        if cold_indicator_emitted and not first_token_seen:
            await clear_indicator()

        prompt_total = l_usage["prompt_tokens"] + r_usage["prompt_tokens"]
        completion_total = l_usage["completion_tokens"] + r_usage["completion_tokens"]

        # Final render: includes the usage footer per spec §5.4.
        usage_footer = (
            _render_usage_footer(prompt_total, completion_total)
            if (prompt_total or completion_total)
            else None
        )
        final_render = _render_dual_column_markdown(
            "".join(l_buf), "".join(r_buf), usage_footer=usage_footer
        )
        # Clear the comparison-mode status caption now that the table
        # body carries the final state on its own.
        if emitter is not None and not cold_indicator_emitted:
            try:
                await emitter(
                    {"type": "status", "data": {"description": "", "done": True}}
                )
            except Exception as exc:
                _log.warning("comparison status clear failed: %s", exc)
        if emitter is not None and final_render != last_emitted:
            try:
                await emitter(
                    {"type": "replace", "data": {"content": final_render}}
                )
            except Exception as exc:
                _log.warning("comparison final replace emit failed: %s", exc)
        yield final_render

        # Skip metering when no tokens were generated (PRD R-3.5: cold-start
        # timeout or upstream failure must not consume allowance).
        if (prompt_total == 0 and completion_total == 0) or (
            side_timed_out and not first_token_seen
        ):
            return

        metadata = body.get("metadata") or {}
        chat_id = metadata.get("chat_id")
        title = metadata.get("chat_title") or "Untitled"

        # Best-effort: any debit/upsert error is logged and swallowed. The
        # message the user already received must not be invalidated by a
        # metering hiccup.
        try:
            await self._call_api(
                _DEBIT_PATH,
                {
                    "accountEmail": email,
                    "promptTokens": prompt_total,
                    "completionTokens": completion_total,
                    "comparisonMode": True,
                    "conversationId": chat_id if isinstance(chat_id, str) else None,
                },
            )
        except PipeError as exc:
            _log.warning("comparison debit failed: %s", exc)

        if isinstance(chat_id, str) and chat_id:
            try:
                await self._call_api(
                    _UPSERT_PATH,
                    {
                        "accountEmail": email,
                        "owuiChatId": chat_id,
                        "title": title,
                        "lastMessageAt": _now_iso(),
                        "comparisonModeUsed": True,
                        "weightedTokensTotal": prompt_total + 3 * completion_total,
                    },
                )
            except PipeError as exc:
                _log.warning("comparison upsert failed: %s", exc)

    # -- Per-model upstream resolution ------------------------------------

    def _resolve_upstream(self, base_model: str) -> str:
        """Return the configured Modal base URL for `base_model`, or '' if unset.

        Used by both the streaming chat-completion call and the cold-start
        probe so they target the same per-model endpoint. Returning an empty
        string lets the caller decide its own fallback (cold-start probe
        falls back to `cold_start_probe_base_url` then `vllm_base_url`; the
        streaming side falls back directly to `vllm_base_url`).
        """
        if not base_model:
            return ""
        return (self.valves.model_base_urls or {}).get(base_model, "").strip()

    # -- Cold-start probe -------------------------------------------------

    async def _maybe_emit_cold_start(
        self,
        emitter: Callable[[dict[str, Any]], Awaitable[None]] | None,
        base_model: str,
    ) -> bool:
        """Probe the upstream and emit a status indicator if cold. Returns True if emitted.

        Resolution order for the probe URL:
          1. ``valves.model_base_urls[base_model]`` — per-model override.
          2. ``valves.cold_start_probe_base_url`` — single fallback.
          3. ``valves.vllm_base_url`` — legacy single-endpoint fallback.
        """
        if not self.valves.cold_start_enabled or emitter is None:
            return False
        probe_base = (
            self._resolve_upstream(base_model)
            or self.valves.cold_start_probe_base_url
            or self.valves.vllm_base_url
        ).strip()
        if not probe_base:
            return False
        warmth = await _modal_warmth.probe_warmth(
            probe_base,
            timeout_s=self.valves.cold_start_probe_timeout_s,
            warm_threshold_s=self.valves.cold_start_warm_threshold_s,
        )
        if warmth != "cold":
            return False
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
        except Exception as exc:
            _log.warning("cold-start emit failed: %s", exc)
            return False
        return True

    @staticmethod
    async def _wrap_first_token_timeout(
        source: AsyncIterator[tuple[str, dict[str, Any] | None]],
        timeout_s: float,
    ) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
        """Wrap a side stream so the first item must arrive within `timeout_s`."""
        iterator = source.__aiter__()
        first = await asyncio.wait_for(iterator.__anext__(), timeout_s)
        yield first
        async for chunk in iterator:
            yield chunk

    # -- Streaming side ---------------------------------------------------

    async def _stream_completion(
        self,
        *,
        base_model: str,
        body: dict[str, Any],
        entropy_source_type: str,
    ) -> AsyncIterator[tuple[str, dict[str, Any] | None]]:
        """POST a streaming chat-completion to vLLM with a per-side entropy override.

        Yields `(delta_text, usage)` tuples. `delta_text` is the incremental
        text from the latest SSE chunk (may be empty); `usage` is the final
        usage block when present. Both can be falsy on a given chunk — the
        caller appends only non-empty deltas and stores the final usage when
        it arrives.
        """
        request_body = self._build_side_body(body, base_model, entropy_source_type)
        upstream = (self._resolve_upstream(base_model) or self.valves.vllm_base_url).rstrip("/")
        url = upstream + "/chat/completions"
        headers = {"content-type": "application/json"}
        if self.valves.vllm_api_key:
            headers["authorization"] = f"Bearer {self.valves.vllm_api_key}"

        timeout = httpx.Timeout(self.valves.vllm_stream_timeout_s)
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream("POST", url, json=request_body, headers=headers) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    _log.warning("non-JSON SSE chunk skipped: %r", payload[:80])
                    continue
                delta_text = _extract_delta_text(chunk)
                usage = chunk.get("usage") if isinstance(chunk, dict) else None
                yield delta_text, usage if isinstance(usage, dict) else None

    def _build_side_body(
        self,
        body: dict[str, Any],
        base_model: str,
        entropy_source_type: str,
    ) -> dict[str, Any]:
        """Build the per-side vLLM request body.

        Same fields as the user's `body` (so seed/temperature/top_p/top_k/
        max_tokens carry through identically) but with `model` rewritten to
        the real base model and `extra_body.qr_entropy_source_type` set to
        the per-side override. `stream` is forced True regardless of what
        OWUI passed — the Pipe is a streaming surface by design.
        """
        # Strip the dual-column scaffolding from prior assistant turns so
        # this lane is conditioned only on its own past answers — otherwise
        # the model echoes the ``| Quantum | Pseudo-random |`` header on
        # follow-up turns (mirrored from qr-llm-chat iter-58, 2026-06-28).
        lane = _LANE_QUANTUM if entropy_source_type == "quantum_grpc" else _LANE_PSEUDO
        side_body: dict[str, Any] = {}
        for k, v in body.items():
            if k in {"metadata", "model", "stream", "extra_body"}:
                continue
            if k == "messages":
                side_body["messages"] = _sanitize_history_for_lane(v, lane)
                continue
            side_body[k] = v
        side_body["model"] = base_model
        side_body["stream"] = True
        existing_extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
        side_body["extra_body"] = {
            **existing_extra,
            "qr_entropy_source_type": entropy_source_type,
        }
        # Disable Qwen3.6's default thinking-mode output on BOTH sides
        # (quantum + PRNG) regardless of whether upstream injected the
        # kwarg. Explicit-set (rather than allow-list pass-through) keeps
        # the comparison sides symmetric even when a caller bypasses the
        # filter. ``chat_template_kwargs`` is a top-level vLLM
        # ChatCompletionRequest field consumed by the Jinja chat template,
        # not a logits-processor parameter — so it goes on ``side_body``
        # directly, not under ``extra_body`` / ``vllm_xargs``.
        side_body["chat_template_kwargs"] = {"enable_thinking": False}
        return side_body

    # -- entropic.science API ---------------------------------------------

    async def _call_api(
        self,
        path: str,
        json_body: dict[str, Any],
    ) -> dict[str, Any]:
        """Sign an `X-Service-Token` header and POST `json_body` to `<api_base_url><path>`."""
        secrets = _split_secrets(self.valves.service_token_secret)
        if not secrets:
            raise PipeError(
                "SERVICE_TOKEN_SECRETS is not configured; allowance metering is offline.",
            )
        signing_secret = secrets[0]

        url = httpx.URL(self.valves.api_base_url.rstrip("/") + path)
        signed_path = url.path
        token = _sign_service_token(signed_path, signing_secret)

        timeout = httpx.Timeout(self.valves.request_timeout_s)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    str(url),
                    json=json_body,
                    headers={
                        "x-service-token": token,
                        "content-type": "application/json",
                    },
                )
        except httpx.HTTPError as exc:
            raise PipeError(f"entropic.science API unreachable: {exc}") from exc

        if response.status_code >= 400:
            raise PipeError(
                f"entropic.science API returned {response.status_code} for {path}",
            )

        try:
            decoded: Any = response.json()
        except ValueError as exc:
            raise PipeError(f"entropic.science API returned non-JSON for {path}") from exc

        if not isinstance(decoded, dict):
            raise PipeError(f"entropic.science API returned non-object for {path}")
        return decoded


# ---------------------------------------------------------------------------
# SSE chunk decoding
# ---------------------------------------------------------------------------


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    """Pull the assistant-text delta out of one OpenAI-style SSE chunk."""
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
    # Some servers (and the final chunk) carry no delta — return empty.
    return ""
