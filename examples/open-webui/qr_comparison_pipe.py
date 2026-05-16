"""
title: QR vs PRNG Comparison
author: qr-sampler
author_url: https://github.com/alchemystack/qr-sampler
version: 0.1.0
license: MIT
description: Streaming dual-column comparison of quantum vs pseudo-random sampling.
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
    from collections.abc import AsyncIterator

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
# Multiplication-sign and middle-dot are intentional per spec section 5.4 markdown.
_INTRO_BLOCKQUOTE = (
    "> Comparison mode is on. Same prompt, same model, only the random source differs.\n"
    f">\n> **Left = {_QUANTUM_LABEL}** \u00b7 **Right = {_PRNG_LABEL}** \u00b7 ~2\u00d7 usage"
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

    Per spec §5.4, the message content is re-emitted on every delta tick with
    the full current state of both buffers. Pipe characters in the buffer
    contents would break the table layout, so they are escaped to `\\|`.
    Newlines inside a buffer are converted to `<br>` because markdown table
    cells cannot contain raw newlines.
    """
    safe_left = _escape_for_table_cell(left)
    safe_right = _escape_for_table_cell(right)
    body = (
        f"{_INTRO_BLOCKQUOTE}\n\n"
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


def _strip_pseudo_suffix(model_id: str) -> str:
    """`gemma-4-31b-reasoning--qr-vs-prng` → `gemma-4-31b-reasoning`."""
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
            default="gemma-4-31b-reasoning,qwen-3.6-27b-reasoning",
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

    def __init__(self) -> None:
        self.valves = self.Valves()
        # Manifold Pipe: OWUI groups entries under this name in the selector.
        self.type = "manifold"
        self.name = "QR vs PRNG · "
        self.id = "qr_vs_prng"

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
    ) -> AsyncIterator[str]:
        """Stream the dual-column comparison.

        Yields raw markdown chunks. OWUI's manifold-Pipe contract treats each
        yielded string as a delta to render into the assistant message. We
        overwrite the full message content on every tick so the table grows
        live in both columns.
        """
        async for chunk in self._run(body, __user__):
            yield chunk

    # -- Orchestration ----------------------------------------------------

    async def _run(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None,
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

        l_buf: list[str] = []
        r_buf: list[str] = []
        l_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        r_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        delta_event = asyncio.Event()

        # Run both sides concurrently. Each side appends to its buffer and
        # signals `delta_event` so the emitter loop wakes up and re-renders.
        async def consume_side(
            entropy_source: str,
            buf: list[str],
            usage_out: dict[str, int],
        ) -> None:
            try:
                async for delta_text, usage in self._stream_completion(
                    base_model=base_model,
                    body=body,
                    entropy_source_type=entropy_source,
                ):
                    if delta_text:
                        buf.append(delta_text)
                    if usage:
                        usage_out["prompt_tokens"] = _coerce_nonneg_int(usage.get("prompt_tokens"))
                        usage_out["completion_tokens"] = _coerce_nonneg_int(
                            usage.get("completion_tokens")
                        )
                    delta_event.set()
            except Exception as exc:
                # Surface in-column, keep the other side running.
                _log.warning("comparison side %s errored: %s", entropy_source, exc)
                buf.append(f"[stream error: {exc}]")
                delta_event.set()

        left_task = asyncio.create_task(
            consume_side("quantum_grpc", l_buf, l_usage),
        )
        right_task = asyncio.create_task(
            consume_side("system", r_buf, r_usage),
        )

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
                if rendered != last_emitted:
                    yield rendered
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
        if final_render != last_emitted:
            yield final_render

        # Skip metering when nothing was actually produced (defensive against
        # vLLM not returning usage on certain error paths).
        if prompt_total == 0 and completion_total == 0:
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
        url = self.valves.vllm_base_url.rstrip("/") + "/chat/completions"
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
        side_body: dict[str, Any] = {
            k: v for k, v in body.items() if k not in {"metadata", "model", "stream", "extra_body"}
        }
        side_body["model"] = base_model
        side_body["stream"] = True
        existing_extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
        side_body["extra_body"] = {
            **existing_extra,
            "qr_entropy_source_type": entropy_source_type,
        }
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
