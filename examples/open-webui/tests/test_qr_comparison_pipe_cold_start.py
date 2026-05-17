"""Tests for `examples/open-webui/qr_comparison_pipe.py` cold-start handling.

Covers the contract surface for the Pipe's cold-start mechanism:
- Single indicator per prompt (probe runs once, shared by both columns).
- Indicator cleared on the first delta from EITHER side.
- Double-debit still fires on success (existing behaviour preserved).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from .conftest import load_pipe

if TYPE_CHECKING:
    import pytest

pipe_mod = load_pipe()
Pipe = pipe_mod.Pipe


class _Emitter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _wire(monkeypatch: pytest.MonkeyPatch, merged_handler: Any) -> Pipe:
    p = Pipe()
    p.valves.api_base_url = "https://api.example/api"
    p.valves.service_token_secret = "s-1"
    p.valves.vllm_base_url = "https://upstream.example/v1"
    p.valves.vllm_api_key = ""
    p.valves.cold_start_enabled = True
    p.valves.cold_start_probe_base_url = "https://upstream.example/v1"
    p.valves.cold_start_probe_timeout_s = 0.5
    p.valves.cold_start_warm_threshold_s = 0.01
    p.valves.cold_start_message = "Spinning up."
    p.valves.cold_start_first_token_timeout_s = 5.0

    real_async_client = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(merged_handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(pipe_mod.httpx, "AsyncClient", _factory)
    return p


def _sse_lines(*chunks: dict[str, Any]) -> bytes:
    """Encode a list of OpenAI-style SSE chunks into the wire format."""
    parts: list[str] = []
    for c in chunks:
        import json as _json

        parts.append(f"data: {_json.dumps(c)}\n\n")
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode("utf-8")


def _delta_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def _final_usage_chunk(prompt: int, completion: int) -> dict[str, Any]:
    return {
        "choices": [{"delta": {}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class TestColdStartIndicator:
    def test_single_indicator_per_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Probe runs once per prompt; both columns share the wake."""
        probe_calls: list[str] = []
        chat_calls: list[str] = []

        async def merged(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD" and request.url.path == "/v1/models":
                probe_calls.append(str(request.url))
                await asyncio.sleep(0.05)  # slower than 10 ms threshold → cold
                return httpx.Response(200)
            if request.url.path == "/v1/chat/completions":
                chat_calls.append(str(request.url))
                payload = _sse_lines(
                    _delta_chunk("hi"),
                    _final_usage_chunk(prompt=5, completion=2),
                )
                return httpx.Response(200, content=payload)
            if request.url.path == "/api/allowance/preflight":
                return httpx.Response(
                    200, json={"ok": True, "balance": 100_000, "nextRefillAt": ""}
                )
            if request.url.path in (
                "/api/allowance/debit",
                "/api/conversations/upsert",
            ):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(500, json={"error": "unscripted"})

        p = _wire(monkeypatch, merged)
        emitter = _Emitter()

        async def scenario() -> list[str]:
            out: list[str] = []
            async for chunk in p.pipe(
                {
                    "model": "gemma-4-31b-reasoning--qr-vs-prng",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "chat-1"},
                },
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            ):
                out.append(chunk)
            return out

        asyncio.run(scenario())

        # Exactly one probe regardless of side count.
        assert len(probe_calls) == 1
        # Two chat completions (one per side).
        assert len(chat_calls) == 2
        # Indicator emitted once + cleared once = 2 status events.
        status_events = [e for e in emitter.events if e.get("type") == "status"]
        assert len(status_events) == 2
        assert status_events[0]["data"]["done"] is False
        assert status_events[1]["data"]["done"] is True

    def test_warm_path_emits_no_indicator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def merged(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD" and request.url.path == "/v1/models":
                return httpx.Response(200)  # instant → warm
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(
                    200,
                    content=_sse_lines(
                        _delta_chunk("ok"),
                        _final_usage_chunk(prompt=4, completion=1),
                    ),
                )
            if request.url.path == "/api/allowance/preflight":
                return httpx.Response(
                    200, json={"ok": True, "balance": 100_000, "nextRefillAt": ""}
                )
            if request.url.path in (
                "/api/allowance/debit",
                "/api/conversations/upsert",
            ):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(500, json={"error": "unscripted"})

        p = _wire(monkeypatch, merged)
        emitter = _Emitter()

        async def scenario() -> None:
            async for _ in p.pipe(
                {
                    "model": "gemma-4-31b-reasoning--qr-vs-prng",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            ):
                pass

        asyncio.run(scenario())
        status_events = [e for e in emitter.events if e.get("type") == "status"]
        assert status_events == []


class TestDoubleDebit:
    def test_debit_fires_with_summed_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        debit_payloads: list[dict[str, Any]] = []

        async def merged(request: httpx.Request) -> httpx.Response:
            if request.method == "HEAD" and request.url.path == "/v1/models":
                return httpx.Response(200)
            if request.url.path == "/v1/chat/completions":
                return httpx.Response(
                    200,
                    content=_sse_lines(
                        _delta_chunk("hi"),
                        _final_usage_chunk(prompt=10, completion=5),
                    ),
                )
            if request.url.path == "/api/allowance/preflight":
                return httpx.Response(
                    200, json={"ok": True, "balance": 100_000, "nextRefillAt": ""}
                )
            if request.url.path == "/api/allowance/debit":
                debit_payloads.append(_decode_json(request))
                return httpx.Response(200, json={"ok": True})
            if request.url.path == "/api/conversations/upsert":
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(500, json={"error": "unscripted"})

        p = _wire(monkeypatch, merged)
        p.valves.cold_start_enabled = False  # warm path; isolating debit logic

        async def scenario() -> None:
            async for _ in p.pipe(
                {
                    "model": "gemma-4-31b-reasoning--qr-vs-prng",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "chat-99"},
                },
                __user__={"email": "u@example.com"},
            ):
                pass

        asyncio.run(scenario())

        assert len(debit_payloads) == 1
        # Both sides contribute: prompt=20, completion=10.
        assert debit_payloads[0]["promptTokens"] == 20
        assert debit_payloads[0]["completionTokens"] == 10
        assert debit_payloads[0]["comparisonMode"] is True


def _decode_json(request: httpx.Request) -> dict[str, Any]:
    import json as _json

    return _json.loads(request.content)  # type: ignore[no-any-return]
