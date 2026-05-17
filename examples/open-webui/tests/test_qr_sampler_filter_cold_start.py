"""Tests for `examples/open-webui/qr_sampler_filter.py` cold-start indicator + timeout."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from .conftest import load_filter

filter_mod = load_filter()
Filter = filter_mod.Filter
iter_with_first_token_timeout = filter_mod.iter_with_first_token_timeout


def _wire(monkeypatch: pytest.MonkeyPatch, handler: Any) -> Filter:
    """Build a Filter wired against a single async/sync httpx handler.

    The handler receives EVERY httpx request the filter makes (preflight,
    debit, upsert, cold-start probe). Routing is the test's responsibility.
    """
    flt = Filter()
    flt.valves.api_base_url = "https://api.example/api"
    flt.valves.service_token_secret = "secret-a"
    flt.valves.cold_start_enabled = True
    flt.valves.cold_start_probe_base_url = "https://upstream.example/v1"
    flt.valves.cold_start_probe_timeout_s = 0.5
    flt.valves.cold_start_warm_threshold_s = 0.01
    flt.valves.cold_start_message = "Spinning up — please wait."

    real_async_client = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(filter_mod.httpx, "AsyncClient", _factory)
    return flt


class _Emitter:
    """Captures `__event_emitter__` events for assertion."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _preflight_ok() -> httpx.Response:
    return httpx.Response(
        200, json={"ok": True, "balance": 100_000, "nextRefillAt": "2026-05-17T00:00:00Z"}
    )


class TestColdStartIndicator:
    def test_cold_response_emits_indicator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "upstream.example":
                await asyncio.sleep(0.05)  # > 10 ms warm threshold → cold
                return httpx.Response(200)
            if request.url.path == "/api/allowance/preflight":
                return _preflight_ok()
            return httpx.Response(500, json={"error": "unscripted"})

        flt = _wire(monkeypatch, handler)
        emitter = _Emitter()

        asyncio.run(
            flt.inlet(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "chat-1"},
                },
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            )
        )

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["type"] == "status"
        assert event["data"]["done"] is False
        assert event["data"]["description"] == "Spinning up — please wait."

    def test_warm_response_emits_no_indicator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "upstream.example":
                return httpx.Response(200)  # instant → warm
            if request.url.path == "/api/allowance/preflight":
                return _preflight_ok()
            return httpx.Response(500, json={"error": "unscripted"})

        flt = _wire(monkeypatch, handler)
        emitter = _Emitter()

        asyncio.run(
            flt.inlet(
                {"messages": [{"role": "user", "content": "hi"}]},
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            )
        )
        assert emitter.events == []

    def test_first_token_clears_indicator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "upstream.example":
                await asyncio.sleep(0.05)
                return httpx.Response(200)
            if request.url.path == "/api/allowance/preflight":
                return _preflight_ok()
            return httpx.Response(500, json={"error": "unscripted"})

        flt = _wire(monkeypatch, handler)
        emitter = _Emitter()

        async def scenario() -> None:
            await flt.inlet(
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "chat-x"},
                },
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            )
            # First chunk has empty content — indicator stays.
            await flt.stream(
                {"chat_id": "chat-x", "choices": [{"delta": {"content": ""}}]},
                __event_emitter__=emitter,
            )
            assert len(emitter.events) == 1
            # First real token — indicator cleared.
            await flt.stream(
                {"chat_id": "chat-x", "choices": [{"delta": {"content": "Hello"}}]},
                __event_emitter__=emitter,
            )
            # Subsequent tokens — no duplicate clear.
            await flt.stream(
                {"chat_id": "chat-x", "choices": [{"delta": {"content": " world"}}]},
                __event_emitter__=emitter,
            )

        asyncio.run(scenario())
        assert len(emitter.events) == 2
        assert emitter.events[1]["type"] == "status"
        assert emitter.events[1]["data"]["done"] is True

    def test_preflight_still_called_when_cold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.host == "upstream.example":
                await asyncio.sleep(0.05)
                return httpx.Response(200)
            if request.url.path == "/api/allowance/preflight":
                return _preflight_ok()
            return httpx.Response(500, json={"error": "unscripted"})

        flt = _wire(monkeypatch, handler)

        asyncio.run(
            flt.inlet(
                {"messages": [{"role": "user", "content": "hi"}]},
                __user__={"email": "u@example.com"},
                __event_emitter__=_Emitter(),
            )
        )

        api_calls = [c for c in captured if c.url.host == "api.example"]
        assert len(api_calls) == 1
        assert api_calls[0].url.path == "/api/allowance/preflight"


class TestFirstTokenTimeout:
    def test_helper_raises_when_first_chunk_is_slow(self) -> None:
        async def slow_source() -> Any:
            await asyncio.sleep(0.1)
            yield "late"

        async def scenario() -> None:
            wrapped = iter_with_first_token_timeout(slow_source(), timeout_s=0.01)
            async for _ in wrapped:
                pass

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(scenario())

    def test_helper_passes_fast_chunks(self) -> None:
        async def fast_source() -> Any:
            yield "a"
            yield "b"
            yield "c"

        async def scenario() -> list[str]:
            collected: list[str] = []
            wrapped = iter_with_first_token_timeout(fast_source(), timeout_s=1.0)
            async for item in wrapped:
                collected.append(item)
            return collected

        assert asyncio.run(scenario()) == ["a", "b", "c"]

    def test_outlet_skips_debit_when_timed_out(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When `mark_first_token_timeout` is called, outlet must skip the debit."""
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(500, json={"error": "should not be reached"})

        flt = _wire(monkeypatch, handler)
        flt.mark_first_token_timeout("chat-timeout")

        result = asyncio.run(
            flt.outlet(
                {
                    "metadata": {"chat_id": "chat-timeout"},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0},
                },
                __user__={"email": "u@example.com"},
            )
        )

        assert result["metadata"]["chat_id"] == "chat-timeout"
        assert captured == []

    def test_outlet_debits_normally_when_not_timed_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.path in (
                "/api/allowance/debit",
                "/api/conversations/upsert",
            ):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(500, json={"error": "unscripted"})

        flt = _wire(monkeypatch, handler)

        asyncio.run(
            flt.outlet(
                {
                    "metadata": {"chat_id": "chat-good"},
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                __user__={"email": "u@example.com"},
            )
        )

        paths = [c.url.path for c in captured]
        assert "/api/allowance/debit" in paths
        assert "/api/conversations/upsert" in paths


class TestColdStartDisabled:
    """When `cold_start_enabled` is False the filter behaves as in v0.2."""

    def test_no_probe_no_emit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            if request.url.path == "/api/allowance/preflight":
                return _preflight_ok()
            return httpx.Response(599)

        flt = _wire(monkeypatch, handler)
        flt.valves.cold_start_enabled = False

        emitter = _Emitter()
        asyncio.run(
            flt.inlet(
                {"messages": [{"role": "user", "content": "hi"}]},
                __user__={"email": "u@example.com"},
                __event_emitter__=emitter,
            )
        )

        upstream_calls = [c for c in captured if c.url.host == "upstream.example"]
        assert upstream_calls == []
        assert emitter.events == []
