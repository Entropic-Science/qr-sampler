"""Tests for `examples/open-webui/_modal_warmth.py` warmth probe."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx

from .conftest import load_modal_warmth

if TYPE_CHECKING:
    import pytest

_mod = load_modal_warmth()
probe_warmth = _mod.probe_warmth


def _patch_async_client(monkeypatch: pytest.MonkeyPatch, handler: object) -> None:
    real_async_client = httpx.AsyncClient

    def _factory(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)  # type: ignore[arg-type]
        return real_async_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(_mod.httpx, "AsyncClient", _factory)


class TestProbeWarmth:
    def test_fast_response_is_warm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        _patch_async_client(monkeypatch, handler)
        result = asyncio.run(probe_warmth("http://example/v1", timeout_s=1.0, warm_threshold_s=0.5))
        assert result == "warm"

    def test_slow_response_is_cold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)
            return httpx.Response(200)

        _patch_async_client(monkeypatch, handler)
        # warm_threshold_s smaller than sleep duration → cold
        result = asyncio.run(
            probe_warmth("http://example/v1", timeout_s=1.0, warm_threshold_s=0.01)
        )
        assert result == "cold"

    def test_timeout_is_cold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.5)
            return httpx.Response(200)

        _patch_async_client(monkeypatch, handler)
        result = asyncio.run(
            probe_warmth("http://example/v1", timeout_s=0.05, warm_threshold_s=0.01)
        )
        assert result == "cold"

    def test_connection_refused_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        _patch_async_client(monkeypatch, handler)
        result = asyncio.run(probe_warmth("http://example/v1", timeout_s=1.0, warm_threshold_s=0.5))
        assert result == "unknown"

    def test_5xx_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        _patch_async_client(monkeypatch, handler)
        result = asyncio.run(probe_warmth("http://example/v1", timeout_s=1.0, warm_threshold_s=0.5))
        assert result == "unknown"

    def test_trailing_slash_normalised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            return httpx.Response(200)

        _patch_async_client(monkeypatch, handler)
        asyncio.run(probe_warmth("http://example/v1/", timeout_s=1.0, warm_threshold_s=0.5))
        assert seen == ["/v1/models"]
