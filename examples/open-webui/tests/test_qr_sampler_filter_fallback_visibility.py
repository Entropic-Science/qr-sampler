"""Tests for the fallback-visibility hook on `Filter.outlet` (plan R2).

Asserts that when the response carries
``qr_metadata.last_source_used == "system"`` but the configured primary is
``QR_ENTROPY_SOURCE_TYPE=quantum_grpc``, the filter emits exactly one
warning status event per chat. Verifies the dedupe, the silent-no-op
paths (missing metadata, missing emitter, missing env var, matching
source), and that the warning does not depend on the user being signed
in (the OWUI-only deploy profile has no email gate).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from .conftest import load_filter

if TYPE_CHECKING:
    import pytest

filter_mod = load_filter()
Filter = filter_mod.Filter


class _Emitter:
    """Captures `__event_emitter__` events for assertion."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)


def _make_filter() -> Any:
    flt = Filter()
    # Disable allowance metering and integration-profile noise — these tests
    # focus exclusively on the fallback-visibility branch of outlet().
    flt.valves.api_base_url = "https://unused.example/api"
    flt.valves.service_token_secret = "unused"
    flt.valves.cold_start_enabled = False
    return flt


def _outlet_body(
    *,
    chat_id: str = "chat-1",
    last_source_used: str | None = "system",
    usage_tokens: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "metadata": {"chat_id": chat_id},
        "usage": {
            "prompt_tokens": usage_tokens[0],
            "completion_tokens": usage_tokens[1],
        },
    }
    if last_source_used is not None:
        body["qr_metadata"] = {"last_source_used": last_source_used}
    return body


class TestFallbackVisibility:
    def test_warns_when_fallback_to_system(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )
        )

        assert len(emitter.events) == 1
        event = emitter.events[0]
        assert event["type"] == "status"
        assert event["data"]["level"] == "warning"
        assert "pseudo-random" in event["data"]["description"]

    def test_dedupes_within_same_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        async def _run() -> None:
            await flt.outlet(
                _outlet_body(chat_id="chat-A", last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )
            await flt.outlet(
                _outlet_body(chat_id="chat-A", last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )

        asyncio.run(_run())
        assert len(emitter.events) == 1, (
            "second outlet call in the same chat must not re-emit the warning"
        )

    def test_distinct_chats_warn_independently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        async def _run() -> None:
            await flt.outlet(
                _outlet_body(chat_id="chat-A", last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )
            await flt.outlet(
                _outlet_body(chat_id="chat-B", last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )

        asyncio.run(_run())
        assert len(emitter.events) == 2

    def test_silent_when_source_matches_primary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="quantum_grpc"),
                __user__=None,
                __event_emitter__=emitter,
            )
        )
        assert emitter.events == []

    def test_silent_when_metadata_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Legacy vLLM serve layer that does not attach qr_metadata must
        not trip the warning (forward compatibility, plan R2 backstop)."""
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used=None),
                __user__=None,
                __event_emitter__=emitter,
            )
        )
        assert emitter.events == []

    def test_silent_when_env_var_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No configured primary -> nothing to compare against -> no warn."""
        monkeypatch.delenv("QR_ENTROPY_SOURCE_TYPE", raising=False)
        flt = _make_filter()
        emitter = _Emitter()

        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )
        )
        assert emitter.events == []

    def test_silent_when_emitter_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No event channel from OWUI -> nowhere to send the warning."""
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()

        result = asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__=None,
                __event_emitter__=None,
            )
        )
        # Body returned unmodified; no exception.
        assert result["metadata"]["chat_id"] == "chat-1"

    def test_fires_without_signed_in_email(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fallback warning runs *before* the email gate so OWUI-only
        deploys (no entropic.science integration) still see it."""
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()
        emitter = _Emitter()

        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__={},  # no email -> outlet returns early before debit
                __event_emitter__=emitter,
            )
        )
        assert len(emitter.events) == 1

    def test_emitter_exception_does_not_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A flaky emitter must not break the response path."""
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        flt = _make_filter()

        async def _bad_emitter(_event: dict[str, Any]) -> None:
            raise RuntimeError("simulated UI failure")

        result = asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__=None,
                __event_emitter__=_bad_emitter,
            )
        )
        # outlet still returned the body; the failed emit was swallowed.
        assert result["metadata"]["chat_id"] == "chat-1"
        # And because the emit failed, dedup did NOT record success; a
        # subsequent retry with a working emitter would try again.
        emitter = _Emitter()
        asyncio.run(
            flt.outlet(
                _outlet_body(last_source_used="system"),
                __user__=None,
                __event_emitter__=emitter,
            )
        )
        assert len(emitter.events) == 1
