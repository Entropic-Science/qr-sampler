"""Tests for `examples/open-webui/qr_sampler_filter.py` allowance metering.

Covers the four scenarios spec §10.2 calls out — sufficient, insufficient,
email-not-verified, model-unavailable — plus the load-bearing edges:
service-token signature fixture, missing-email rejection, outlet best-effort
debit + upsert, swallowed-error behaviour. The filter file is not part of the
`qr_sampler` Python package (it lives in `examples/open-webui/`), so we load
it via `importlib.util` at the top of this module.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_FILTER_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "open-webui" / "qr_sampler_filter.py"
)


def _load_filter_module() -> Any:
    spec = importlib.util.spec_from_file_location("qr_sampler_owui_filter", _FILTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["qr_sampler_owui_filter"] = module
    spec.loader.exec_module(module)
    return module


filter_mod = _load_filter_module()
Filter = filter_mod.Filter
FilterError = filter_mod.FilterError
_sign_service_token = filter_mod._sign_service_token


# ---------------------------------------------------------------------------
# Mock-transport helpers
# ---------------------------------------------------------------------------


class _RecordingHandler:
    """Captures every httpx request and returns scripted JSON responses.

    `responses` maps a URL path to a list of `(status_code, body)` tuples,
    one per call (consumed in order). Any request to an unscripted path
    short-circuits to 500 so tests catch unexpected calls loudly.
    """

    def __init__(self, responses: dict[str, list[tuple[int, dict[str, Any]]]]) -> None:
        self._responses = responses
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        scripted = self._responses.get(request.url.path)
        if not scripted:
            return httpx.Response(500, json={"error": "unscripted path"})
        status, body = scripted.pop(0)
        return httpx.Response(status, json=body)


def _patched_filter(monkeypatch: pytest.MonkeyPatch, handler: _RecordingHandler) -> Any:
    """Build a Filter with a Valves stub and AsyncClient pointed at the handler."""
    flt = Filter()
    flt.valves.api_base_url = "https://api.example/api"
    flt.valves.service_token_secret = "secret-a,secret-b"

    real_async_client = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(filter_mod.httpx, "AsyncClient", _factory)
    return flt


# ---------------------------------------------------------------------------
# Service-token signing
# ---------------------------------------------------------------------------


class TestSignServiceToken:
    """Pin the HMAC wire format so the entropic.science server can verify."""

    def test_matches_known_fixture(self) -> None:
        secret = "fixture-secret"
        ts = 1700000000
        path = "/api/allowance/preflight"
        expected_hmac = hmac.new(
            secret.encode("utf-8"),
            f"{ts}{path}".encode(),
            hashlib.sha256,
        ).hexdigest()
        token = _sign_service_token(path, secret, unix_ts=ts)
        assert token == f"{ts}.{expected_hmac}"

    def test_path_change_changes_signature(self) -> None:
        ts = 1700000000
        a = _sign_service_token("/api/allowance/preflight", "s", unix_ts=ts)
        b = _sign_service_token("/api/allowance/debit", "s", unix_ts=ts)
        assert a != b

    def test_first_secret_signs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A multi-entry secret vector signs with the FIRST entry."""

        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (200, {"ok": True, "balance": 128000, "nextRefillAt": "2026-05-17T00:00:00Z"})
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)
        flt.valves.service_token_secret = "primary,secondary,tertiary"

        import asyncio

        asyncio.run(
            flt.inlet(
                {"messages": [{"role": "user", "content": "hi"}]},
                __user__={"email": "user@example.com"},
            )
        )

        token = handler.calls[0].headers["x-service-token"]
        ts_str, provided_hmac = token.split(".")
        ts = int(ts_str)
        expected = hmac.new(
            b"primary",
            f"{ts}/api/allowance/preflight".encode(),
            hashlib.sha256,
        ).hexdigest()
        assert provided_hmac == expected


# ---------------------------------------------------------------------------
# inlet() scenarios
# ---------------------------------------------------------------------------


class TestInletPreflight:
    def test_sufficient_injects_qr_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": True,
                            "balance": 100_000,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)
        body = {
            "messages": [{"role": "user", "content": "Tell me about quantum sampling."}],
            "stream": True,
        }

        import asyncio

        result = asyncio.run(flt.inlet(body, __user__={"email": "user@example.com"}))

        assert result["stream"] is True
        assert result["qr_sample_count"] == flt.valves.sample_count
        assert result["qr_fixed_temperature"] == flt.valves.fixed_temperature
        # Preflight payload shape
        req = handler.calls[0]
        decoded = json.loads(req.content)
        assert decoded["accountEmail"] == "user@example.com"
        assert decoded["comparisonMode"] is False
        assert decoded["estimatedPromptTokens"] >= 1

    def test_insufficient_raises_with_waitlist_markdown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": False,
                            "reason": "insufficient",
                            "balance": 0,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(
                flt.inlet(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    __user__={"email": "user@example.com"},
                )
            )
        msg = str(excinfo.value)
        assert "Daily allowance used up" in msg
        assert "2026-05-17 00:00" in msg
        assert "entropic.science/account/waitlist?from=allowance" in msg

    def test_email_not_verified_raises_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": False,
                            "reason": "email_not_verified",
                            "balance": 128000,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(
                flt.inlet(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    __user__={"email": "user@example.com"},
                )
            )
        assert "email_not_verified" in str(excinfo.value)

    def test_model_unavailable_raises_generic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": False,
                            "reason": "model_unavailable",
                            "balance": 128000,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(
                flt.inlet(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    __user__={"email": "user@example.com"},
                )
            )
        assert "model_unavailable" in str(excinfo.value)

    def test_comparison_mode_metadata_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": True,
                            "balance": 100_000,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"qr_comparison_mode": True},
        }

        import asyncio

        asyncio.run(flt.inlet(body, __user__={"email": "user@example.com"}))

        decoded = json.loads(handler.calls[0].content)
        assert decoded["comparisonMode"] is True

    def test_missing_email_raises_clearly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})  # no calls expected
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(flt.inlet({"messages": [{"role": "user", "content": "hi"}]}, __user__=None))
        assert "signed-in account" in str(excinfo.value)
        assert handler.calls == []

    def test_missing_secret_raises_offline_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [
                    (
                        200,
                        {
                            "ok": True,
                            "balance": 100_000,
                            "nextRefillAt": "2026-05-17T00:00:00Z",
                        },
                    )
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)
        flt.valves.service_token_secret = ""

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(
                flt.inlet(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    __user__={"email": "user@example.com"},
                )
            )
        assert "SERVICE_TOKEN_SECRETS" in str(excinfo.value)
        assert handler.calls == []

    def test_api_5xx_raises_filter_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {"/api/allowance/preflight": [(503, {"error": "Service Unavailable"})]}
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        with pytest.raises(FilterError) as excinfo:
            asyncio.run(
                flt.inlet(
                    {"messages": [{"role": "user", "content": "hi"}]},
                    __user__={"email": "user@example.com"},
                )
            )
        assert "503" in str(excinfo.value)

    def test_disabled_filter_bypasses_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})
        flt = _patched_filter(monkeypatch, handler)
        flt.valves.enable_qr_sampling = False

        import asyncio

        result = asyncio.run(
            flt.inlet(
                {"messages": [{"role": "user", "content": "hi"}]},
                __user__=None,  # no email needed when disabled
            )
        )
        assert result == {"messages": [{"role": "user", "content": "hi"}]}
        assert handler.calls == []


# ---------------------------------------------------------------------------
# outlet() scenarios
# ---------------------------------------------------------------------------


class TestOutletDebitAndUpsert:
    def _outlet_body(self, *, comparison: bool = False) -> dict[str, Any]:
        return {
            "usage": {"prompt_tokens": 50, "completion_tokens": 200},
            "metadata": {
                "chat_id": "chat-abc",
                "chat_title": "Quantum talk",
                "qr_comparison_mode": comparison,
            },
        }

    def test_debit_and_upsert_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/debit": [
                    (
                        200,
                        {
                            "ok": True,
                            "debited": 650,
                            "balance": 127_350,
                            "ledgerId": "11111111-1111-1111-1111-111111111111",
                        },
                    )
                ],
                "/api/conversations/upsert": [
                    (
                        200,
                        {"id": "22222222-2222-2222-2222-222222222222", "created": True},
                    )
                ],
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        asyncio.run(flt.outlet(self._outlet_body(), __user__={"email": "user@example.com"}))

        # Two API calls in order: debit then upsert
        assert [c.url.path for c in handler.calls] == [
            "/api/allowance/debit",
            "/api/conversations/upsert",
        ]
        debit_body = json.loads(handler.calls[0].content)
        assert debit_body == {
            "accountEmail": "user@example.com",
            "promptTokens": 50,
            "completionTokens": 200,
            "comparisonMode": False,
            "conversationId": "chat-abc",
        }
        upsert_body = json.loads(handler.calls[1].content)
        assert upsert_body["accountEmail"] == "user@example.com"
        assert upsert_body["owuiChatId"] == "chat-abc"
        assert upsert_body["title"] == "Quantum talk"
        assert upsert_body["comparisonModeUsed"] is False
        assert upsert_body["weightedTokensTotal"] == 50 + 3 * 200
        assert "lastMessageAt" in upsert_body and upsert_body["lastMessageAt"].endswith("Z")

    def test_comparison_mode_flag_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/debit": [
                    (200, {"ok": True, "debited": 1300, "balance": 126_700, "ledgerId": None})
                ],
                "/api/conversations/upsert": [(200, {"id": "id", "created": False})],
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        asyncio.run(flt.outlet(self._outlet_body(comparison=True), __user__={"email": "u@e.com"}))

        debit_body = json.loads(handler.calls[0].content)
        upsert_body = json.loads(handler.calls[1].content)
        assert debit_body["comparisonMode"] is True
        assert upsert_body["comparisonModeUsed"] is True

    def test_debit_error_does_not_block_upsert_or_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A debit 5xx is logged but does not prevent the upsert nor raise."""
        handler = _RecordingHandler(
            {
                "/api/allowance/debit": [(500, {"error": "boom"})],
                "/api/conversations/upsert": [(200, {"id": "id", "created": True})],
            }
        )
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        result = asyncio.run(flt.outlet(self._outlet_body(), __user__={"email": "u@e.com"}))
        # Returns body unmodified despite the debit failure
        assert "usage" in result
        # Both calls were attempted
        assert [c.url.path for c in handler.calls] == [
            "/api/allowance/debit",
            "/api/conversations/upsert",
        ]

    def test_zero_usage_skips_all_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        body = {
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            "metadata": {"chat_id": "x"},
        }
        asyncio.run(flt.outlet(body, __user__={"email": "u@e.com"}))
        assert handler.calls == []

    def test_missing_email_in_outlet_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})
        flt = _patched_filter(monkeypatch, handler)

        import asyncio

        asyncio.run(flt.outlet(self._outlet_body(), __user__=None))
        assert handler.calls == []

    def test_missing_chat_id_skips_upsert_but_still_debits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/debit": [
                    (200, {"ok": True, "debited": 650, "balance": 100, "ledgerId": "x"})
                ]
            }
        )
        flt = _patched_filter(monkeypatch, handler)
        body = {
            "usage": {"prompt_tokens": 50, "completion_tokens": 200},
            "metadata": {},  # no chat_id
        }

        import asyncio

        asyncio.run(flt.outlet(body, __user__={"email": "u@e.com"}))
        assert [c.url.path for c in handler.calls] == ["/api/allowance/debit"]
        debit_body = json.loads(handler.calls[0].content)
        assert debit_body["conversationId"] is None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestPromptTokenEstimate:
    def test_string_content(self) -> None:
        body = {"messages": [{"role": "user", "content": "x" * 40}]}
        assert filter_mod._estimate_prompt_tokens(body) == 10  # 40 / 4

    def test_multipart_text_content(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "x" * 20},
                        {"type": "image_url", "image_url": {"url": "https://x"}},
                        {"type": "text", "text": "y" * 20},
                    ],
                }
            ]
        }
        assert filter_mod._estimate_prompt_tokens(body) == 10  # 40 / 4

    def test_minimum_floor_is_one(self) -> None:
        assert filter_mod._estimate_prompt_tokens({"messages": []}) == 1

    def test_handles_missing_messages_key(self) -> None:
        assert filter_mod._estimate_prompt_tokens({}) == 1


class TestRenderOutOfAllowanceMarkdown:
    def test_includes_refill_time_and_waitlist(self) -> None:
        md = filter_mod._render_out_of_allowance_markdown(
            {"nextRefillAt": "2099-01-01T00:00:00Z", "balance": 0}
        )
        assert "Daily allowance used up" in md
        assert "2099-01-01 00:00 UTC" in md
        assert "[Register interest →]" in md
        assert "entropic.science/account/waitlist?from=allowance" in md

    def test_falls_back_when_timestamp_missing(self) -> None:
        md = filter_mod._render_out_of_allowance_markdown({"balance": 0})
        # Should still render the waitlist CTA without crashing on parse failure
        assert "[Register interest →]" in md


class TestCoerceNonnegInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            (-3, 0),
            ("12", 12),
            ("abc", 0),
            (None, 0),
            (3.7, 3),
            ({}, 0),
        ],
    )
    def test_coercion(self, value: Any, expected: int) -> None:
        assert filter_mod._coerce_nonneg_int(value) == expected


class TestSplitSecrets:
    def test_trims_whitespace_and_drops_empties(self) -> None:
        assert filter_mod._split_secrets(" a , b ,, c ") == ["a", "b", "c"]

    def test_empty_returns_empty(self) -> None:
        assert filter_mod._split_secrets("") == []
        assert filter_mod._split_secrets("   ") == []


class TestReadComparisonFlag:
    def test_metadata_takes_precedence(self) -> None:
        assert (
            filter_mod._read_comparison_flag(
                {"metadata": {"qr_comparison_mode": True}, "qr_comparison_mode": False}
            )
            is True
        )

    def test_falls_back_to_top_level(self) -> None:
        assert filter_mod._read_comparison_flag({"qr_comparison_mode": True}) is True

    def test_default_false(self) -> None:
        assert filter_mod._read_comparison_flag({}) is False
