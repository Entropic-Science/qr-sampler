"""Tests for `examples/open-webui/qr_comparison_pipe.py` streaming dual-column.

Covers the seven scenarios spec §5.4 calls out — pipes() registry,
streaming fan-out + extra_body wiring, intermediate yields, final yield
with summed usage, comparison-flagged debit after streams close, preflight
insufficient short-circuit, one-side stream error tolerance.

The pipe file is not part of the `qr_sampler` Python package (it lives in
`examples/open-webui/`), so we load it via `importlib.util` at the top of
this module — same pattern as `test_open_webui_integration.py`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_PIPE_PATH = (
    Path(__file__).resolve().parent.parent / "examples" / "open-webui" / "qr_comparison_pipe.py"
)


def _load_pipe_module() -> Any:
    spec = importlib.util.spec_from_file_location("qr_sampler_owui_comparison_pipe", _PIPE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["qr_sampler_owui_comparison_pipe"] = module
    spec.loader.exec_module(module)
    return module


pipe_mod = _load_pipe_module()
Pipe = pipe_mod.Pipe
PipeError = pipe_mod.PipeError


# ---------------------------------------------------------------------------
# Mock-transport helpers
# ---------------------------------------------------------------------------


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    """Encode an OpenAI-style SSE stream from a list of chunk dicts.

    Mirrors the wire format vLLM emits: `data: {json}\\n\\n` per chunk,
    terminated by `data: [DONE]\\n\\n`.
    """
    parts = []
    for c in chunks:
        parts.append(f"data: {json.dumps(c)}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _delta_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def _final_usage_chunk(prompt_tokens: int, completion_tokens: int) -> dict[str, Any]:
    return {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _RecordingHandler:
    """Captures every httpx request and returns scripted JSON or SSE responses.

    `responses` maps a URL path to a list of `(status_code, body)` tuples,
    one per call. A `body` of type `bytes` is treated as a streaming body
    (used for `/v1/chat/completions`); a `dict` is JSON-encoded. Unknown
    paths return 500.
    """

    def __init__(
        self,
        responses: dict[str, list[tuple[int, dict[str, Any] | bytes]]],
    ) -> None:
        self._responses = responses
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        scripted = self._responses.get(request.url.path)
        if not scripted:
            return httpx.Response(500, json={"error": f"unscripted path {request.url.path}"})
        status, body = scripted.pop(0)
        if isinstance(body, bytes):
            return httpx.Response(
                status,
                content=body,
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(status, json=body)


def _patched_pipe(monkeypatch: pytest.MonkeyPatch, handler: _RecordingHandler) -> Any:
    """Build a Pipe with Valves stubs and an httpx.AsyncClient pointed at the handler."""
    p = Pipe()
    p.valves.api_base_url = "https://api.example/api"
    p.valves.service_token_secret = "secret-a,secret-b"
    p.valves.vllm_base_url = "https://vllm.example/v1"
    p.valves.vllm_api_key = "vllm-key"
    p.valves.base_models = "gemma-4-31b-reasoning,qwen-3.6-27b-reasoning"

    real_async_client = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(pipe_mod.httpx, "AsyncClient", _factory)
    return p


async def _collect(agen: Any) -> list[Any]:
    """Drain an async generator into a list."""
    out: list[Any] = []
    async for item in agen:
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# pipes() registry
# ---------------------------------------------------------------------------


class TestPipesRegistry:
    def test_default_two_entries(self) -> None:
        p = Pipe()
        entries = p.pipes()
        ids = [e["id"] for e in entries]
        assert ids == [
            "gemma-4-31b-reasoning--qr-vs-prng",
            "qwen-3.6-27b-reasoning--qr-vs-prng",
        ]
        # Display name carries the base + the comparison hint.
        assert all("Quantum vs Pseudo-random" in e["name"] for e in entries)

    def test_one_entry_per_base_model(self) -> None:
        p = Pipe()
        p.valves.base_models = "single-model"
        entries = p.pipes()
        assert entries == [
            {"id": "single-model--qr-vs-prng", "name": "single-model (Quantum vs Pseudo-random)"}
        ]

    def test_whitespace_in_registry_tolerated(self) -> None:
        p = Pipe()
        p.valves.base_models = " a , b , c "
        entries = p.pipes()
        assert [e["id"] for e in entries] == [
            "a--qr-vs-prng",
            "b--qr-vs-prng",
            "c--qr-vs-prng",
        ]

    def test_empty_registry(self) -> None:
        p = Pipe()
        p.valves.base_models = ""
        assert p.pipes() == []


# ---------------------------------------------------------------------------
# Streaming fan-out
# ---------------------------------------------------------------------------


def _ok_preflight() -> tuple[int, dict[str, Any]]:
    return (
        200,
        {
            "ok": True,
            "balance": 100_000,
            "nextRefillAt": "2026-05-17T00:00:00Z",
        },
    )


def _ok_envelope() -> tuple[int, dict[str, Any]]:
    return (200, {"ok": True})


class TestStreamingFanOut:
    def test_two_streams_with_correct_extra_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        left_stream = _sse([_delta_chunk("Q1 "), _final_usage_chunk(20, 5)])
        right_stream = _sse([_delta_chunk("P1 "), _final_usage_chunk(20, 5)])
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [(200, left_stream), (200, right_stream)],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)
        body = {
            "model": "gemma-4-31b-reasoning--qr-vs-prng",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 64,
        }

        asyncio.run(_collect(pipe.pipe(body, __user__={"email": "user@example.com"})))

        # Two vLLM calls were issued. Check both bodies have stream=True and
        # the right per-side extra_body. The Pipe also forwards seed/temp/etc.
        vllm_calls = [c for c in handler.calls if c.url.path == "/v1/chat/completions"]
        assert len(vllm_calls) == 2
        bodies = [json.loads(c.content) for c in vllm_calls]

        assert all(b["stream"] is True for b in bodies)
        assert all(b["model"] == "gemma-4-31b-reasoning" for b in bodies)
        assert all(b["seed"] == 42 for b in bodies)
        assert all(b["temperature"] == 0.7 for b in bodies)
        assert all(b["top_p"] == 0.95 for b in bodies)
        assert all(b["top_k"] == 50 for b in bodies)
        assert all(b["max_tokens"] == 64 for b in bodies)

        sources = {b["extra_body"]["qr_entropy_source_type"] for b in bodies}
        assert sources == {"quantum_grpc", "system"}

        # `model` is rewritten to the base model — pseudo-suffix stripped.
        # `metadata` is NOT forwarded to vLLM (it's OWUI-internal).
        for b in bodies:
            assert "metadata" not in b
            assert not b["model"].endswith("--qr-vs-prng")

    def test_qwen_pseudo_routes_to_qwen_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([_delta_chunk("a"), _final_usage_chunk(1, 1)])),
                    (200, _sse([_delta_chunk("b"), _final_usage_chunk(1, 1)])),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)
        body = {
            "model": "qwen-3.6-27b-reasoning--qr-vs-prng",
            "messages": [{"role": "user", "content": "hi"}],
        }

        asyncio.run(_collect(pipe.pipe(body, __user__={"email": "u@e"})))

        vllm_bodies = [
            json.loads(c.content) for c in handler.calls if c.url.path == "/v1/chat/completions"
        ]
        assert all(b["model"] == "qwen-3.6-27b-reasoning" for b in vllm_bodies)

    def test_intermediate_yields_have_table_and_buffer_contents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (
                        200,
                        _sse(
                            [
                                _delta_chunk("Quantum bit "),
                                _delta_chunk("two."),
                                _final_usage_chunk(10, 4),
                            ]
                        ),
                    ),
                    (
                        200,
                        _sse(
                            [
                                _delta_chunk("Pseudo bit "),
                                _delta_chunk("two."),
                                _final_usage_chunk(10, 4),
                            ]
                        ),
                    ),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        assert chunks, "Pipe yielded nothing"
        # Every yield carries the dual-column table header.
        for c in chunks:
            assert "| Quantum | Pseudo-random |" in c
            assert "Comparison mode is on" in c
        # The final yield carries the cumulative content from both sides.
        final = chunks[-1]
        assert "Quantum bit two." in final
        assert "Pseudo bit two." in final

    def test_final_yield_includes_summed_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([_delta_chunk("Q"), _final_usage_chunk(50, 100)])),
                    (200, _sse([_delta_chunk("P"), _final_usage_chunk(50, 100)])),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        # Summed usage: 100 prompt + 200 completion. Footer rendered on final yield.
        final = chunks[-1]
        assert "100 prompt" in final
        assert "200 completion" in final
        # Multiplication-sign in the usage footer is intentional per spec.
        assert "2\u00d7" in final
        # Earlier yields do NOT render the usage footer (it's pinned to the final tick).
        for earlier in chunks[:-1]:
            assert "100 prompt" not in earlier


# ---------------------------------------------------------------------------
# Debit / upsert after streams close
# ---------------------------------------------------------------------------


class TestPostStreamMetering:
    def test_one_comparison_flagged_debit_after_streams(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([_delta_chunk("Q"), _final_usage_chunk(40, 80)])),
                    (200, _sse([_delta_chunk("P"), _final_usage_chunk(40, 80)])),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        debit_calls = [c for c in handler.calls if c.url.path == "/api/allowance/debit"]
        assert len(debit_calls) == 1, "exactly one debit call after both streams close"
        debit_body = json.loads(debit_calls[0].content)
        assert debit_body["accountEmail"] == "u@e"
        assert debit_body["promptTokens"] == 80  # 40 + 40 summed
        assert debit_body["completionTokens"] == 160  # 80 + 80 summed
        assert debit_body["comparisonMode"] is True
        assert debit_body["conversationId"] is None  # no chat_id in metadata

    def test_upsert_when_chat_id_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([_delta_chunk("Q"), _final_usage_chunk(40, 80)])),
                    (200, _sse([_delta_chunk("P"), _final_usage_chunk(40, 80)])),
                ],
                "/api/allowance/debit": [_ok_envelope()],
                "/api/conversations/upsert": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                        "metadata": {"chat_id": "chat-7", "chat_title": "Comparison run"},
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        upsert_calls = [c for c in handler.calls if c.url.path == "/api/conversations/upsert"]
        assert len(upsert_calls) == 1
        upsert_body = json.loads(upsert_calls[0].content)
        assert upsert_body["owuiChatId"] == "chat-7"
        assert upsert_body["title"] == "Comparison run"
        assert upsert_body["comparisonModeUsed"] is True
        # weighted = (40+40) prompt + 3*(80+80) completion = 80 + 480 = 560
        assert upsert_body["weightedTokensTotal"] == 560

    def test_skips_metering_when_no_tokens_produced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Streams that produce nothing (e.g. vLLM error path that still 200s).
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([])),
                    (200, _sse([])),
                ],
                # No debit/upsert scripted — they MUST NOT fire.
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        assert not any(c.url.path == "/api/allowance/debit" for c in handler.calls)


# ---------------------------------------------------------------------------
# Preflight short-circuit
# ---------------------------------------------------------------------------


class TestPreflightInsufficient:
    def test_does_not_invoke_vllm(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
                ],
                # No vLLM scripted — Pipe must not call it.
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        # Exactly one chunk: the rejection markdown with refill + waitlist.
        assert len(chunks) == 1
        msg = chunks[0]
        assert "Daily allowance used up" in msg
        assert "2026-05-17 00:00" in msg
        assert "entropic.science/account/waitlist?from=allowance" in msg

        # vLLM was never called.
        assert not any(c.url.path == "/v1/chat/completions" for c in handler.calls)

    def test_preflight_uses_comparison_mode_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    (200, _sse([_delta_chunk("Q"), _final_usage_chunk(1, 1)])),
                    (200, _sse([_delta_chunk("P"), _final_usage_chunk(1, 1)])),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        preflight_call = next(c for c in handler.calls if c.url.path == "/api/allowance/preflight")
        body = json.loads(preflight_call.content)
        assert body["comparisonMode"] is True


# ---------------------------------------------------------------------------
# One-side error tolerance
# ---------------------------------------------------------------------------


class TestOneSideErrorTolerated:
    def test_failed_side_tagged_other_side_completes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler(
            {
                "/api/allowance/preflight": [_ok_preflight()],
                "/v1/chat/completions": [
                    # Left (quantum) side: 500 error.
                    (500, {"error": "quantum stream blew up"}),
                    # Right (pseudo) side: produces tokens normally.
                    (
                        200,
                        _sse(
                            [
                                _delta_chunk("Pseudo "),
                                _delta_chunk("output."),
                                _final_usage_chunk(20, 10),
                            ]
                        ),
                    ),
                ],
                "/api/allowance/debit": [_ok_envelope()],
            }
        )
        pipe = _patched_pipe(monkeypatch, handler)

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        final = chunks[-1]
        # Failed column carries an inline error tag.
        assert "[stream error" in final
        # Successful column rendered normally.
        assert "Pseudo output." in final

        # Debit fires for tokens actually produced (only the right side here).
        debit_calls = [c for c in handler.calls if c.url.path == "/api/allowance/debit"]
        assert len(debit_calls) == 1
        debit_body = json.loads(debit_calls[0].content)
        assert debit_body["promptTokens"] == 20
        assert debit_body["completionTokens"] == 10
        assert debit_body["comparisonMode"] is True


# ---------------------------------------------------------------------------
# Sign-in-required + missing-secret guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_missing_email_yields_signin_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})  # no scripted calls — none should fire
        pipe = _patched_pipe(monkeypatch, handler)

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__=None,
                )
            )
        )

        assert len(chunks) == 1
        assert "Sign-in required" in chunks[0]
        assert handler.calls == []

    def test_missing_secret_yields_offline_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler = _RecordingHandler({})
        pipe = _patched_pipe(monkeypatch, handler)
        pipe.valves.service_token_secret = ""

        chunks = asyncio.run(
            _collect(
                pipe.pipe(
                    {
                        "model": "gemma-4-31b-reasoning--qr-vs-prng",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    __user__={"email": "u@e"},
                )
            )
        )

        assert len(chunks) == 1
        assert "Service offline" in chunks[0]
        assert handler.calls == []

    def test_pseudo_suffix_strip_handles_unsuffixed_id(self) -> None:
        assert pipe_mod._strip_pseudo_suffix("foo--qr-vs-prng") == "foo"
        assert pipe_mod._strip_pseudo_suffix("plain-model") == "plain-model"
        assert pipe_mod._strip_pseudo_suffix("") == ""


# ---------------------------------------------------------------------------
# Markdown table escaping
# ---------------------------------------------------------------------------


class TestTableCellEscaping:
    def test_pipe_chars_escaped(self) -> None:
        rendered = pipe_mod._render_dual_column_markdown("a | b", "c | d")
        # Each `|` inside a buffer becomes `\|` so it doesn't break the table.
        assert "a \\| b" in rendered
        assert "c \\| d" in rendered

    def test_newlines_become_br(self) -> None:
        rendered = pipe_mod._render_dual_column_markdown("line1\nline2", "")
        assert "line1<br>line2" in rendered

    def test_empty_buffers_render_blank_cells(self) -> None:
        rendered = pipe_mod._render_dual_column_markdown("", "")
        assert "|  |  |" in rendered

    def test_usage_footer_appended_when_present(self) -> None:
        rendered = pipe_mod._render_dual_column_markdown(
            "Q",
            "P",
            usage_footer=(
                "_Tokens: 5 prompt + 7 completion (billed at 2\u00d7 under comparison mode)._"
            ),
        )
        assert "Tokens: 5 prompt + 7 completion" in rendered
        # Footer comes after the table.
        table_idx = rendered.index("| Q | P |")
        footer_idx = rendered.index("Tokens:")
        assert footer_idx > table_idx
