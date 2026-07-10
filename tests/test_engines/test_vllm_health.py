"""Tests for the passive ``/health/entropy`` middleware (qr-server profile).

The middleware restores the health route a bare ``vllm serve`` lacks, answering
exclusively from the cross-process status files (file IPC — never a live gRPC
probe). Payload contract is consumed by three OWUI-side readers: the setup
guard (``rpc_ok``/``tcp_ok``/``summary``), the ``/api/qr-status`` chip
(boolean ``rpc_ok`` authoritative), and the comparison Pipe's no-silent-PRNG
banner (``fallback_count`` + ``sampler.currently_degraded``/``age_s``).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from qr_sampler.engines.vllm.health import (
    HEALTH_PATH,
    STALE_AFTER_S,
    build_entropy_health_payload,
    entropy_health_middleware,
)
from qr_sampler.entropy.fallback import FallbackEntropySource


class _FixedBytesSource:
    """Minimal always-healthy EntropySource stand-in."""

    def __init__(self, name: str = "system") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_random_bytes(self, n: int) -> bytes:
        return b"\xbb" * n

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def warmup(self) -> None:  # pragma: no cover - trivial
        pass


@pytest.fixture()
def status_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    path = tmp_path / "qr_entropy_status.json"
    monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(path))
    # Point the perf channel at the same tmpdir so a developer-machine
    # leftover perf file can never leak into assertions.
    monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "perf.json"))
    return path


def _write_status(path: Any, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestBuildPayload:
    def test_no_status_file_is_unknown_not_error(self, status_path: Any) -> None:
        payload = build_entropy_health_payload()
        assert payload["rpc_ok"] is None
        assert payload["tcp_ok"] is False
        assert payload["fallback_count"] is None
        assert payload["sampler"] is None
        assert "no entropy status yet" in payload["summary"]

    def test_healthy_snapshot(self, status_path: Any) -> None:
        _write_status(
            status_path,
            {
                "primary_name": "quantum_grpc",
                "fallback_name": "system",
                "last_source_used": "quantum_grpc",
                "fallback_count": 3,
                "currently_degraded": False,
                "updated_at": 1_000.0,
            },
        )
        payload = build_entropy_health_payload(now=1_005.0)
        assert payload["rpc_ok"] is True
        assert payload["tcp_ok"] is True
        assert payload["fallback_count"] == 3
        assert payload["last_source_used"] == "quantum_grpc"
        assert payload["primary_name"] == "quantum_grpc"
        assert payload["sampler"] == {"currently_degraded": False, "age_s": 5.0}
        assert "ok" in payload["summary"]
        assert "gate" not in payload  # no gate keys published

    def test_degraded_snapshot(self, status_path: Any) -> None:
        _write_status(
            status_path,
            {
                "primary_name": "quantum_grpc",
                "fallback_name": "system",
                "last_source_used": "system",
                "fallback_count": 42,
                "currently_degraded": True,
                "updated_at": 2_000.0,
            },
        )
        payload = build_entropy_health_payload(now=2_001.0)
        assert payload["rpc_ok"] is False
        assert payload["sampler"]["currently_degraded"] is True
        assert payload["sampler"]["age_s"] == 1.0
        assert "DEGRADED" in payload["summary"]
        assert "'system'" in payload["summary"]

    def test_stale_snapshot_flagged_in_summary(self, status_path: Any) -> None:
        _write_status(
            status_path,
            {
                "currently_degraded": False,
                "fallback_count": 0,
                "updated_at": 0.0,
            },
        )
        payload = build_entropy_health_payload(now=STALE_AFTER_S + 100.0)
        assert payload["rpc_ok"] is True  # staleness is surfaced, not conflated
        assert "old" in payload["summary"]

    def test_gate_block_included_when_published(self, status_path: Any) -> None:
        _write_status(
            status_path,
            {
                "currently_degraded": False,
                "updated_at": 10.0,
                "gate_open": True,
                "gate_boost": 0.25,
                "coherence_valid": True,
            },
        )
        payload = build_entropy_health_payload(now=11.0)
        assert payload["gate"] == {
            "gate_open": True,
            "gate_boost": 0.25,
            "coherence_valid": True,
        }


class _FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeRequest:
    def __init__(self, path: str, method: str = "GET") -> None:
        self.url = _FakeURL(path)
        self.method = method


class TestMiddleware:
    def test_health_path_answers_json_200(self, status_path: Any) -> None:
        _write_status(
            status_path,
            {"currently_degraded": False, "fallback_count": 0, "updated_at": 5.0},
        )

        async def _boom(_request: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("call_next must not be reached for the health path")

        response = asyncio.run(entropy_health_middleware(_FakeRequest(HEALTH_PATH), _boom))
        assert response.status_code == 200
        body = json.loads(bytes(response.body))
        assert body["rpc_ok"] is True
        assert body["tcp_ok"] is True

    def test_other_paths_pass_through(self, status_path: Any) -> None:
        sentinel = object()
        seen: list[Any] = []

        async def _next(request: Any) -> Any:
            seen.append(request.url.path)
            return sentinel

        result = asyncio.run(
            entropy_health_middleware(_FakeRequest("/v1/chat/completions", "POST"), _next)
        )
        assert result is sentinel
        assert seen == ["/v1/chat/completions"]

    def test_non_get_on_health_path_passes_through(self, status_path: Any) -> None:
        sentinel = object()

        async def _next(_request: Any) -> Any:
            return sentinel

        result = asyncio.run(entropy_health_middleware(_FakeRequest(HEALTH_PATH, "POST"), _next))
        assert result is sentinel


class TestSharedFileCrossTalk:
    """Regression pin for the qr-server shared-status-file design.

    A per-request ``qr_entropy_source_type=system`` draw runs on the
    pre-init **system-primary** pipeline, whose wrapper never receives
    ``enable_status_publishing()`` (only the default/quantum pipeline owns
    the file — ``VLLMAdapter.__init__``). A deliberate PRNG-lane request
    must therefore never flip the shared file to degraded nor inflate its
    ``fallback_count`` — confirmed live on qr-server (2026-07-10) and pinned
    here at the wrapper level.
    """

    def test_system_lane_draws_do_not_clobber_quantum_status(self, status_path: Any) -> None:
        quantum = FallbackEntropySource(
            _FixedBytesSource("quantum_grpc"), _FixedBytesSource("system")
        )
        quantum.enable_status_publishing()

        system_lane = FallbackEntropySource(
            _FixedBytesSource("system"), _FixedBytesSource("system")
        )
        system_lane.get_random_bytes(64)  # deliberate PRNG-lane draw

        payload = build_entropy_health_payload()
        assert payload["rpc_ok"] is True
        assert payload["primary_name"] == "quantum_grpc"
        assert payload["fallback_count"] == 0
        assert payload["sampler"]["currently_degraded"] is False
