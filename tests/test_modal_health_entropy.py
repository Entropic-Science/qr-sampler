"""Tests for the /health/entropy middleware (iter-53 cross-process rewrite).

The middleware imports fastapi, which is not a qr-sampler dependency —
it ships in the vLLM production image and in the qr-llm-chat dev venv.
Skip cleanly where it's absent.
"""

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")

from qr_sampler.connectors.modal import health_entropy_middleware as mw  # noqa: E402
from qr_sampler.entropy.status_file import write_entropy_status  # noqa: E402


@pytest.fixture(autouse=True)
def clean_module_state(tmp_path, monkeypatch):
    """Isolate the status file + module globals per test."""
    monkeypatch.setenv(
        "QR_ENTROPY_STATUS_FILE", str(tmp_path / "qr_entropy_status.json")
    )
    mw.set_fallback_source(None)
    mw.reset_probe_state_for_tests()
    yield
    mw.set_fallback_source(None)
    mw.reset_probe_state_for_tests()


class TestCombineRpcOk:
    def test_probe_failure_is_definitive(self) -> None:
        sampler = {"currently_degraded": False, "age_s": 0.0}
        assert mw._combine_rpc_ok(False, sampler) is False

    def test_fresh_degraded_overrides_probe_success(self) -> None:
        sampler = {"currently_degraded": True, "age_s": 1.0}
        assert mw._combine_rpc_ok(True, sampler) is False

    def test_stale_degraded_defers_to_probe(self) -> None:
        sampler = {
            "currently_degraded": True,
            "age_s": mw._SAMPLER_DEGRADED_FRESH_S + 1.0,
        }
        assert mw._combine_rpc_ok(True, sampler) is True

    def test_probe_success_clean_sampler(self) -> None:
        sampler = {"currently_degraded": False, "age_s": 5.0}
        assert mw._combine_rpc_ok(True, sampler) is True

    def test_no_probe_falls_back_to_sampler_flag(self) -> None:
        assert mw._combine_rpc_ok(None, {"currently_degraded": True, "age_s": 999.0}) is False
        assert mw._combine_rpc_ok(None, {"currently_degraded": False, "age_s": 999.0}) is True

    def test_nothing_known(self) -> None:
        assert mw._combine_rpc_ok(None, None) is None

    def test_probe_ok_no_sampler(self) -> None:
        assert mw._combine_rpc_ok(True, None) is True


class TestResolveSamplerState:
    def test_none_when_no_channel(self) -> None:
        state, source = mw._resolve_sampler_state()
        assert state is None
        assert source == "none"

    def test_reads_status_file(self) -> None:
        write_entropy_status(
            {
                "primary_name": "quantum_grpc",
                "fallback_name": "system",
                "last_source_used": "system",
                "fallback_count": 7,
                "currently_degraded": True,
            }
        )
        state, source = mw._resolve_sampler_state()
        assert source == "status_file"
        assert state is not None
        assert state["primary_name"] == "quantum_grpc"
        assert state["fallback_count"] == 7
        assert state["currently_degraded"] is True
        assert state["age_s"] < 5.0

    def test_in_process_source_wins_over_file(self) -> None:
        write_entropy_status({"primary_name": "from_file"})

        class _Src:
            primary_name = "quantum_grpc"
            last_source_used = "quantum_grpc"
            fallback_count = 0
            currently_degraded = False

        mw.set_fallback_source(_Src())
        state, source = mw._resolve_sampler_state()
        assert source == "in_process"
        assert state is not None
        assert state["primary_name"] == "quantum_grpc"
        assert state["age_s"] == 0.0

    def test_garbage_updated_at_treated_as_stale(self, tmp_path) -> None:
        # Bypass write_entropy_status (it restamps updated_at) and plant
        # a raw record with a non-numeric timestamp: the state must parse
        # but read as infinitely stale, so a passing live probe wins.
        import json

        (tmp_path / "qr_entropy_status.json").write_text(
            json.dumps({"currently_degraded": True, "updated_at": "bogus"}),
            encoding="utf-8",
        )
        state, source = mw._resolve_sampler_state()
        assert source == "status_file"
        assert state is not None
        assert state["age_s"] == float("inf")
        assert mw._combine_rpc_ok(True, state) is True


class TestLiveProbe:
    def test_disabled_for_non_quantum_primary(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "system")
        result = mw._live_probe_sync()
        assert result["ok"] is None
        assert "probe n/a" in result["error"]

    def test_unreachable_endpoint_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        # Reserved port 1 on loopback: refused immediately by the TCP
        # pre-probe, no multi-second gRPC hang.
        monkeypatch.setenv("QR_GRPC_SERVER_ADDRESS", "127.0.0.1:1")
        t0 = time.perf_counter()
        result = mw._live_probe_sync()
        elapsed = time.perf_counter() - t0
        assert result["ok"] is False
        assert result["error"]
        assert elapsed < 3.0

    def test_verdict_is_cached(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "quantum_grpc")
        monkeypatch.setenv("QR_GRPC_SERVER_ADDRESS", "127.0.0.1:1")
        first = mw._live_probe_sync()
        # A second call inside the TTL must not re-touch the socket; we
        # prove it by breaking the address — a cache miss would now
        # produce a different error string.
        monkeypatch.setenv("QR_GRPC_SERVER_ADDRESS", "garbage")
        second = mw._live_probe_sync()
        assert second == first


class TestEndpoint:
    @pytest.fixture()
    def client(self, monkeypatch):
        httpx = pytest.importorskip("httpx")  # noqa: F841 — TestClient dep
        from fastapi.testclient import TestClient

        app = fastapi.FastAPI()
        app.middleware("http")(mw.health_entropy_middleware)

        @app.get("/passthrough")
        def passthrough() -> dict:
            return {"hello": "world"}

        return TestClient(app)

    def test_passthrough_untouched(self, client) -> None:
        response = client.get("/passthrough")
        assert response.status_code == 200
        assert response.json() == {"hello": "world"}

    def test_503_when_nothing_known(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {"ok": None, "latency_ms": None, "error": "probe n/a"},
        )
        response = client.get("/health/entropy")
        assert response.status_code == 503
        body = response.json()
        assert body["rpc_ok"] is None
        assert body["error"] == "not_initialised"

    def test_200_healthy(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {"ok": True, "tcp_ok": True, "latency_ms": 42.0, "error": None},
        )
        write_entropy_status(
            {
                "primary_name": "quantum_grpc",
                "last_source_used": "quantum_grpc",
                "fallback_count": 0,
                "currently_degraded": False,
            }
        )
        response = client.get("/health/entropy")
        assert response.status_code == 200
        body = response.json()
        assert body["rpc_ok"] is True
        assert body["tcp_ok"] is True
        assert "quantum entropy OK" in body["summary"]
        assert body["fallback_count"] == 0
        assert body["primary_name"] == "quantum_grpc"
        assert body["probe"]["ok"] is True
        assert body["sampler_source"] == "status_file"

    def test_perf_block_included_when_present(
        self, client, monkeypatch, tmp_path
    ) -> None:
        """iter-55: the adapter's perf aggregate rides along when published."""
        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "perf.json"))
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {"ok": True, "tcp_ok": True, "latency_ms": 42.0, "error": None},
        )
        write_entropy_status(
            {
                "primary_name": "quantum_grpc",
                "last_source_used": "quantum_grpc",
                "fallback_count": 0,
                "currently_degraded": False,
            }
        )
        from qr_sampler.entropy.status_file import write_perf_status

        write_perf_status(
            {
                "window_tokens": 10,
                "stage_ms": {"total": {"avg": 12.0, "p95": 20.0}},
                "prefetch": {"hit_ratio": 1.0, "echo_verified_ratio": 0.0},
            }
        )
        body = client.get("/health/entropy").json()
        assert body["perf"]["window_tokens"] == 10
        assert body["perf"]["stage_ms"]["total"]["avg"] == 12.0
        assert isinstance(body["perf"]["age_s"], (int, float))

    def test_perf_block_null_when_absent(self, client, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "missing.json"))
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {"ok": True, "tcp_ok": True, "latency_ms": 42.0, "error": None},
        )
        write_entropy_status(
            {
                "primary_name": "quantum_grpc",
                "last_source_used": "quantum_grpc",
                "fallback_count": 0,
                "currently_degraded": False,
            }
        )
        body = client.get("/health/entropy").json()
        assert body["perf"] is None

    def test_200_degraded_by_probe(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {
                "ok": False,
                "tcp_ok": False,
                "latency_ms": 500.0,
                "error": "unreachable",
            },
        )
        response = client.get("/health/entropy")
        assert response.status_code == 200
        body = response.json()
        assert body["rpc_ok"] is False
        assert body["tcp_ok"] is False
        assert "QRNG unreachable" in body["summary"]
        assert body["probe"]["error"] == "unreachable"

    def test_200_degraded_by_fresh_sampler_state(self, client, monkeypatch) -> None:
        monkeypatch.setattr(
            mw,
            "_live_probe_sync",
            lambda: {"ok": True, "tcp_ok": True, "latency_ms": 42.0, "error": None},
        )
        write_entropy_status(
            {
                "primary_name": "quantum_grpc",
                "last_source_used": "system",
                "fallback_count": 9,
                "currently_degraded": True,
            }
        )
        response = client.get("/health/entropy")
        assert response.status_code == 200
        body = response.json()
        assert body["rpc_ok"] is False
        assert "PRNG fallback (count=9)" in body["summary"]
        assert body["fallback_count"] == 9
        assert body["last_source_used"] == "system"
