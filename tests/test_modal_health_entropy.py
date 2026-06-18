"""Tests for the passive /health/entropy middleware.

The endpoint reports last-known entropy health from the cross-process
status file (or an in-process FallbackEntropySource) — it never opens a
gRPC channel or probes the QRNG. The middleware imports fastapi, which
ships in the vLLM production image and the qr-llm-chat dev venv but is
not a qr-sampler dependency; skip cleanly where it's absent.
"""

from __future__ import annotations

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
    yield
    mw.set_fallback_source(None)


class TestRpcOkFromSampler:
    def test_healthy_when_not_degraded(self) -> None:
        assert mw._rpc_ok_from_sampler({"currently_degraded": False}) is True

    def test_degraded_is_false(self) -> None:
        assert mw._rpc_ok_from_sampler({"currently_degraded": True}) is False

    def test_missing_flag_reads_healthy(self) -> None:
        assert mw._rpc_ok_from_sampler({}) is True

    def test_no_sampler_is_unknown(self) -> None:
        assert mw._rpc_ok_from_sampler(None) is None


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
        # A non-numeric timestamp must parse but read as infinitely stale;
        # the degraded flag still drives the verdict.
        import json

        (tmp_path / "qr_entropy_status.json").write_text(
            json.dumps({"currently_degraded": True, "updated_at": "bogus"}),
            encoding="utf-8",
        )
        state, source = mw._resolve_sampler_state()
        assert source == "status_file"
        assert state is not None
        assert state["age_s"] == float("inf")
        assert mw._rpc_ok_from_sampler(state) is False


class TestEndpoint:
    @pytest.fixture()
    def client(self):
        pytest.importorskip("httpx")  # TestClient dependency
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

    def test_503_when_nothing_known(self, client) -> None:
        response = client.get("/health/entropy")
        assert response.status_code == 503
        body = response.json()
        assert body["rpc_ok"] is None
        assert body["error"] == "not_initialised"

    def test_200_healthy(self, client) -> None:
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
        assert body["tcp_ok"] is None
        assert "quantum entropy OK" in body["summary"]
        assert body["fallback_count"] == 0
        assert body["primary_name"] == "quantum_grpc"
        assert body["sampler_source"] == "status_file"

    def test_200_degraded(self, client) -> None:
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

    def test_perf_block_included_when_present(self, client, monkeypatch, tmp_path) -> None:
        """iter-55: the adapter's perf aggregate rides along when published."""
        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "perf.json"))
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
