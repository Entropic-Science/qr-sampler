"""Unit tests for the TcpPreprobe state machine."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from qr_sampler.entropy.qgrpc.preprobe import TcpPreprobe
from qr_sampler.exceptions import EntropyUnavailableError


class TestTcpPreprobe:
    def test_disabled_probe_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QR_GRPC_PREPROBE_ENABLED=0 disables the probe entirely."""
        monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "0")
        probe = TcpPreprobe("localhost:1")  # nothing listens on port 1
        probe.check()  # must not raise, must not touch the socket
        assert probe.backoff_active() is False

    def test_malformed_address_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "1")
        probe = TcpPreprobe("no-port-here")
        with pytest.raises(EntropyUnavailableError, match="Malformed"):
            probe.check()

    def test_failure_engages_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A connect failure stamps the backoff; the next check short-circuits."""
        monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "1")
        probe = TcpPreprobe("localhost:50051")

        connects: list[None] = []

        def failing_connect(*args: Any, **kwargs: Any) -> Any:
            connects.append(None)
            raise OSError("refused")

        monkeypatch.setattr("socket.create_connection", failing_connect)
        with pytest.raises(EntropyUnavailableError, match="unreachable"):
            probe.check()
        assert probe.backoff_active() is True
        # Within the backoff window: raises WITHOUT re-touching the socket.
        with pytest.raises(EntropyUnavailableError, match="backoff in effect"):
            probe.check()
        assert len(connects) == 1

    def test_healthy_window_suppresses_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A recent fetch success skips the probe entirely."""
        monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "1")
        probe = TcpPreprobe("localhost:50051")

        connects: list[None] = []

        def counting_connect(*args: Any, **kwargs: Any) -> Any:
            connects.append(None)
            return MagicMock()

        monkeypatch.setattr("socket.create_connection", counting_connect)
        probe.check()  # no success yet: probe runs
        assert len(connects) == 1
        probe.note_fetch_success()
        probe.check()  # healthy: suppressed
        probe.check()
        assert len(connects) == 1
