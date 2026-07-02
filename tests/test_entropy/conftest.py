"""Shared fixtures for the entropy test package."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_grpc_preprobe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the live ``socket.create_connection`` pre-probe.

    Lazy init defers gRPC channel creation to first fetch. The pre-probe
    still runs ahead of that fetch and would call
    ``socket.create_connection(("localhost", 50051), ...)`` against
    whatever the test host has bound there. Disabling it keeps the qgrpc
    tests pure unit-level against the mocked ``grpc.aio.insecure_channel``
    (individual tests re-enable it via monkeypatch when the probe itself
    is under test).
    """
    monkeypatch.setenv("QR_GRPC_PREPROBE_ENABLED", "0")
