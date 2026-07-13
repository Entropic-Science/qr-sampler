"""Tests for FallbackEntropySource."""

from __future__ import annotations

import pytest

from qr_sampler.entropy import fallback as fallback_module
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.telemetry import status_file


class _AlwaysFailSource(EntropySource):
    """Test double: always raises EntropyUnavailableError."""

    @property
    def name(self) -> str:
        return "always_fail"

    @property
    def is_available(self) -> bool:
        return False

    def get_random_bytes(self, n: int) -> bytes:
        raise EntropyUnavailableError("always fails")

    def close(self) -> None:
        pass


class _RuntimeErrorSource(EntropySource):
    """Test double: always raises RuntimeError (not EntropyUnavailableError)."""

    @property
    def name(self) -> str:
        return "runtime_error"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        raise RuntimeError("unexpected error")

    def close(self) -> None:
        pass


class _FixedBytesSource(EntropySource):
    """Test double: returns a fixed byte pattern."""

    def __init__(self, pattern: int = 0xAA) -> None:
        self._pattern = pattern
        self.call_count = 0

    @property
    def name(self) -> str:
        return f"fixed_{self._pattern:#04x}"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        self.call_count += 1
        return bytes([self._pattern] * n)

    def close(self) -> None:
        pass


class TestFallbackEntropySource:
    """Tests for the composition fallback wrapper."""

    def test_delegates_to_primary(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        data = source.get_random_bytes(4)
        assert data == bytes([0xAA] * 4)
        assert primary.call_count == 1
        assert fallback.call_count == 0

    def test_falls_back_on_entropy_unavailable(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        data = source.get_random_bytes(4)
        assert data == bytes([0xBB] * 4)
        assert fallback.call_count == 1

    def test_last_source_used_tracks_primary(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        source.get_random_bytes(4)
        assert source.last_source_used == primary.name

    def test_last_source_used_tracks_fallback(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        source.get_random_bytes(4)
        assert source.last_source_used == fallback.name

    def test_does_not_catch_non_entropy_errors(self) -> None:
        """RuntimeError should propagate, not trigger fallback."""
        primary = _RuntimeErrorSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        with pytest.raises(RuntimeError, match="unexpected error"):
            source.get_random_bytes(4)
        assert fallback.call_count == 0

    def test_raises_when_both_fail(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)

        with pytest.raises(EntropyUnavailableError):
            source.get_random_bytes(4)

    def test_name_is_compound(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        assert source.name == f"{primary.name}+{fallback.name}"

    def test_is_available_when_primary_available(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is True

    def test_is_available_when_only_fallback_available(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is True

    def test_is_unavailable_when_both_unavailable(self) -> None:
        primary = _AlwaysFailSource()
        fallback = _AlwaysFailSource()
        source = FallbackEntropySource(primary, fallback)
        assert source.is_available is False

    def test_close_closes_both(self) -> None:
        closed: list[str] = []

        class _TrackClose(EntropySource):
            def __init__(self, id: str) -> None:
                self._id = id

            @property
            def name(self) -> str:
                return self._id

            @property
            def is_available(self) -> bool:
                return True

            def get_random_bytes(self, n: int) -> bytes:
                return b"\x00" * n

            def close(self) -> None:
                closed.append(self._id)

        primary = _TrackClose("p")
        fallback = _TrackClose("f")
        source = FallbackEntropySource(primary, fallback)
        source.close()
        assert "p" in closed
        assert "f" in closed

    def test_health_check(self) -> None:
        primary = _FixedBytesSource(0xAA)
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)

        health = source.health_check()
        assert health["source"] == source.name
        assert health["healthy"] is True
        assert "primary" in health
        assert "fallback" in health
        assert "last_source_used" in health


class _FlakySource(EntropySource):
    """Test double: fails when ``should_fail`` is set, else fixed bytes."""

    def __init__(self) -> None:
        self.should_fail = False

    @property
    def name(self) -> str:
        return "flaky"

    @property
    def is_available(self) -> bool:
        return not self.should_fail

    def get_random_bytes(self, n: int) -> bytes:
        if self.should_fail:
            raise EntropyUnavailableError("flaking")
        return b"\xaa" * n

    def close(self) -> None:
        pass


class TestStatusPublishing:
    """iter-53: cross-process status-file writes (see telemetry/status_file.py)."""

    @pytest.fixture()
    def status_path(self, tmp_path, monkeypatch):
        path = tmp_path / "qr_entropy_status.json"
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(path))
        return path

    def test_no_writes_unless_enabled(self, status_path) -> None:
        source = FallbackEntropySource(_AlwaysFailSource(), _FixedBytesSource(0xBB))
        source.get_random_bytes(4)
        assert not status_path.exists()

    def test_enable_writes_initial_state(self, status_path) -> None:
        primary = _FixedBytesSource(0xAA)
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["primary_name"] == primary.name
        assert data["last_source_used"] == primary.name
        assert data["fallback_count"] == 0
        assert data["currently_degraded"] is False

    def test_fallback_writes_degraded_state(self, status_path) -> None:
        primary = _AlwaysFailSource()
        fallback = _FixedBytesSource(0xBB)
        source = FallbackEntropySource(primary, fallback)
        source.enable_status_publishing()

        source.get_random_bytes(4)
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["currently_degraded"] is True
        assert data["fallback_count"] == 1
        assert data["last_source_used"] == fallback.name

    def test_recovery_writes_all_clear(self, status_path) -> None:
        primary = _FlakySource()
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()

        primary.should_fail = True
        source.get_random_bytes(4)
        primary.should_fail = False
        source.get_random_bytes(4)

        data = status_file.read_entropy_status()
        assert data is not None
        assert data["currently_degraded"] is False
        assert data["fallback_count"] == 1
        assert data["last_source_used"] == primary.name

    def test_draw_failure_writes_degraded_state(self, status_path) -> None:
        # 2026-07 blind-spot fix: a failed server-integrated draw must
        # publish degraded state to the status file /health/entropy reads,
        # even though the primary's RAW bytes still work (get_draw raises
        # via the base default; get_random_bytes succeeds).
        primary = _FixedBytesSource(0xAA)
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()

        with pytest.raises(EntropyUnavailableError):
            source.get_draw(1024, "qrng-a")

        data = status_file.read_entropy_status()
        assert data is not None
        assert data["currently_degraded"] is True
        assert data["fallback_count"] == 1

    def test_raw_success_after_draw_failure_keeps_status_degraded(self, status_path) -> None:
        # Regression for the exact multi-day-invisible-degrade bug: the raw
        # byte fetch that serves the pipeline's PRNG token succeeds on the
        # same primary and must NOT flip the status file back to healthy
        # while draws are still failing.
        primary = _FixedBytesSource(0xAA)
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()

        with pytest.raises(EntropyUnavailableError):
            source.get_draw(1024, "qrng-a")
        source.get_random_bytes(4)  # pipeline's PRNG-token byte fetch (succeeds)

        data = status_file.read_entropy_status()
        assert data is not None
        assert data["currently_degraded"] is True

    def test_mid_outage_refresh_is_throttled(self, status_path) -> None:
        primary = _AlwaysFailSource()
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()

        # First fallback is a transition (forced write); the second lands
        # inside the refresh window and must NOT rewrite the file.
        source.get_random_bytes(4)
        source.get_random_bytes(4)
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["fallback_count"] == 1
        assert source.fallback_count == 2

    def test_mid_outage_refresh_after_interval(self, status_path, monkeypatch) -> None:
        monkeypatch.setattr(fallback_module, "_STATUS_REFRESH_MIN_INTERVAL_S", 0.0)
        primary = _AlwaysFailSource()
        source = FallbackEntropySource(primary, _FixedBytesSource(0xBB))
        source.enable_status_publishing()

        source.get_random_bytes(4)
        source.get_random_bytes(4)
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["fallback_count"] == 2
