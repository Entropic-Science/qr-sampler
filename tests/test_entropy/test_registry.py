"""Tests for EntropySourceRegistry."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry


class _DummySource(EntropySource):
    """Minimal concrete source for registry tests."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return b"\x00" * n

    def close(self) -> None:
        pass


class TestEntropySourceRegistry:
    """Tests for the decorator-based registry with entry-point discovery."""

    def setup_method(self) -> None:
        """Save and clear registry state before each test."""
        self._saved_registry = dict(EntropySourceRegistry._registry)
        self._saved_loaded = EntropySourceRegistry._entry_points_loaded

    def teardown_method(self) -> None:
        """Restore registry state after each test."""
        EntropySourceRegistry._registry = self._saved_registry
        EntropySourceRegistry._entry_points_loaded = self._saved_loaded

    def test_register_and_get(self) -> None:
        @EntropySourceRegistry.register("test_source")
        class TestSource(_DummySource):
            pass

        cls = EntropySourceRegistry.get("test_source")
        assert cls is TestSource

    def test_get_unknown_raises_key_error(self) -> None:
        # Force entry points to be "loaded" so it doesn't try to discover.
        EntropySourceRegistry._entry_points_loaded = True
        with pytest.raises(KeyError, match="no_such_source"):
            EntropySourceRegistry.get("no_such_source")

    def test_list_available_includes_registered(self) -> None:
        @EntropySourceRegistry.register("test_list")
        class ListSource(_DummySource):
            pass

        available = EntropySourceRegistry.list_available()
        assert "test_list" in available

    def test_list_available_is_sorted(self) -> None:
        EntropySourceRegistry.register("zzz_source")(_DummySource)
        EntropySourceRegistry.register("aaa_source")(_DummySource)
        available = EntropySourceRegistry.list_available()
        assert available == sorted(available)

    def test_entry_point_discovery(self) -> None:
        """Entry points should be loaded lazily on first get()."""
        EntropySourceRegistry._entry_points_loaded = False
        # Remove any existing registration for "ep_source".
        EntropySourceRegistry._registry.pop("ep_source", None)

        mock_ep = MagicMock()
        mock_ep.name = "ep_source"
        mock_ep.value = "some.module:SomeClass"
        mock_ep.load.return_value = _DummySource

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            cls = EntropySourceRegistry.get("ep_source")

        assert cls is _DummySource
        mock_ep.load.assert_called_once()

    def test_builtin_takes_precedence_over_entry_point(self) -> None:
        """If a source is registered via decorator, entry points don't override."""

        @EntropySourceRegistry.register("builtin_source")
        class BuiltinSource(_DummySource):
            pass

        EntropySourceRegistry._entry_points_loaded = False

        mock_ep = MagicMock()
        mock_ep.name = "builtin_source"
        mock_ep.value = "other.module:OtherClass"

        class OtherClass(_DummySource):
            pass

        mock_ep.load.return_value = OtherClass

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            cls = EntropySourceRegistry.get("builtin_source")

        assert cls is BuiltinSource
        # Entry point should NOT have been loaded (decorator takes precedence).
        mock_ep.load.assert_not_called()

    def test_broken_entry_point_does_not_crash(self) -> None:
        """A broken entry point should be logged and skipped."""
        EntropySourceRegistry._entry_points_loaded = False

        mock_ep = MagicMock()
        mock_ep.name = "broken_source"
        mock_ep.value = "broken.module:BrokenClass"
        mock_ep.load.side_effect = ImportError("module not found")

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            # Should not raise, but "broken_source" won't be available.
            available = EntropySourceRegistry.list_available()

        assert "broken_source" not in available

    def test_entry_points_loaded_only_once(self) -> None:
        """Entry points should only be loaded on the first call."""
        EntropySourceRegistry._entry_points_loaded = False

        with patch("importlib.metadata.entry_points", return_value=[]) as mock_eps:
            EntropySourceRegistry.list_available()
            EntropySourceRegistry.list_available()

        mock_eps.assert_called_once()

    def test_reset_clears_state(self) -> None:
        """_reset() should clear all registrations."""
        EntropySourceRegistry.register("reset_test")(_DummySource)
        EntropySourceRegistry._reset()
        assert "reset_test" not in EntropySourceRegistry._registry
        assert EntropySourceRegistry._entry_points_loaded is False

    def test_all_sources_returns_full_mapping(self) -> None:
        """all_sources() returns name->class for every registered source."""

        @EntropySourceRegistry.register("all_a")
        class SourceA(_DummySource):
            pass

        @EntropySourceRegistry.register("all_b")
        class SourceB(_DummySource):
            pass

        mapping = EntropySourceRegistry.all_sources()
        assert mapping["all_a"] is SourceA
        assert mapping["all_b"] is SourceB

    def test_all_sources_triggers_entry_point_load(self) -> None:
        """all_sources() loads entry points on first call, like list_available()."""
        EntropySourceRegistry._entry_points_loaded = False
        EntropySourceRegistry._registry.pop("ep_only", None)

        mock_ep = MagicMock()
        mock_ep.name = "ep_only"
        mock_ep.value = "some.module:SomeClass"
        mock_ep.load.return_value = _DummySource

        with patch("importlib.metadata.entry_points", return_value=[mock_ep]):
            mapping = EntropySourceRegistry.all_sources()

        assert "ep_only" in mapping
        assert mapping["ep_only"] is _DummySource

    def test_all_sources_returns_independent_copy(self) -> None:
        """Mutating the returned dict does not affect the registry."""
        EntropySourceRegistry.register("copy_test")(_DummySource)
        mapping = EntropySourceRegistry.all_sources()
        mapping.pop("copy_test", None)
        assert "copy_test" in EntropySourceRegistry._registry
