"""Tests for EngineAdapterRegistry.

Covers decorator registration, entry-point discovery, listing,
and error handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from qr_sampler.engines.base import EngineAdapter
from qr_sampler.engines.registry import EngineAdapterRegistry

if TYPE_CHECKING:
    from qr_sampler.core.pipeline import SamplingPipeline


class TestEngineAdapterRegistry:
    """Test EngineAdapterRegistry decorator and lookup."""

    def test_vllm_registered(self) -> None:
        """The vllm adapter resolves through the lazy builtin table."""
        cls = EngineAdapterRegistry.get("vllm")
        from qr_sampler.engines.vllm import VLLMAdapter

        assert cls is VLLMAdapter

    def test_vllm_in_builtin_table(self) -> None:
        """The builtin table is the explicit home of the shipped adapter."""
        assert EngineAdapterRegistry._BUILTINS == {"vllm": "qr_sampler.engines.vllm:VLLMAdapter"}

    def test_unknown_adapter_raises(self) -> None:
        """Looking up an unregistered name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown engine adapter"):
            EngineAdapterRegistry.get("nonexistent_engine_xyz")

    def test_list_available_includes_vllm(self) -> None:
        """list_available() includes 'vllm'."""
        available = EngineAdapterRegistry.list_available()
        assert "vllm" in available

    def test_list_available_returns_sorted(self) -> None:
        """list_available() returns a sorted list."""
        available = EngineAdapterRegistry.list_available()
        assert available == sorted(available)

    def test_register_custom_adapter(self) -> None:
        """Custom adapters can be registered via decorator."""

        @EngineAdapterRegistry.register("test_custom")
        class CustomAdapter(EngineAdapter):
            def get_pipeline(self) -> SamplingPipeline:
                raise NotImplementedError

        try:
            assert EngineAdapterRegistry.get("test_custom") is CustomAdapter
            assert "test_custom" in EngineAdapterRegistry.list_available()
        finally:
            # Clean up to avoid polluting other tests.
            EngineAdapterRegistry._registry.pop("test_custom", None)

    def test_get_is_class_not_instance(self) -> None:
        """get() returns the class, not an instance."""
        cls = EngineAdapterRegistry.get("vllm")
        assert isinstance(cls, type)
        assert issubclass(cls, EngineAdapter)
