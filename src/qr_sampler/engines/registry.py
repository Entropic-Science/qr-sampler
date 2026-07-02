"""Registry for engine adapter implementations.

Built-in adapters are declared in an explicit lazy table
(:data:`EngineAdapterRegistry._BUILTINS`) and imported on first ``get()``
— no import-side-effect registration. Third-party adapters are discovered
via the ``qr_sampler.engine_adapters`` entry-point group; the builtin
table takes precedence.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.engines.base import EngineAdapter

logger = logging.getLogger("qr_sampler")

_ENTRY_POINT_GROUP = "qr_sampler.engine_adapters"


class EngineAdapterRegistry:
    """Registry for engine adapter classes.

    Discovery chain (first hit wins):

    1. Runtime registrations via ``@EngineAdapterRegistry.register()``
    2. The lazy builtin table (``_BUILTINS``), imported on demand
    3. Third-party adapters discovered via ``qr_sampler.engine_adapters``
       entry points (loaded lazily on first miss)
    """

    #: Built-in adapters, resolved lazily on first ``get()``.
    _BUILTINS: ClassVar[dict[str, str]] = {
        "vllm": "qr_sampler.engines.vllm:VLLMAdapter",
    }

    _registry: ClassVar[dict[str, type[EngineAdapter]]] = {}
    _entry_points_loaded: ClassVar[bool] = False

    @classmethod
    def register(cls, name: str) -> Callable[[type[EngineAdapter]], type[EngineAdapter]]:
        """Decorator to register an adapter class under a string key.

        Args:
            name: Unique identifier for the adapter (e.g., ``'vllm'``).

        Returns:
            The original class, unmodified.

        Example::

            @EngineAdapterRegistry.register("my_engine")
            class MyAdapter(EngineAdapter):
                ...
        """

        def decorator(adapter_cls: type[EngineAdapter]) -> type[EngineAdapter]:
            cls._registry[name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EngineAdapter]:
        """Look up an adapter class by name.

        Resolves the builtin table lazily; loads entry points on the
        first miss if not already loaded.

        Args:
            name: Registered identifier for the adapter.

        Returns:
            The engine adapter class (not an instance).

        Raises:
            KeyError: If *name* is not found after loading entry points.
        """
        if name in cls._registry:
            return cls._registry[name]
        if name in cls._BUILTINS:
            module_path, _, attr = cls._BUILTINS[name].partition(":")
            adapter_cls: type[EngineAdapter] = getattr(importlib.import_module(module_path), attr)
            cls._registry[name] = adapter_cls
            return adapter_cls

        # Lazy-load third-party entry points.
        if not cls._entry_points_loaded:
            cls._load_entry_points()
            if name in cls._registry:
                return cls._registry[name]

        available = ", ".join(sorted(set(cls._registry) | set(cls._BUILTINS))) or "(none)"
        raise KeyError(f"Unknown engine adapter: {name!r}. Available: {available}")

    @classmethod
    def list_available(cls) -> list[str]:
        """Return all registered adapter names (builtins included).

        Triggers entry-point loading if not yet done; builtin names are
        listed without importing their modules.

        Returns:
            Sorted list of registered adapter identifiers.
        """
        if not cls._entry_points_loaded:
            cls._load_entry_points()
        return sorted(set(cls._registry) | set(cls._BUILTINS))

    @classmethod
    def _load_entry_points(cls) -> None:
        """Discover and register adapters from the entry-point group.

        Each entry point maps a name to a fully-qualified class path.
        Names already claimed by a runtime registration or the builtin
        table are skipped (builtins take precedence). Errors during
        individual entry-point loading are logged as warnings but do not
        prevent other adapters from loading.
        """
        cls._entry_points_loaded = True
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception:  # Intentional: must not crash on broken metadata
            logger.warning(
                "Failed to load entry points for %s",
                _ENTRY_POINT_GROUP,
                exc_info=True,
            )
            return

        for ep in eps:
            if ep.name in cls._registry or ep.name in cls._BUILTINS:
                # Builtin / runtime registration takes precedence.
                continue
            try:
                adapter_cls = ep.load()
                cls._registry[ep.name] = adapter_cls
                logger.debug("Loaded engine adapter %r from entry point", ep.name)
            except Exception:  # Intentional: one bad plugin must not block others
                logger.warning(
                    "Failed to load engine adapter entry point %r: %s",
                    ep.name,
                    ep.value,
                    exc_info=True,
                )

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state. **Test-only** -- not part of public API."""
        cls._registry.clear()
        cls._entry_points_loaded = False
