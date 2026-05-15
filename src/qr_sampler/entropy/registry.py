"""Entropy source registry with entry-point auto-discovery.

Built-in sources are registered at module import time via the
``@register_entropy_source`` decorator. Third-party sources from other
packages are discovered lazily on the first :meth:`EntropySourceRegistry.get`
call via the ``qr_sampler.entropy_sources`` entry-point group.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.entropy.base import EntropySource

logger = logging.getLogger("qr_sampler")

_ENTRY_POINT_GROUP = "qr_sampler.entropy_sources"


class EntropySourceRegistry:
    """Registry for entropy source classes.

    Discovery chain:

    1. Built-in sources registered via ``@register_entropy_source`` decorator
    2. Third-party sources discovered via ``qr_sampler.entropy_sources``
       entry points (loaded lazily on first ``get()`` call)
    """

    _registry: ClassVar[dict[str, type[EntropySource]]] = {}
    _entry_points_loaded: ClassVar[bool] = False

    @classmethod
    def register(cls, name: str) -> Callable[[type[EntropySource]], type[EntropySource]]:
        """Decorator to register a source class under a string key.

        Args:
            name: Unique identifier for the source (e.g., ``'system'``).

        Returns:
            The original class, unmodified.

        Example::

            @EntropySourceRegistry.register("my_source")
            class MySource(EntropySource):
                ...
        """

        def decorator(source_cls: type[EntropySource]) -> type[EntropySource]:
            cls._registry[name] = source_cls
            return source_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[EntropySource]:
        """Look up a source class by name.

        Loads entry points on the first call if not already loaded.

        Args:
            name: Registered identifier for the source.

        Returns:
            The entropy source class (not an instance).

        Raises:
            KeyError: If *name* is not found after loading entry points.
        """
        if name in cls._registry:
            return cls._registry[name]

        # Lazy-load third-party entry points.
        if not cls._entry_points_loaded:
            cls._load_entry_points()
            if name in cls._registry:
                return cls._registry[name]

        available = ", ".join(sorted(cls._registry.keys())) or "(none)"
        raise KeyError(f"Unknown entropy source: {name!r}. Available: {available}")

    @classmethod
    def list_available(cls) -> list[str]:
        """Return all registered source names.

        Triggers entry-point loading if not yet done.

        Returns:
            Sorted list of registered source identifiers.
        """
        if not cls._entry_points_loaded:
            cls._load_entry_points()
        return sorted(cls._registry.keys())

    @classmethod
    def all_sources(cls) -> dict[str, type[EntropySource]]:
        """Return a copy of the full ``name -> class`` registry mapping.

        Triggers entry-point loading on first call so third-party sources
        are included. The returned dict is a shallow copy; mutating it
        does not affect the registry. Used by engine adapters at startup
        to pre-initialise pipelines for every available entropy source.

        Returns:
            A new dict mapping each registered identifier to its class.
        """
        if not cls._entry_points_loaded:
            cls._load_entry_points()
        return dict(cls._registry)

    @classmethod
    def _load_entry_points(cls) -> None:
        """Discover and register sources from the entry-point group.

        Each entry point maps a name to a fully-qualified class path.
        Errors during individual entry-point loading are logged as warnings
        but do not prevent other sources from loading.
        """
        cls._entry_points_loaded = True
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception:  # Intentional: must not crash on broken metadata
            logger.warning("Failed to load entry points for %s", _ENTRY_POINT_GROUP, exc_info=True)
            return

        for ep in eps:
            if ep.name in cls._registry:
                # Built-in decorator registration takes precedence.
                continue
            try:
                source_cls = ep.load()
                cls._registry[ep.name] = source_cls
                logger.debug("Loaded entropy source %r from entry point", ep.name)
            except Exception:  # Intentional: one bad plugin must not block others
                logger.warning(
                    "Failed to load entropy source entry point %r: %s",
                    ep.name,
                    ep.value,
                    exc_info=True,
                )

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state. **Test-only** — not part of public API."""
        cls._registry.clear()
        cls._entry_points_loaded = False


# Convenience alias used as a decorator in source modules.
register_entropy_source = EntropySourceRegistry.register
