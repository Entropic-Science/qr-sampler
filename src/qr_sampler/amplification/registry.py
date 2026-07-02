"""Registry for signal amplifier implementations.

Built-in amplifiers are declared in an explicit lazy table
(:data:`AmplifierRegistry._BUILTINS`) and imported on first ``get()`` —
no import-side-effect registration. Third-party amplifiers register at
runtime via the ``@AmplifierRegistry.register()`` decorator.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.amplification.base import SignalAmplifier


class AmplifierRegistry:
    """Registry mapping string names to SignalAmplifier classes.

    Lookup precedence: runtime ``register()`` registrations, then the
    lazy builtin table. The ``build()`` class method instantiates the
    appropriate amplifier based on the config's ``signal_amplifier_type``
    field.
    """

    #: Built-in amplifiers, resolved lazily on first ``get()``.
    _BUILTINS: ClassVar[dict[str, str]] = {
        "zscore_mean": "qr_sampler.amplification.zscore:ZScoreMeanAmplifier",
        "ecdf": "qr_sampler.amplification.ecdf:ECDFAmplifier",
        "zscore_thought": "qr_sampler.amplification.zscore_thought:ZScoreThoughtAmplifier",
    }

    _registry: ClassVar[dict[str, type[SignalAmplifier]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[type[SignalAmplifier]], type[SignalAmplifier]]:
        """Decorator that registers a SignalAmplifier class under *name*.

        Args:
            name: Identifier used in config ``signal_amplifier_type``.

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(klass: type[SignalAmplifier]) -> type[SignalAmplifier]:
            if name in cls._registry:
                raise ValueError(f"Amplifier '{name}' is already registered")
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[SignalAmplifier]:
        """Return the amplifier class registered under *name*.

        Resolves the builtin table lazily on first use of a builtin name.

        Args:
            name: Identifier to look up.

        Returns:
            The registered SignalAmplifier subclass.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name in cls._registry:
            return cls._registry[name]
        if name in cls._BUILTINS:
            module_path, _, attr = cls._BUILTINS[name].partition(":")
            klass: type[SignalAmplifier] = getattr(importlib.import_module(module_path), attr)
            cls._registry[name] = klass
            return klass
        available = ", ".join(sorted(set(cls._registry) | set(cls._BUILTINS))) or "(none)"
        raise KeyError(f"Unknown signal amplifier '{name}'. Available: {available}")

    @classmethod
    def build(cls, config: Any) -> SignalAmplifier:
        """Instantiate the amplifier specified by *config.signal_amplifier_type*.

        Args:
            config: A QRSamplerConfig (or compatible object) with a
                ``signal_amplifier_type`` attribute.

        Returns:
            A fully constructed SignalAmplifier instance.
        """
        klass = cls.get(config.signal_amplifier_type)
        return klass(config)  # type: ignore[call-arg]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of registered amplifier names (builtins included)."""
        return sorted(set(cls._registry) | set(cls._BUILTINS))
