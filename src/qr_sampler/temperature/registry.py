"""Registry for temperature strategy implementations.

Built-in strategies are declared in an explicit lazy table
(:data:`TemperatureStrategyRegistry._BUILTINS`) and imported on first
``get()`` — no import-side-effect registration. Third-party strategies
register at runtime via the ``@TemperatureStrategyRegistry.register()``
decorator. The ``build()`` method handles the optional ``vocab_size``
constructor argument needed by some strategies (e.g., EDT).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.temperature.base import TemperatureStrategy


class TemperatureStrategyRegistry:
    """Registry mapping string names to TemperatureStrategy classes.

    Lookup precedence: runtime ``register()`` registrations, then the
    lazy builtin table. The ``build()`` class method instantiates the
    appropriate strategy, passing ``vocab_size`` if the constructor
    accepts it.
    """

    #: Built-in strategies, resolved lazily on first ``get()``.
    _BUILTINS: ClassVar[dict[str, str]] = {
        "fixed": "qr_sampler.temperature.fixed:FixedTemperatureStrategy",
        "edt": "qr_sampler.temperature.edt:EDTTemperatureStrategy",
        "hvh_drift": "qr_sampler.temperature.hvh_drift:HVHDriftStrategy",
        "coherence_gate": "qr_sampler.temperature.coherence_gate:CoherenceGateStrategy",
        "tt_exchange": "qr_sampler.temperature.tt_exchange:TTExchangeStrategy",
        "evdt_tt": "qr_sampler.temperature.evdt_tt:EVDTTTStrategy",
        "gdt": "qr_sampler.temperature.gdt:GDTStrategy",
        "dynatemp": "qr_sampler.temperature.dynatemp:DynaTempStrategy",
        "belltemp": "qr_sampler.temperature.belltemp:BellTempStrategy",
        "mix_temperatures": "qr_sampler.temperature.mix_temperatures:MixTemperaturesStrategy",
        "ring_buffer_ar": "qr_sampler.temperature.ring_buffer_ar:RingBufferARStrategy",
    }

    _registry: ClassVar[dict[str, type[TemperatureStrategy]]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[TemperatureStrategy]], type[TemperatureStrategy]]:
        """Decorator that registers a TemperatureStrategy class under *name*.

        Args:
            name: Identifier used in config ``temperature_strategy``.

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If *name* is already registered.
        """

        def decorator(klass: type[TemperatureStrategy]) -> type[TemperatureStrategy]:
            if name in cls._registry:
                raise ValueError(f"Temperature strategy '{name}' is already registered")
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def get(cls, name: str) -> type[TemperatureStrategy]:
        """Return the strategy class registered under *name*.

        Resolves the builtin table lazily on first use of a builtin name.

        Args:
            name: Identifier to look up.

        Returns:
            The registered TemperatureStrategy subclass.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name in cls._registry:
            return cls._registry[name]
        if name in cls._BUILTINS:
            module_path, _, attr = cls._BUILTINS[name].partition(":")
            klass: type[TemperatureStrategy] = getattr(importlib.import_module(module_path), attr)
            cls._registry[name] = klass
            return klass
        available = ", ".join(sorted(set(cls._registry) | set(cls._BUILTINS))) or "(none)"
        raise KeyError(f"Unknown temperature strategy '{name}'. Available: {available}")

    @classmethod
    def build(cls, config: Any, vocab_size: int) -> TemperatureStrategy:
        """Instantiate the strategy specified by *config.temperature_strategy*.

        If the strategy constructor accepts a ``vocab_size`` argument
        (detected via signature inspection), it is passed. Otherwise, the
        constructor is called with no arguments.

        Args:
            config: A QRSamplerConfig (or compatible object) with a
                ``temperature_strategy`` attribute.
            vocab_size: Vocabulary size of the model.

        Returns:
            A fully constructed TemperatureStrategy instance.
        """
        klass = cls.get(config.temperature_strategy)
        # iter-55: signature-based detection. The previous
        # ``try: klass(vocab_size) except TypeError: klass()`` swallowed
        # TypeErrors raised INSIDE a strategy's __init__ body, silently
        # constructing a mis-initialised no-arg instance instead of
        # surfacing the bug. Falls back to the legacy probe only when the
        # signature itself is unintrospectable (exotic C extensions).
        try:
            import inspect

            params = inspect.signature(klass).parameters
            takes_arg = any(
                p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                for p in params.values()
            )
        except (ValueError, TypeError):
            try:
                return klass(vocab_size)  # type: ignore[call-arg]
            except TypeError:
                return klass()
        return klass(vocab_size) if takes_arg else klass()  # type: ignore[call-arg]

    @classmethod
    def list_registered(cls) -> list[str]:
        """Return sorted list of registered strategy names (builtins included)."""
        return sorted(set(cls._registry) | set(cls._BUILTINS))
