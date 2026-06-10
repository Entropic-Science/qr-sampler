"""Registry for temperature strategy implementations.

Uses a decorator pattern for registration, mirroring the amplifier registry.
The ``build()`` method handles the optional ``vocab_size`` constructor
argument needed by some strategies (e.g., EDT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from qr_sampler.temperature.base import TemperatureStrategy


class TemperatureStrategyRegistry:
    """Registry mapping string names to TemperatureStrategy classes.

    Built-in strategies register via the
    ``@TemperatureStrategyRegistry.register()`` decorator. The ``build()``
    class method instantiates the appropriate strategy, passing ``vocab_size``
    if the constructor accepts it.
    """

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

        Args:
            name: Identifier to look up.

        Returns:
            The registered TemperatureStrategy subclass.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise KeyError(f"Unknown temperature strategy '{name}'. Available: {available}")
        return cls._registry[name]

    @classmethod
    def build(cls, config: Any, vocab_size: int) -> TemperatureStrategy:
        """Instantiate the strategy specified by *config.temperature_strategy*.

        If the strategy constructor accepts a ``vocab_size`` argument
        (detected via try/except), it is passed. Otherwise, the constructor
        is called with no arguments.

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
        """Return sorted list of registered strategy names."""
        return sorted(cls._registry)
