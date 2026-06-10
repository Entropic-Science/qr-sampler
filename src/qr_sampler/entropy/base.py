"""Abstract base class for all entropy sources.

Every entropy source — whether quantum hardware, OS randomness, CPU timing,
or a test mock — implements this interface. The ABC provides a default
``get_random_float64()`` that delegates to ``get_random_bytes()`` and a
concrete ``health_check()`` method. Subclasses must implement the four
abstract members: ``name``, ``is_available``, ``get_random_bytes()``, and
``close()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class EntropySource(ABC):
    """Abstract base for all entropy sources.

    Implementations must provide random bytes on demand.
    The ``get_random_bytes()`` call must satisfy the just-in-time constraint:
    physical entropy generation occurs only when this method is called.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source identifier (e.g., ``'quantum_grpc'``, ``'system'``)."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the source can currently provide entropy."""

    @abstractmethod
    def get_random_bytes(self, n: int) -> bytes:
        """Return exactly *n* random bytes.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes of entropy.

        Raises:
            EntropyUnavailableError: If the source cannot provide bytes.
        """

    def get_random_float64(
        self,
        shape: tuple[int, ...],
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return random float64 values in [0, 1).

        The default implementation converts ``get_random_bytes()`` output to
        float64 via ``np.frombuffer(dtype=uint8) / 255.0``. Subclasses may
        override for more efficient native float generation.

        If *out* is provided, the result is written into it (zero-allocation
        hot path). If *out* is ``None``, a new array is allocated and returned.

        Args:
            shape: Desired output shape.
            out: Optional pre-allocated array to write into.

        Returns:
            Array of float64 values in [0, 1).
        """
        total = 1
        for dim in shape:
            total *= dim
        raw = self.get_random_bytes(total)
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) / 255.0
        if out is not None:
            np.copyto(out, values.reshape(shape))
            return out
        return values.reshape(shape)

    @abstractmethod
    def close(self) -> None:
        """Release resources (channels, connections, file handles)."""

    def warmup(self) -> None:  # noqa: B027 -- deliberate optional hook, not a forgotten abstractmethod
        """Eagerly establish any expensive connections this source needs.

        Default no-op — sources that don't have a connection lifecycle
        (system entropy, mock sources) inherit this and do nothing.

        Sources that DO have a connection (e.g. ``QuantumGrpcSource``)
        override this to open the channel + verify reachability *before*
        the first ``get_random_bytes()`` call. The engine adapter calls
        ``warmup()`` after pipeline construction so that per-token
        fetches never pay the channel-establishment cost.

        Idempotent: safe to call multiple times. Should not raise on
        unreachable backends — fallback wrappers handle that case
        transparently at fetch time.
        """

    def health_check(self) -> dict[str, Any]:
        """Return a status dictionary for this source.

        Returns:
            Dictionary with at least ``'source'`` and ``'healthy'`` keys.
        """
        return {"source": self.name, "healthy": self.is_available}
