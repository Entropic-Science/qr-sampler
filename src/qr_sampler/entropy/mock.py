"""Configurable mock entropy source for testing and bias simulation.

Generates bytes from a normal distribution with configurable mean, allowing
deterministic tests (via seed) and controlled bias experiments.
"""

from __future__ import annotations

import threading

import numpy as np

from qr_sampler.entropy.base import EntropySource


class MockUniformSource(EntropySource):
    """Configurable mock entropy source for testing.

    Generates bytes from a normal distribution with configurable *mean*.
    Supports seeded reproducibility for deterministic tests.

    Usage:
        - **Null hypothesis testing**: ``mean=127.5`` (no bias)
        - **Weak-signal bias simulation**: ``mean=128.0`` (positive bias)

    Args:
        mean: Centre of the normal distribution for byte generation.
            Default is 127.5, the expected mean of a uniform byte distribution.
        seed: Optional RNG seed for reproducible output.
    """

    _MOCK_BYTE_STD: float = 40.0
    """Fixed standard deviation for test consistency."""

    def __init__(self, mean: float = 127.5, seed: int | None = None) -> None:
        self._mean = mean
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # numpy Generators are not thread-safe, and this source can serve
        # as the production fallback leg (fallback_mode="mock_uniform")
        # under the adapter's concurrent per-row sampling.
        self._rng_lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return ``'mock_uniform'``."""
        return "mock_uniform"

    @property
    def is_available(self) -> bool:
        """Always returns ``True``."""
        return True

    def get_random_bytes(self, n: int) -> bytes:
        """Generate *n* bytes from a normal distribution.

        Values are drawn from ``N(mean, _MOCK_BYTE_STD)`` and clamped to
        ``[0, 255]``.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes.
        """
        with self._rng_lock:
            samples = self._rng.normal(loc=self._mean, scale=self._MOCK_BYTE_STD, size=n)
        clamped = np.clip(samples, 0, 255).astype(np.uint8)
        return bytes(clamped)

    def close(self) -> None:
        """No-op — no resources to release."""
