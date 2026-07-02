"""OpenEntropy entropy source using the ``openentropy`` Python library.

Wraps the ``openentropy.EntropyPool`` API to provide hardware-sourced
entropy from any platform-available source (e.g., camera noise, audio
noise, sensor jitter). The ``openentropy`` package is optional — this
module degrades gracefully when it is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qr_sampler.entropy.base import EntropySource
from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

logger = logging.getLogger("qr_sampler")

# ---------------------------------------------------------------------------
# Import guard — no crash when openentropy is not installed
# ---------------------------------------------------------------------------

try:
    from openentropy import EntropyPool

    _OPENENTROPY_AVAILABLE = True
except ImportError:
    _OPENENTROPY_AVAILABLE = False


class OpenEntropySource(EntropySource):
    """Hardware entropy via the ``openentropy`` library.

    Uses ``EntropyPool.auto()`` to auto-discover platform-available entropy
    sources (camera noise, audio noise, sensor jitter, etc.) and exposes
    them through the standard ``EntropySource`` interface.

    The ``openentropy`` package must be installed separately::

        pip install openentropy

    Configuration fields used from ``QRSamplerConfig``:

    * ``oe_conditioning`` — conditioning mode (``"raw"``, ``"vonneumann"``,
      or ``"sha256"``). Per-request overridable.
    * ``oe_sources`` — comma-separated list of specific source names to
      sample from. Empty string means use all sources via ``collect_all()``.
    * ``oe_parallel`` — whether to collect from sources in parallel.
    * ``oe_timeout`` — timeout in seconds for ``collect_all()``.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """Initialize the OpenEntropy source.

        Args:
            config: Sampler configuration providing ``oe_*`` fields.

        Raises:
            EntropyUnavailableError: If the ``openentropy`` package is not
                installed.
        """
        if not _OPENENTROPY_AVAILABLE:
            raise EntropyUnavailableError(
                "openentropy package not installed. Install with: pip install openentropy"
            )
        self._config = config
        self._pool = EntropyPool.auto()
        self._closed = False

    @property
    def name(self) -> str:
        """Return ``'openentropy'``."""
        return "openentropy"

    @property
    def is_available(self) -> bool:
        """Whether OpenEntropy has at least one working source."""
        return _OPENENTROPY_AVAILABLE and self._pool.source_count > 0

    def get_random_bytes(self, n: int) -> bytes:
        """Return exactly *n* random bytes from OpenEntropy sources.

        If ``oe_sources`` is configured, samples from each named source
        individually via ``get_source_bytes()`` and combines the results.
        Otherwise, calls ``collect_all()`` followed by ``get_bytes()``.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes of entropy.

        Raises:
            EntropyUnavailableError: If the source is closed or collection
                fails.
        """
        if self._closed:
            raise EntropyUnavailableError("OpenEntropySource is closed")

        try:
            oe_sources = self._config.oe_sources.strip()
            if oe_sources:
                # Sample from specific named sources.
                source_names = [s.strip() for s in oe_sources.split(",") if s.strip()]
                combined = b""
                for source_name in source_names:
                    # Some sources have per-call byte limits, so collect
                    # in chunks of up to 8192 bytes until we have enough.
                    remaining = n
                    while remaining > 0:
                        request_size = min(remaining, 8192)
                        chunk = self._pool.get_source_bytes(
                            source_name,
                            request_size,
                            conditioning=self._config.oe_conditioning,
                        )
                        if chunk is None:
                            raise EntropyUnavailableError(
                                f"OpenEntropy source '{source_name}' returned no data"
                            )
                        combined += chunk
                        remaining -= len(chunk)
                return combined[:n]

            # Collect from all sources, then draw bytes.
            self._pool.collect_all(
                parallel=self._config.oe_parallel,
                timeout=self._config.oe_timeout,
            )
            raw_bytes = self._pool.get_bytes(n, conditioning=self._config.oe_conditioning)
            result: bytes = bytes(raw_bytes)
            return result
        except RuntimeError as e:
            raise EntropyUnavailableError(f"OpenEntropy failed: {e}") from e

    def close(self) -> None:
        """Mark the source as closed (idempotent)."""
        self._closed = True

    def health_check(self) -> dict[str, object]:
        """Return a status dictionary for this source.

        Returns:
            Dictionary with source name, health status, source count,
            and conditioning mode.
        """
        if not _OPENENTROPY_AVAILABLE:
            return {
                "source": "openentropy",
                "healthy": False,
                "reason": "openentropy not installed",
            }
        return {
            "source": "openentropy",
            "healthy": self.is_available,
            "source_count": self._pool.source_count,
            "conditioning": self._config.oe_conditioning,
        }
