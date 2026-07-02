"""System entropy source using ``os.urandom()``.

This is the default fallback source. It is cryptographically secure and
always available on all platforms, but does not use quantum randomness.
"""

from __future__ import annotations

import os

from qr_sampler.entropy.base import EntropySource


class SystemEntropySource(EntropySource):
    """``os.urandom()`` wrapper — always available, cryptographically secure.

    Suitable as a fallback when the primary QRNG source is unreachable.
    Not suitable for consciousness-research experiments that require true
    quantum randomness.
    """

    @property
    def name(self) -> str:
        """Return ``'system'``."""
        return "system"

    @property
    def is_available(self) -> bool:
        """Always returns ``True`` — ``os.urandom()`` never fails."""
        return True

    def get_random_bytes(self, n: int) -> bytes:
        """Return *n* bytes from the OS CSPRNG.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes of entropy from ``os.urandom()``.
        """
        return os.urandom(n)

    def close(self) -> None:
        """No-op — no resources to release."""
