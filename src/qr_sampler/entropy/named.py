"""Instance-name wrapper for entropy sources.

``InstanceNamedSource`` wraps any :class:`~qr_sampler.entropy.base.EntropySource`
and reports a caller-chosen *instance name* instead of the inner source's
type name, delegating every operation unchanged. It exists for the named
entropy-source instances feature (``QRSamplerConfig.entropy_source_instances``):
two pipelines of the same source type (e.g. ``quantum_grpc``) can then be
told apart end-to-end — ``TokenSamplingRecord.entropy_source_used``, the
``entropy.degraded`` / ``entropy.recovered`` log legs, and the cross-process
status file's ``primary_name`` all carry the instance name, so a PRNG lane
served through a quantum-shaped transport is loudly labelled as PRNG.

The wrapper is applied to the PRIMARY source (inside any
``FallbackEntropySource``), deliberately: the fallback wrapper's diagnostics
read ``primary.name`` for every leg label, so renaming the primary renames
every operator-visible surface without touching the record schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qr_sampler.entropy.base import EntropySource

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.entropy.base import DrawMeta


class InstanceNamedSource(EntropySource):
    """Delegating wrapper that renames an entropy source to an instance name.

    Every method forwards to the wrapped source; only :attr:`name` (and the
    ``source`` key of :meth:`health_check`) report the instance name.
    Unknown attributes forward via ``__getattr__`` so capability probes on
    the wrapped source (e.g. ``enable_status_publishing``) keep working.

    Args:
        inner: The real entropy source being renamed.
        instance_name: The declared instance name to report.
    """

    def __init__(self, inner: EntropySource, instance_name: str) -> None:
        self._inner = inner
        self._instance_name = instance_name

    @property
    def name(self) -> str:
        """The declared instance name (NOT the inner source's type name)."""
        return self._instance_name

    @property
    def inner(self) -> EntropySource:
        """The wrapped source. Test/diagnostic introspection only."""
        return self._inner

    @property
    def is_available(self) -> bool:
        """Delegate availability to the wrapped source."""
        return self._inner.is_available

    def get_random_bytes(self, n: int) -> bytes:
        """Delegate to the wrapped source."""
        return self._inner.get_random_bytes(n)

    def get_random_float64(
        self,
        shape: tuple[int, ...],
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Delegate to the wrapped source (it may override the default)."""
        return self._inner.get_random_float64(shape, out)

    def prefetch(self, n: int, nonce: int | None = None) -> Any | None:
        """Delegate to the wrapped source."""
        return self._inner.prefetch(n, nonce)

    def get_random_bytes_with_ticket(self, n: int, ticket: Any | None) -> bytes:
        """Delegate to the wrapped source."""
        return self._inner.get_random_bytes_with_ticket(n, ticket)

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: Any | None = None
    ) -> tuple[float, DrawMeta]:
        """Delegate to the wrapped source."""
        return self._inner.get_draw(block_bytes, source_id, ticket)

    def prefetch_draw(
        self, block_bytes: int, source_id: str, nonce: int | None = None
    ) -> Any | None:
        """Delegate to the wrapped source."""
        return self._inner.prefetch_draw(block_bytes, source_id, nonce)

    def warmup(self) -> None:
        """Delegate to the wrapped source."""
        self._inner.warmup()

    def close(self) -> None:
        """Delegate to the wrapped source."""
        self._inner.close()

    def health_check(self) -> dict[str, Any]:
        """The wrapped source's health dict, relabelled with the instance name.

        The inner type name is preserved under ``inner_source`` so an
        operator can still see what transport actually serves the instance.
        """
        inner_health = self._inner.health_check()
        return {
            **inner_health,
            "source": self._instance_name,
            "inner_source": inner_health.get("source"),
        }

    def __getattr__(self, item: str) -> Any:
        """Forward unknown attributes to the wrapped source.

        Keeps duck-typed capability probes (``getattr(source, "...", None)``)
        transparent across the rename.
        """
        return getattr(self._inner, item)
