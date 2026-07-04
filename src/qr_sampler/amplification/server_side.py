"""Server-side draw "amplifier" — a marker for the server-integrated path.

``ServerDrawAmplifier`` fills the amplifier slot when the uniform ``u`` is
produced by the entropy SERVER (the ``qr_purity.PurityService`` protocol:
the server integrates a raw block itself and returns ``u = Phi(z)``),
so no local bytes-to-uniform amplification exists. The sampling pipeline
branches on :attr:`ServerDrawAmplifier.requires_server_draw` and calls
``EntropySource.get_draw()`` instead of the fetch-then-amplify stages;
``amplify()`` is deliberately a dead end.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from qr_sampler.amplification.base import AmplificationResult, SignalAmplifier
from qr_sampler.exceptions import SignalAmplificationError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig


class ServerDrawAmplifier(SignalAmplifier):
    """Marker amplifier: ``u`` comes from a server-integrated draw.

    Registered as ``"server"``. The pipeline detects
    ``requires_server_draw`` (duck-typed via ``getattr``, default False)
    and replaces the local fetch-then-amplify stages with a single
    ``get_draw()`` round trip. When the draw path degrades (server
    unavailable), the pipeline substitutes a lazily-built local
    ``zscore_mean`` amplifier — it never calls :meth:`amplify` here.
    """

    #: Pipeline branch switch: stages 2-3 become ``get_draw()``.
    requires_server_draw: ClassVar[bool] = True

    def __init__(self, config: QRSamplerConfig) -> None:
        """Accept the registry-uniform config argument; no local params.

        Args:
            config: The sampler configuration (unused — every draw
                parameter travels per-request via ``draw_source_id`` /
                ``draw_block_bytes``, read by the pipeline itself).
        """
        del config

    def amplify(self, raw_bytes: bytes) -> AmplificationResult:
        """Always raises — the server-draw amplifier has no local path.

        Raises:
            SignalAmplificationError: Unconditionally. Reaching this means
                a caller bypassed the pipeline's ``requires_server_draw``
                branch; failing loudly beats silently fabricating a ``u``
                that never saw the server.
        """
        raise SignalAmplificationError("server-draw amplifier has no local amplify path")
