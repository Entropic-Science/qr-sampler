"""Data types for the engine-agnostic sampling pipeline.

Defines ``SamplingResult``, the frozen dataclass returned by
``SamplingPipeline.sample_token()``, and ``PrefetchContext``, the
per-request carrier for the pipelined (commit-then-fetch) entropy path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.entropy.base import DrawMeta
    from qr_sampler.logging.types import TokenSamplingRecord


@dataclass(frozen=True, slots=True)
class SamplingResult:
    """Result of a single token sampling operation.

    Returned by ``SamplingPipeline.sample_token()``. The ``one_hot`` array
    is engine-agnostic (numpy); engine adapters convert it to their native
    tensor format.

    Attributes:
        token_id: Selected vocabulary index.
        one_hot: 1-D numpy array of shape ``(vocab_size,)`` with ``-inf``
            everywhere except ``0.0`` at ``token_id``. ``None`` when the
            caller passed ``build_onehot=False`` (engine adapters that
            force the one-hot directly on their own tensors skip the
            ~vocab-size numpy allocation per token).
        record: Full sampling record for logging and diagnostics.
        next_ticket: In-flight entropy prefetch ticket for this request's
            NEXT token, fired immediately after this token's selection
            (commit-then-fetch). ``None`` when prefetch is disabled or the
            source has no async transport. The engine adapter stores it in
            per-request state and threads it back in via
            ``PrefetchContext.ticket`` on the next step.
        draw_meta: Metadata of the server-integrated draw that produced
            this token's ``u`` (server-draw mode only). ``None`` on the
            local byte-amplification path and when the draw path degraded
            to fallback bytes. Temperature strategies observe it via the
            duck-typed ``observe_draw_meta`` hook — the pipeline invokes
            that directly, so most consumers only need the record fields.
    """

    token_id: int
    one_hot: np.ndarray | None
    record: TokenSamplingRecord
    next_ticket: Any | None = None
    draw_meta: DrawMeta | None = None


@dataclass(slots=True)
class PrefetchContext:
    """Per-request state for the pipelined entropy fetch.

    Owned by the engine adapter (one per in-flight request) and passed to
    ``SamplingPipeline.sample_token()`` each step.

    Attributes:
        salt: Random per-request salt mixed into every commitment nonce so
            nonces are unlinkable across requests and unforgeable without
            the client's audit log.
        step: 0-based index of the token currently being sampled. Fetch
            ``i``'s commitment nonce is
            ``derive_commit_nonce(salt, i, token_id_{i-1})`` with the
            ``token_id_{-1} = -1`` sentinel for the first fetch.
        ticket: The in-flight ticket for the CURRENT token's entropy
            (fired when the previous token was selected), or ``None`` to
            fetch serially.
    """

    salt: bytes
    step: int = 0
    ticket: Any | None = None
