"""Ring-Buffer-AR: anti-repetition gate over recently emitted tokens.

Per-request stateful strategy ported from createmp-evalsuite's V6
``RingBufferARProcessor`` (V6 research spec §7.6), replication-first. A
ring buffer holds the last ``rba_buffer_n`` emitted token ids (fed by the
pipeline's ``observe_selected_token`` hook — history observed at token t
first affects token t+1, a structural one-token lag). Each candidate v is
penalised by similarity to the recent-token context:

    penalty_v = rba_lam * max(0, sim_v - rba_threshold)
    logits'   = logits - penalty                (published via the
                                                 transformed-logits seam)
    T_t       = clip(rba_t, 0.3, 2.2)           (selector applies T to the
                                                 penalised logits — the V6
                                                 order: penalty, then T,
                                                 then min_p)
    min_p_t   = clip(rba_min_p, 0, 0.15)

Similarity has two modes:

- **Embedding mode** (V6-faithful): after ``attach_embeddings(E)`` with a
  ``(vocab, d)`` numpy table, ``sim_v = cos(E_v, centroid(buffered))``.
  Engine adapters that can reach the model's input-embedding table inject
  it here; the qr-sampler vLLM V1 logits processor currently has no model
  handle, so this mode is exercised in tests/offline analysis until an
  injection path exists.
- **Exact-id fallback** (default, loudly documented): without an embedding
  table, ``sim_v = 1.0`` for token ids currently in the buffer and ``0.0``
  otherwise — a plain ring-buffer repetition penalty of
  ``rba_lam * (1 - rba_threshold)`` logits on recently emitted ids. The
  assessment (§7.4) found the V6 cosine gate at threshold 0.65 "mostly
  penalises the recently-emitted tokens themselves and near-duplicates",
  so the fallback keeps the dominant mechanism while dropping only the
  near-duplicate generalisation. ``diagnostics["embeddings_attached"]``
  records which mode produced every token.

Family hypothesis (assessment §9): ``V6_RBA_R00_07`` posted the largest
static-twin gap in the V6 corpus (8.60 vs ~3.0) — possibly the most real
V6 effect, possibly winner's curse; replication comes first.

Defaults are the V6 §7.6 predicted values.

**Static-clone parameterisation (FR-8.5):** ``rba_lam = 0`` disables the
gate entirely — no transformed logits are emitted and the strategy reduces
exactly to fixed ``T = clip(rba_t)`` with constant
``min_p = clip(rba_min_p)`` on every token, on the identical selector path
as a fixed-strategy run.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.temperature.base import (
    TemperatureResult,
    TemperatureStrategy,
    compute_entropy_varentropy,
)

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

# Repo-wide V6 guardrail box — hard clamps applied to the static knobs.
_TEMP_CLAMP: tuple[float, float] = (0.3, 2.2)
_MIN_P_CLAMP: tuple[float, float] = (0.0, 0.15)

_logger = logging.getLogger("qr_sampler")


class RingBufferARStrategy(TemperatureStrategy):
    """Per-request stateful Ring-Buffer-AR strategy.

    Each instance owns its own emitted-token history; engine adapters
    build a fresh instance per request (invariant 14) so the buffer never
    leaks across sequences. ``vocab_size`` is used to validate an attached
    embedding table.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize empty history.

        Args:
            vocab_size: Vocabulary size; validates ``attach_embeddings``.
        """
        self._vocab_size = vocab_size
        self._history: deque[int] = deque()
        self._embeddings: np.ndarray | None = None
        self._emb_norms: np.ndarray | None = None

    def attach_embeddings(self, embedding_matrix: np.ndarray) -> None:
        """Attach a ``(vocab, d)`` input-embedding table (embedding mode).

        Args:
            embedding_matrix: 2-D numpy array, one row per vocabulary id.

        Raises:
            ValueError: If the table is not 2-D with ``vocab_size`` rows.
        """
        if embedding_matrix.ndim != 2 or embedding_matrix.shape[0] != self._vocab_size:
            raise ValueError(
                f"embedding_matrix must have shape ({self._vocab_size}, d), "
                f"got {embedding_matrix.shape}"
            )
        self._embeddings = embedding_matrix
        norms = np.linalg.norm(embedding_matrix, axis=-1)
        self._emb_norms = np.maximum(norms, 1e-12)

    def observe_selected_token(self, token_id: int) -> None:
        """Pipeline hook: record the token id selected for this step.

        Args:
            token_id: Vocabulary id of the just-selected token.
        """
        self._history.append(token_id)

    def _similarities(self, buffered: list[int]) -> np.ndarray:
        """Per-vocab similarity to the buffered-token context.

        Args:
            buffered: Token ids currently in the ring buffer (non-empty).

        Returns:
            1-D array of ``sim_v`` over the vocabulary — cosine to the
            buffered-embedding centroid in embedding mode, or the exact-id
            indicator in fallback mode.
        """
        if self._embeddings is not None and self._emb_norms is not None:
            centroid = self._embeddings[np.asarray(buffered, dtype=np.intp)].mean(axis=0)
            c_norm = max(float(np.linalg.norm(centroid)), 1e-12)
            sims: np.ndarray = (self._embeddings @ centroid) / (self._emb_norms * c_norm)
            return sims
        sims = np.zeros(self._vocab_size, dtype=np.float64)
        sims[np.asarray(buffered, dtype=np.intp)] = 1.0
        return sims

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Penalise repetition-similar candidates, then static (T, min_p).

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the 5 Ring-Buffer-AR
                hyperparameters (``rba_buffer_n``, ``rba_lam``,
                ``rba_threshold``, ``rba_t``, ``rba_min_p``).

        Returns:
            TemperatureResult with the static temperature, Shannon entropy
            of the RAW distribution, and diagnostics containing ``min_p``,
            ``varentropy``, ``n_buffered``, ``n_penalized``,
            ``embeddings_attached`` (plus ``transformed_logits`` whenever
            the gate is active).
        """
        h, vh = compute_entropy_varentropy(logits)

        # Trim history to the configured window (config arrives per call;
        # the constructor has no config access by registry contract).
        while len(self._history) > config.rba_buffer_n:
            self._history.popleft()
        buffered = list(self._history)

        temperature = float(np.clip(config.rba_t, *_TEMP_CLAMP))
        min_p = float(np.clip(config.rba_min_p, *_MIN_P_CLAMP))

        diagnostics: dict[str, Any] = {
            "strategy": "ring_buffer_ar",
            "min_p": min_p,
            "varentropy": vh,
            "n_buffered": len(buffered),
            "n_penalized": 0,
            "embeddings_attached": self._embeddings is not None,
        }

        if config.rba_lam > 0.0 and buffered:
            excess = np.maximum(self._similarities(buffered) - config.rba_threshold, 0.0)
            penalty = config.rba_lam * excess
            n_penalized = int(np.count_nonzero(penalty))
            if n_penalized > 0:
                diagnostics["n_penalized"] = n_penalized
                diagnostics["transformed_logits"] = logits - penalty

        if _logger.isEnabledFor(logging.DEBUG):
            _logger.debug(
                "ring_buffer_ar: H=%.4f VH=%.4f buffered=%d penalized=%d T=%.4f min_p=%.4f",
                h,
                vh,
                len(buffered),
                diagnostics["n_penalized"],
                temperature,
                min_p,
            )

        return TemperatureResult(
            temperature=temperature,
            shannon_entropy=h,
            diagnostics=diagnostics,
        )
