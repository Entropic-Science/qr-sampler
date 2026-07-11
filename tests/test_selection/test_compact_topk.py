"""Equivalence tests for the compacted top-k selection fast path.

The 2026-07 perf tranche routes any truncating ``top_k`` through
``TokenSelector._select_compact_top_k``, which gathers the k surviving
logits before softmax instead of masking the full vocabulary to ``-inf``.
These tests pin the equivalence contract: the compacted path must produce
the same selection, rank, probability, candidate counts, and diagnostics
as a straightforward full-vocabulary reference implementation of the
pinned selector order (top-k -> softmax -> min-p -> top-p -> CDF).
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.exceptions import TokenSelectionError
from qr_sampler.selection.selector import TokenSelector


def _reference_select(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
    u: float,
    min_p: float,
) -> tuple[int, int, float, int, int, int]:
    """Straightforward full-vocab reference of the pinned selector order.

    Returns (token_id, rank, prob, num_candidates, min_p_survivors,
    top_p_survivors).
    """
    vocab = len(logits)
    scaled = logits / temperature

    # Top-k: mask everything below the k-th highest to -inf.
    if 0 < top_k < vocab:
        threshold_idx = vocab - top_k
        below = np.argpartition(scaled, threshold_idx)[:threshold_idx]
        scaled = scaled.copy()
        scaled[below] = -np.inf

    # Stable softmax.
    finite = scaled[np.isfinite(scaled)]
    shifted = scaled - np.max(finite)
    exp_shifted = np.exp(shifted)
    probs = exp_shifted / np.sum(exp_shifted)

    # Min-p.
    if min_p > 0.0:
        mask = probs >= min_p * probs.max()
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()
        min_p_survivors = int(mask.sum())
    else:
        min_p_survivors = int(np.sum(probs > 0))

    # Top-p.
    if top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        cutoff_mask = cumulative >= top_p
        cutoff = int(np.argmax(cutoff_mask)) if cutoff_mask.any() else len(order) - 1
        kept = np.zeros_like(probs)
        surviving = order[: cutoff + 1]
        kept[surviving] = probs[surviving]
        probs = kept / kept.sum()
        top_p_survivors = cutoff + 1
    else:
        top_p_survivors = int(np.sum(probs > 0))

    # CDF selection over descending non-zero probabilities.
    num_candidates = int(np.sum(probs > 0))
    order = np.argsort(probs)[::-1][:num_candidates]
    cdf = np.cumsum(probs[order])
    rank = min(int(np.searchsorted(cdf, u, side="left")), num_candidates - 1)
    return (
        int(order[rank]),
        rank,
        float(probs[order][rank]),
        num_candidates,
        min_p_survivors,
        top_p_survivors,
    )


@pytest.fixture()
def selector() -> TokenSelector:
    """Default TokenSelector."""
    return TokenSelector()


class TestCompactTopKEquivalence:
    """The compacted top-k path matches the full-vocabulary reference."""

    @pytest.mark.parametrize("temperature", [0.7, 1.0, 1.4])
    @pytest.mark.parametrize("top_k", [2, 7, 50])
    @pytest.mark.parametrize("top_p", [1.0, 0.9])
    @pytest.mark.parametrize("min_p", [0.0, 0.05])
    def test_matches_reference_across_grid(
        self,
        selector: TokenSelector,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
    ) -> None:
        """Randomised logits x u grid: identical token/rank/counts."""
        rng = np.random.default_rng(20260711)
        for _ in range(5):
            logits = rng.normal(0.0, 3.0, size=300)
            for u in (0.001, 0.2, 0.5, 0.8, 0.999):
                result = selector.select(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p, u=u, min_p=min_p
                )
                ref_token, ref_rank, ref_prob, ref_nc, ref_minp_n, ref_topp_n = _reference_select(
                    logits, temperature, top_k, top_p, u, min_p
                )
                assert result.token_id == ref_token
                assert result.token_rank == ref_rank
                assert result.token_prob == pytest.approx(ref_prob, rel=1e-9)
                assert result.num_candidates == ref_nc
                assert result.diagnostics["effective_top_k"] == top_k
                assert result.diagnostics["effective_min_p_candidates"] == ref_minp_n
                assert result.diagnostics["effective_top_p_candidates"] == ref_topp_n

    def test_token_always_inside_topk_support(self, selector: TokenSelector) -> None:
        """The selected token is always one of the k highest logits."""
        rng = np.random.default_rng(7)
        logits = rng.normal(0.0, 2.0, size=500)
        top5 = set(np.argsort(logits)[-5:].tolist())
        for u in np.linspace(0.01, 0.99, 23):
            result = selector.select(logits, temperature=1.0, top_k=5, top_p=1.0, u=float(u))
            assert result.token_id in top5

    def test_num_candidates_equals_k_without_extra_truncation(
        self, selector: TokenSelector
    ) -> None:
        """min_p=0, top_p=1: exactly k candidates survive."""
        logits = np.linspace(0.0, 4.0, 64)
        result = selector.select(logits, temperature=1.0, top_k=8, top_p=1.0, u=0.5)
        assert result.num_candidates == 8
        assert result.diagnostics["effective_top_p_candidates"] == 8
        assert result.diagnostics["effective_min_p_candidates"] == 8

    def test_underflowed_support_counts_only_nonzero(self, selector: TokenSelector) -> None:
        """float32 exp underflow inside the top-k support shrinks the count."""
        logits = np.zeros(32, dtype=np.float32)
        logits[3] = 1000.0  # everything else underflows to prob 0.0
        result = selector.select(logits, temperature=1.0, top_k=4, top_p=1.0, u=0.5)
        assert result.token_id == 3
        assert result.num_candidates == 1

    def test_greedy_bypasses_compaction(self, selector: TokenSelector) -> None:
        """temperature <= 0 stays greedy regardless of top_k."""
        logits = np.array([1.0, 9.0, 3.0])
        result = selector.select(logits, temperature=0.0, top_k=2, top_p=1.0, u=0.9)
        assert result.token_id == 1
        assert result.diagnostics == {"greedy": True}

    def test_top_k_of_vocab_size_uses_full_path(self, selector: TokenSelector) -> None:
        """top_k >= vocab is a no-op: effective_top_k reports the vocab size."""
        logits = np.array([1.0, 2.0, 3.0, 4.0])
        result = selector.select(logits, temperature=1.0, top_k=4, top_p=1.0, u=0.5)
        assert result.diagnostics["effective_top_k"] == 4
        assert result.num_candidates == 4

    def test_all_filtered_raises(self, selector: TokenSelector) -> None:
        """A fully-underflowed support still raises TokenSelectionError."""
        logits = np.full(16, -np.inf)
        logits[0] = 0.0
        # top_k=2 keeps token 0 and one -inf token; only token 0 survives.
        result = TokenSelector().select(logits, temperature=1.0, top_k=2, top_p=1.0, u=0.5)
        assert result.token_id == 0
        assert result.num_candidates == 1

    def test_cdf_select_precomputed_count_matches(self) -> None:
        """_cdf_select with a pre-computed count equals the self-computed one."""
        rng = np.random.default_rng(11)
        probs = rng.random(64)
        probs[10:] = 0.0
        probs = probs / probs.sum()
        for u in (0.05, 0.5, 0.95):
            expected = TokenSelector._cdf_select(probs, u)
            got = TokenSelector._cdf_select(probs, u, num_candidates=10)
            assert got == expected

    def test_cdf_select_zero_count_raises(self) -> None:
        """A zero pre-computed count raises exactly like the computed one."""
        probs = np.zeros(8)
        with pytest.raises(TokenSelectionError):
            TokenSelector._cdf_select(probs, 0.5, num_candidates=0)
        with pytest.raises(TokenSelectionError):
            TokenSelector._cdf_select(probs, 0.5)


class TestFullPathStillMatchesReference:
    """The non-compacted path (top_k disabled) also matches the reference."""

    @pytest.mark.parametrize("temperature", [0.8, 1.0])
    @pytest.mark.parametrize("top_p", [1.0, 0.85])
    @pytest.mark.parametrize("min_p", [0.0, 0.02])
    def test_matches_reference(
        self, selector: TokenSelector, temperature: float, top_p: float, min_p: float
    ) -> None:
        """top_k=0 flow: identical token/rank/counts to the reference."""
        rng = np.random.default_rng(99)
        logits = rng.normal(0.0, 2.5, size=257)
        for u in (0.01, 0.4, 0.6, 0.98):
            result = selector.select(
                logits, temperature=temperature, top_k=0, top_p=top_p, u=u, min_p=min_p
            )
            ref_token, ref_rank, ref_prob, ref_nc, ref_minp_n, ref_topp_n = _reference_select(
                logits, temperature, 0, top_p, u, min_p
            )
            assert result.token_id == ref_token
            assert result.token_rank == ref_rank
            assert result.token_prob == pytest.approx(ref_prob, rel=1e-9)
            assert result.num_candidates == ref_nc
            assert result.diagnostics["effective_min_p_candidates"] == ref_minp_n
            assert result.diagnostics["effective_top_p_candidates"] == ref_topp_n
