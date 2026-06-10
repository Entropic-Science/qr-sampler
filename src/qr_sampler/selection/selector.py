"""CDF-based token selector.

Implements the full token selection pipeline: temperature scaling -> top-k ->
softmax -> top-p -> descending sort -> CDF -> binary search with the uniform
value u from signal amplification.

Semantic interpretation of u:
    u near 0.0: selects the most probable token (conservative)
    u near 1.0: selects the least probable surviving token (creative)
"""

from __future__ import annotations

import numpy as np

from qr_sampler.exceptions import TokenSelectionError
from qr_sampler.selection.types import SelectionResult

# iter-55: CDF fast-path head size. The selection CDF concentrates its
# mass in the head of the descending-probability order, so sorting the
# top-_CDF_FAST_HEAD candidates (found via O(n) argpartition) almost
# always suffices to locate the selected token — replacing the full
# O(n log n) argsort over the whole vocabulary (~152k for Qwen3.6) that
# previously ran on EVERY token. When ``u`` falls beyond the head's
# nonzero cumulative mass the selector escalates to the original
# full-sort path, so selection semantics are preserved exactly (tie
# order among equal probabilities was already unspecified — np.argsort's
# default quicksort is not stable).
_CDF_FAST_HEAD = 512

# Only engage the fast path when it meaningfully beats the full sort.
_CDF_FAST_MIN_VOCAB = _CDF_FAST_HEAD * 4


class TokenSelector:
    """Stateless CDF-based token selector.

    All methods are either static or depend only on the arguments passed.
    The selector has no mutable state, making it safe for concurrent use.
    """

    def select(
        self,
        logits: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
        u: float,
        min_p: float = 0.0,
    ) -> SelectionResult:
        """Select one token from the logit distribution using CDF lookup.

        Pipeline:
            1. Temperature scaling: logits / T
            2. Top-k filtering: keep only k highest logits
            3. Softmax: convert to probabilities
            4. Min-p mask: keep tokens with prob >= min_p * max_prob
            5. Top-p (nucleus): keep minimal set with cumulative prob >= p
            6. Descending sort by probability
            7. Build CDF via cumulative sum
            8. Binary search with u to select token

        Args:
            logits: 1-D logit array (vocab_size,).
            temperature: Sampling temperature (must be > 0).
            top_k: Number of top tokens to keep (<=0 disables).
            top_p: Nucleus sampling threshold in (0, 1] (1.0 disables).
            u: Uniform random value from signal amplification, in (0, 1).
            min_p: Dynamic floor relative to peak probability. Keeps tokens with
                ``prob >= min_p * max_prob``. ``0.0`` (default) disables, making
                this a no-op so existing call sites are unaffected.

        Returns:
            SelectionResult with the selected token and diagnostics.

        Raises:
            TokenSelectionError: If no candidate tokens survive filtering.
        """
        # 1. Temperature scaling.
        if temperature > 0:
            scaled = logits / temperature
        else:
            # Zero temperature -> greedy (pick max logit).
            scaled = logits.copy()
            max_idx = int(np.argmax(scaled))
            return SelectionResult(
                token_id=max_idx,
                token_rank=0,
                token_prob=1.0,
                num_candidates=1,
                diagnostics={"greedy": True},
            )

        # 2. Top-k filtering.
        scaled, effective_k = self._apply_top_k(scaled, top_k)

        # 3. Softmax.
        probs = self._stable_softmax(scaled)

        # 4. Min-p mask.
        probs, effective_min_p_candidates = self._apply_min_p(probs, min_p)

        # 5. Top-p (nucleus) filtering.
        probs, effective_n = self._apply_top_p(probs, top_p)

        if effective_n == 0:
            raise TokenSelectionError("No candidate tokens survived top-k and top-p filtering")

        # 6-8. CDF selection.
        vocab_idx, rank, prob, num_candidates = self._cdf_select(probs, u)

        return SelectionResult(
            token_id=vocab_idx,
            token_rank=rank,
            token_prob=prob,
            num_candidates=num_candidates,
            diagnostics={
                "effective_top_k": effective_k,
                "effective_top_p_candidates": effective_n,
                "effective_min_p_candidates": effective_min_p_candidates,
                "min_p_used": min_p,
                "u": u,
            },
        )

    @staticmethod
    def _apply_min_p(probs: np.ndarray, min_p: float) -> tuple[np.ndarray, int]:
        """Dynamic floor mask: keep tokens with prob >= min_p * max_prob.

        Acts as a peak-relative cutoff that adapts to distribution shape: when
        the distribution is peaked, few tokens survive; when flat, many do. This
        is a numpy port of the V6 HVH-Drift reference (hvh_drift.py:157-163).

        Args:
            probs: Probability array (vocab_size,). Must already be normalized.
            min_p: Threshold scale in [0, 1]. ``0.0`` disables (no-op).

        Returns:
            Tuple of (renormalized probability array, number of surviving tokens).
            If ``min_p == 0.0``, returns the input unchanged with the count of
            non-zero entries.
        """
        if min_p <= 0.0:
            return probs, int(np.sum(probs > 0))

        top_prob = float(probs.max())
        mask = probs >= min_p * top_prob

        if not mask.any():
            # Pathological case: reserve argmax to avoid empty CDF.
            mask = np.zeros_like(probs, dtype=bool)
            mask[int(np.argmax(probs))] = True

        result = np.where(mask, probs, 0.0)
        total = float(result.sum())
        if total > 0.0:
            result = result / total

        return result, int(mask.sum())

    @staticmethod
    def _apply_top_k(logits: np.ndarray, k: int) -> tuple[np.ndarray, int]:
        """Keep only the top-k logits, setting the rest to -inf.

        Args:
            logits: 1-D logit array (may be modified in place).
            k: Number of top tokens to keep. <=0 disables filtering.

        Returns:
            Tuple of (filtered logits, effective k).
        """
        vocab_size = len(logits)
        if k <= 0 or k >= vocab_size:
            return logits, vocab_size

        # Use argpartition for O(n) selection of top-k indices.
        threshold_idx = vocab_size - k
        partitioned = np.argpartition(logits, threshold_idx)
        below_k = partitioned[:threshold_idx]

        result = logits.copy()
        result[below_k] = -np.inf
        return result, k

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax via shift-by-max.

        Args:
            logits: 1-D logit array (may contain -inf for masked tokens).

        Returns:
            Probability array of the same shape, summing to 1.0.
        """
        # iter-55: the max over ALL logits equals the max over finite
        # logits whenever any finite value exists (-inf never wins), so
        # the boolean-mask fancy-index copy of the full vocab is only
        # needed on the all-masked degenerate branch.
        max_logit = np.max(logits)
        if not np.isfinite(max_logit):
            finite_mask = np.isfinite(logits)
            if not np.any(finite_mask):
                # All masked -- return uniform over all tokens (degenerate case).
                n = len(logits)
                return np.full(n, 1.0 / n)
            max_logit = np.max(logits[finite_mask])

        shifted = logits - max_logit
        # -inf - max_logit is still -inf, exp(-inf) = 0.
        exp_shifted = np.exp(shifted)
        total = np.sum(exp_shifted)

        if total == 0.0:
            # Shouldn't happen after shift, but guard anyway.
            n = int(np.sum(finite_mask))
            probs = np.zeros_like(logits, dtype=np.float64)
            probs[finite_mask] = 1.0 / max(n, 1)
            return probs

        result: np.ndarray = exp_shifted / total
        return result

    @staticmethod
    def _apply_top_p(probs: np.ndarray, top_p: float) -> tuple[np.ndarray, int]:
        """Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p.

        Tokens outside the nucleus are zeroed and probabilities are renormalized.

        Args:
            probs: Probability array (vocab_size,).
            top_p: Cumulative probability threshold in (0, 1]. 1.0 disables.

        Returns:
            Tuple of (renormalized probability array, number of surviving tokens).
        """
        if top_p >= 1.0:
            return probs, int(np.sum(probs > 0))

        # Sort in descending probability order.
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)

        # Find the cutoff: first index where cumulative >= top_p.
        # Include that token (so we always have at least one).
        cutoff_mask = cumulative >= top_p
        cutoff_idx = int(np.argmax(cutoff_mask)) if np.any(cutoff_mask) else len(sorted_probs) - 1

        # Zero out tokens beyond the cutoff.
        result = np.zeros_like(probs)
        surviving_indices = sorted_indices[: cutoff_idx + 1]
        result[surviving_indices] = probs[surviving_indices]

        # Renormalize.
        total = np.sum(result)
        if total > 0:
            result = result / total

        num_surviving = cutoff_idx + 1
        return result, num_surviving

    @staticmethod
    def _cdf_select(probs: np.ndarray, u: float) -> tuple[int, int, float, int]:
        """Select a token via CDF binary search.

        Sorts tokens by descending probability, builds a CDF, and uses
        np.searchsorted to find the token corresponding to uniform value u.

        Args:
            probs: Probability array (vocab_size,). Must sum to ~1.0.
            u: Uniform random value in (0, 1).

        Returns:
            Tuple of (vocabulary index, rank, probability, num_candidates).

        Raises:
            TokenSelectionError: If no tokens have non-zero probability.
        """
        # Get non-zero probability tokens.
        num_candidates = int(np.count_nonzero(probs > 0))
        if num_candidates == 0:
            raise TokenSelectionError("No tokens with non-zero probability for CDF selection")

        # iter-55 fast path: locate the selection inside the top-K head
        # (O(n) argpartition + O(K log K) sort) instead of full-sorting
        # the whole vocabulary. Escalates to the full sort whenever ``u``
        # is not strictly covered by the head's nonzero cumulative mass —
        # including the u-beyond-total-mass clamp case — so every
        # selection this path returns is identical to the full path's.
        n = probs.size
        if n >= _CDF_FAST_MIN_VOCAB:
            head_count = min(_CDF_FAST_HEAD, n)
            head_unsorted = np.argpartition(probs, n - head_count)[n - head_count :]
            head_order = np.argsort(probs[head_unsorted])[::-1]
            head_indices = head_unsorted[head_order]
            head_probs = probs[head_indices]
            head_nonzero = int(np.count_nonzero(head_probs > 0))
            if head_nonzero > 0:
                head_cdf = np.cumsum(head_probs[:head_nonzero])
                if u < float(head_cdf[-1]):
                    rank = int(np.searchsorted(head_cdf, u, side="left"))
                    rank = min(rank, head_nonzero - 1)
                    return (
                        int(head_indices[rank]),
                        rank,
                        float(head_probs[rank]),
                        num_candidates,
                    )
            # Fall through: selection lies beyond the head (rare tail
            # draw) — take the exact full-sort path below.

        # Sort by descending probability.
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Trim to non-zero candidates.
        candidate_indices = sorted_indices[:num_candidates]
        candidate_probs = sorted_probs[:num_candidates]

        # Build CDF.
        cdf = np.cumsum(candidate_probs)

        # Binary search: find first CDF value >= u.
        rank = int(np.searchsorted(cdf, u, side="left"))

        # Clamp rank to valid range.
        rank = min(rank, num_candidates - 1)

        vocab_idx = int(candidate_indices[rank])
        prob = float(candidate_probs[rank])

        return vocab_idx, rank, prob, num_candidates
