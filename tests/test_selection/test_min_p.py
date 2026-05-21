"""Tests for the per-token min-p mask in TokenSelector.

Min-p is the dynamic floor introduced as part of the HVH-Drift sampler
(creative-sampling preset). It acts on probabilities AFTER softmax and BEFORE
top-p, keeping tokens whose probability is at least ``min_p * max_prob``.
Default value ``0.0`` must be a complete no-op so every existing call site
is unaffected (NFR-7).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qr_sampler.selection.selector import TokenSelector


@pytest.fixture()
def selector() -> TokenSelector:
    """Default TokenSelector."""
    return TokenSelector()


def _logits_from_probs(target_probs: np.ndarray) -> np.ndarray:
    """Build logits whose softmax(logits, T=1) == target_probs exactly.

    log of zero-prob tokens becomes -inf which is fine for the softmax path.
    """
    return np.log(target_probs)


class TestZeroMinPIsNoop:
    """min_p=0.0 must be byte-identical to omitting the parameter."""

    @pytest.mark.parametrize(
        "logits",
        [
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            np.array([10.0, 0.0, -5.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([-np.inf, 5.0, 3.0, -np.inf]),
        ],
    )
    @pytest.mark.parametrize("top_k", [0, 3])
    @pytest.mark.parametrize("top_p", [1.0, 0.9])
    @pytest.mark.parametrize("u", [0.001, 0.5, 0.999])
    def test_zero_min_p_is_noop(
        self,
        selector: TokenSelector,
        logits: np.ndarray,
        top_k: int,
        top_p: float,
        u: float,
    ) -> None:
        """Default min_p=0.0 must reproduce the legacy 4-arg select() outcome."""
        without = selector.select(logits, temperature=1.0, top_k=top_k, top_p=top_p, u=u)
        with_zero = selector.select(
            logits, temperature=1.0, top_k=top_k, top_p=top_p, u=u, min_p=0.0
        )

        assert without.token_id == with_zero.token_id
        assert without.token_rank == with_zero.token_rank
        assert without.token_prob == with_zero.token_prob
        assert without.num_candidates == with_zero.num_candidates


class TestThreshold:
    """The min-p threshold removes tokens with prob < min_p * top_prob."""

    def test_threshold_removes_mass_below_min_p_times_top(self, selector: TokenSelector) -> None:
        """probs=[0.5, 0.25, 0.15, 0.1], min_p=0.5 -> threshold 0.25 keeps first two."""
        probs = np.array([0.5, 0.25, 0.15, 0.1])
        logits = _logits_from_probs(probs)

        # u=0.999 should select the LAST surviving token (rank 1 after masking,
        # since only 2 tokens survive: 0.5 and 0.25 -> renormalized to ~0.667/0.333).
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.999, min_p=0.5)

        assert result.diagnostics["effective_min_p_candidates"] == 2
        assert result.diagnostics["min_p_used"] == 0.5
        assert result.num_candidates == 2
        # Last surviving in descending-prob order is token 1 (prob 0.25 -> 0.333).
        assert result.token_id == 1

    def test_renormalization_sums_to_one(self, selector: TokenSelector) -> None:
        """After masking, the internal CDF probabilities must sum to ~1.0."""
        probs = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        logits = _logits_from_probs(probs)

        # Direct inspection of the helper guarantees the renormalization step.
        result_probs, count = TokenSelector._apply_min_p(probs.copy(), min_p=0.4)

        # 0.4 * 0.4 = 0.16 threshold => keep probs >= 0.16 -> indices 0, 1
        assert count == 2
        assert math.isclose(result_probs.sum(), 1.0, abs_tol=1e-12)
        assert result_probs[0] > 0.0
        assert result_probs[1] > 0.0
        # Tokens below threshold zeroed.
        assert result_probs[2] == 0.0
        assert result_probs[3] == 0.0
        assert result_probs[4] == 0.0

        # End-to-end: diagnostics report the correct count for the selector call.
        e2e = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5, min_p=0.4)
        assert e2e.diagnostics["effective_min_p_candidates"] == 2

    def test_all_masked_fallback_keeps_argmax(self, selector: TokenSelector) -> None:
        """min_p=1.0 keeps only tokens equal to the max; degenerate case must not crash."""
        # Three tokens equal-ish but only one strictly maximal after softmax.
        probs = np.array([0.5, 0.3, 0.2])
        logits = _logits_from_probs(probs)

        # min_p=1.0 means threshold = 0.5 (max). Only token 0 exactly meets it.
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5, min_p=1.0)
        assert result.token_id == 0
        assert result.diagnostics["effective_min_p_candidates"] == 1
        assert result.num_candidates == 1
        assert math.isclose(result.token_prob, 1.0, abs_tol=1e-9)

    def test_all_masked_fallback_no_division_by_zero(self) -> None:
        """The helper's argmax fallback is exercised when nothing meets the threshold.

        With min_p marginally above 1.0, no probability can satisfy
        ``prob >= min_p * max_prob`` (since max_prob is the largest entry).
        The helper must reserve argmax and still produce a sum-to-one output.
        """
        probs = np.array([0.5, 0.3, 0.2])
        result_probs, count = TokenSelector._apply_min_p(probs.copy(), min_p=1.0 + 1e-9)
        assert count == 1
        assert math.isclose(result_probs.sum(), 1.0, abs_tol=1e-12)
        assert result_probs[int(np.argmax(probs))] == pytest.approx(1.0)


class TestPipelineOrder:
    """Min-p applies BEFORE top-p so a sharp floor culls first."""

    def test_min_p_then_top_p_order(self, selector: TokenSelector) -> None:
        """Min-p first vs top-p first must produce different selections.

        Construct probs [0.45, 0.30, 0.15, 0.10]:
          - min_p=0.5 first: threshold = 0.225 -> keep [0.45, 0.30] -> renorm [0.6, 0.4],
            then top_p=0.5 keeps top 1 (0.6 >= 0.5) -> always token 0.
          - top_p=0.5 first: keep top 1 (0.45 >= 0.5? no -> top 2 [0.45, 0.30] -> renorm
            [0.6, 0.4]). Then min_p on the renormalized distribution would also keep
            both. Different intermediate state.

        We assert the qr-sampler outcome corresponds to "min-p first" by picking
        a u that probes the second-ranked token under min-p-first (which has been
        culled) vs top-p-first.
        """
        probs = np.array([0.45, 0.30, 0.15, 0.10])
        logits = _logits_from_probs(probs)

        # With min_p=0.5 first then top_p=0.5:
        #   step 4 (min-p): threshold 0.225 -> keep [0.45, 0.30] -> [0.6, 0.4]
        #   step 5 (top-p=0.5): cumulative >= 0.5 hits at idx 0 (0.6) -> single
        #     survivor with renormalized prob 1.0
        #   any u in (0,1) selects token 0.
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=0.5, u=0.7, min_p=0.5)
        assert result.token_id == 0
        assert result.num_candidates == 1

        # Without min-p (top-p only): probs [0.45, 0.30, 0.15, 0.10], top_p=0.5
        # cumulative >= 0.5 hits at idx 1 (0.75) -> survivors [0.45, 0.30] renormalized
        # to [0.6, 0.4]. u=0.7 lands past the first CDF bin (0.6) so token 1 wins.
        result_no_minp = selector.select(logits, temperature=1.0, top_k=0, top_p=0.5, u=0.7)
        assert result_no_minp.token_id == 1
        assert result_no_minp.num_candidates == 2

        # Confirms qr-sampler chose the min-p-first outcome.
        assert result.token_id != result_no_minp.token_id


class TestDiagnostics:
    """Selector diagnostics expose min-p surface fields."""

    def test_diagnostics_include_min_p_keys(self, selector: TokenSelector) -> None:
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5, min_p=0.0)
        assert "effective_min_p_candidates" in result.diagnostics
        assert "min_p_used" in result.diagnostics
        assert result.diagnostics["min_p_used"] == 0.0
        # With min_p=0.0 every non-zero token survives.
        assert result.diagnostics["effective_min_p_candidates"] == 5
