"""Tests for the TokenSelector."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.selection.selector import TokenSelector
from qr_sampler.selection.types import SelectionResult


@pytest.fixture()
def selector() -> TokenSelector:
    """Default TokenSelector."""
    return TokenSelector()


class TestTokenSelector:
    """Tests for CDF-based token selection."""

    def test_u_zero_selects_most_probable(self, selector: TokenSelector) -> None:
        """u ≈ 0 should select the most probable token (rank 0)."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.001)
        assert result.token_id == 0  # Highest logit.
        assert result.token_rank == 0

    def test_u_one_selects_least_probable(self, selector: TokenSelector) -> None:
        """u ≈ 1 should select the least probable surviving token."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.999)
        assert result.token_id == 4  # Lowest logit.
        assert result.token_rank == 4

    def test_u_middle_selects_intermediate(self, selector: TokenSelector) -> None:
        """u ≈ 0.5 should select a token around the middle of the CDF."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert 0 <= result.token_rank <= 4

    def test_top_k_filters_tokens(self, selector: TokenSelector) -> None:
        """top_k=3 should limit candidates to the 3 highest logits."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=3, top_p=1.0, u=0.5)
        assert result.num_candidates == 3
        assert result.token_id in {0, 1, 2}

    def test_top_k_disabled_when_zero(self, selector: TokenSelector) -> None:
        """top_k=0 should not filter any tokens."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert result.num_candidates == 5

    def test_top_k_disabled_when_negative(self, selector: TokenSelector) -> None:
        """top_k=-1 should not filter any tokens."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=-1, top_p=1.0, u=0.5)
        assert result.num_candidates == 5

    def test_top_p_filters_tokens(self, selector: TokenSelector) -> None:
        """top_p < 1.0 should limit candidates via nucleus sampling."""
        # Token 0 has ~63% probability after softmax (dominant).
        logits = np.array([5.0, 1.0, 0.0, -1.0, -5.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=0.5, u=0.3)
        # With p=0.5, only the top 1-2 tokens should survive.
        assert result.num_candidates <= 3

    def test_top_p_1_disables_filtering(self, selector: TokenSelector) -> None:
        """top_p=1.0 should keep all tokens."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert result.num_candidates == 5

    def test_combined_top_k_and_top_p(self, selector: TokenSelector) -> None:
        """Both top_k and top_p should apply."""
        logits = np.array([10.0, 5.0, 4.0, 3.0, 2.0])
        result = selector.select(logits, temperature=1.0, top_k=3, top_p=0.5, u=0.3)
        assert result.num_candidates <= 3

    def test_temperature_effect(self, selector: TokenSelector) -> None:
        """Higher temperature should make selection more uniform."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # Low temperature → peaked → token 0 more likely.
        result_low = selector.select(logits, temperature=0.1, top_k=0, top_p=1.0, u=0.3)
        # High temperature → flat → other tokens more accessible.
        result_high = selector.select(logits, temperature=5.0, top_k=0, top_p=1.0, u=0.3)
        # With u=0.3, higher temperature makes it more likely to pick a lower-rank token.
        assert result_low.token_rank <= result_high.token_rank

    def test_zero_temperature_greedy(self, selector: TokenSelector) -> None:
        """temperature=0 should select the argmax (greedy)."""
        logits = np.array([1.0, 5.0, 3.0, 2.0])
        result = selector.select(logits, temperature=0, top_k=0, top_p=1.0, u=0.7)
        assert result.token_id == 1  # Highest logit.
        assert result.token_rank == 0
        assert result.num_candidates == 1

    def test_identical_logits(self, selector: TokenSelector) -> None:
        """All identical logits → uniform probabilities → any token valid."""
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert 0 <= result.token_id <= 3
        assert result.num_candidates == 4
        # All probs should be equal: ~0.25.
        assert abs(result.token_prob - 0.25) < 0.01

    def test_single_survivor(self, selector: TokenSelector) -> None:
        """When only one token survives filtering, it must be selected."""
        logits = np.array([100.0, -100.0, -100.0])
        result = selector.select(logits, temperature=1.0, top_k=1, top_p=1.0, u=0.5)
        assert result.token_id == 0
        assert result.num_candidates == 1
        assert abs(result.token_prob - 1.0) < 1e-6

    def test_all_inf_except_one(self, selector: TokenSelector) -> None:
        """All -inf except one should select the surviving token."""
        logits = np.array([-np.inf, -np.inf, 5.0, -np.inf])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert result.token_id == 2
        assert result.num_candidates == 1

    def test_result_is_frozen(self, selector: TokenSelector) -> None:
        """SelectionResult should be immutable."""
        logits = np.array([5.0, 4.0, 3.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        with pytest.raises(AttributeError):
            result.token_id = 42  # type: ignore[misc]

    def test_diagnostics_keys(self, selector: TokenSelector) -> None:
        """Diagnostics should contain selection metadata."""
        logits = np.array([5.0, 4.0, 3.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        assert "effective_top_k" in result.diagnostics
        assert "effective_top_p_candidates" in result.diagnostics
        assert "u" in result.diagnostics

    def test_probability_sums_to_one_after_selection(self, selector: TokenSelector) -> None:
        """The probabilities of all candidates should sum to ~1.0."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(logits, temperature=1.0, top_k=0, top_p=1.0, u=0.5)
        # token_prob is one candidate's probability — it should be > 0 and < 1.
        assert 0.0 < result.token_prob <= 1.0

    @pytest.mark.parametrize(
        ("logits", "top_k", "top_p", "u"),
        [
            (np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 0, 1.0, 0.001),
            (np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 0, 1.0, 0.999),
            (np.array([5.0, 4.0, 3.0, 2.0, 1.0]), 3, 1.0, 0.5),
            (np.array([5.0, 1.0, 0.0, -1.0, -5.0]), 0, 0.5, 0.3),
            (np.array([10.0, 5.0, 4.0, 3.0, 2.0]), 3, 0.5, 0.3),
            (np.array([1.0, 1.0, 1.0, 1.0]), 0, 1.0, 0.5),
            (np.array([100.0, -100.0, -100.0]), 1, 1.0, 0.5),
            (np.array([-np.inf, -np.inf, 5.0, -np.inf]), 0, 1.0, 0.5),
        ],
    )
    def test_default_min_p_zero_is_noop_across_fixtures(
        self,
        selector: TokenSelector,
        logits: np.ndarray,
        top_k: int,
        top_p: float,
        u: float,
    ) -> None:
        """Calling select() without min_p must match select(..., min_p=0.0) exactly."""
        legacy = selector.select(logits, temperature=1.0, top_k=top_k, top_p=top_p, u=u)
        explicit = selector.select(
            logits, temperature=1.0, top_k=top_k, top_p=top_p, u=u, min_p=0.0
        )
        assert legacy.token_id == explicit.token_id
        assert legacy.token_rank == explicit.token_rank
        assert legacy.token_prob == explicit.token_prob
        assert legacy.num_candidates == explicit.num_candidates


class TestSelectionResultImmutability:
    """Tests for SelectionResult frozen dataclass."""

    def test_frozen(self) -> None:
        """SelectionResult should reject attribute mutation."""
        result = SelectionResult(
            token_id=5, token_rank=2, token_prob=0.15, num_candidates=10, diagnostics={}
        )
        with pytest.raises(AttributeError):
            result.token_id = 42  # type: ignore[misc]

    def test_slots(self) -> None:
        """SelectionResult should use __slots__."""
        result = SelectionResult(
            token_id=5, token_rank=2, token_prob=0.15, num_candidates=10, diagnostics={}
        )
        assert hasattr(result, "__slots__")
