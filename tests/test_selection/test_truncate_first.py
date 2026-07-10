"""Tests for the ``truncate_first`` selector-order option (EVDT-TT order).

AGENTS.md invariant 15 pins the default selector order
``top-k -> softmax -> min-p -> top-p -> CDF``. The per-request flag
``qr_truncate_first`` (config ``truncate_first``) is the one explicit,
test-pinned exception: min-p is applied to the RAW (temperature-free)
distribution and temperature to the kept support afterwards.

The default ``truncate_first=False`` MUST be a strict no-op — byte-identical
to the pre-flag selector — mirroring the ``min_p=0.0`` no-op pin in
``test_min_p.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import resolve_config
from qr_sampler.config.model import PER_REQUEST_FIELDS, QRSamplerConfig
from qr_sampler.selection.selector import TokenSelector


@pytest.fixture()
def selector() -> TokenSelector:
    """Default TokenSelector."""
    return TokenSelector()


class TestDefaultOffStrictNoop:
    """truncate_first=False must be byte-identical to omitting the parameter."""

    @pytest.mark.parametrize(
        "logits",
        [
            np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            np.array([10.0, 0.0, -5.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([-np.inf, 5.0, 3.0, -np.inf]),
        ],
    )
    @pytest.mark.parametrize("temperature", [0.5, 1.0, 2.0])
    @pytest.mark.parametrize("min_p", [0.0, 0.1])
    @pytest.mark.parametrize("u", [0.001, 0.5, 0.999])
    def test_false_flag_is_noop(
        self,
        selector: TokenSelector,
        logits: np.ndarray,
        temperature: float,
        min_p: float,
        u: float,
    ) -> None:
        """Explicit False must reproduce the legacy select() outcome exactly."""
        without = selector.select(
            logits, temperature=temperature, top_k=0, top_p=1.0, u=u, min_p=min_p
        )
        with_false = selector.select(
            logits,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            u=u,
            min_p=min_p,
            truncate_first=False,
        )
        assert with_false.token_id == without.token_id
        assert with_false.token_rank == without.token_rank
        assert with_false.token_prob == without.token_prob
        assert with_false.num_candidates == without.num_candidates
        assert with_false.diagnostics == without.diagnostics

    def test_default_path_has_no_truncate_first_diagnostic(self, selector: TokenSelector) -> None:
        """The default path emits the exact pre-flag diagnostics dict."""
        result = selector.select(
            np.array([3.0, 2.0, 1.0]), temperature=1.0, top_k=0, top_p=1.0, u=0.5
        )
        assert "truncate_first" not in result.diagnostics


class TestTruncateFirstOrder:
    """The truncate-first order is observably different from the default."""

    def test_support_set_decided_before_temperature(self, selector: TokenSelector) -> None:
        """High T flattens the post-T distribution so the default order keeps
        more tokens; truncate-first keeps only the raw-distribution survivors.

        logits = [3, 1, 0], min_p = 0.3:
          raw probs ~ [0.844, 0.114, 0.042] -> only token 0 survives min-p.
          T=5 probs ~ [0.451, 0.302, 0.247] -> all three survive min-p.
        """
        logits = np.array([3.0, 1.0, 0.0])
        default = selector.select(logits, temperature=5.0, top_k=0, top_p=1.0, u=0.999, min_p=0.3)
        tf = selector.select(
            logits,
            temperature=5.0,
            top_k=0,
            top_p=1.0,
            u=0.999,
            min_p=0.3,
            truncate_first=True,
        )
        assert default.diagnostics["effective_min_p_candidates"] == 3
        assert tf.diagnostics["effective_min_p_candidates"] == 1
        # u near 1 selects the least-probable survivor: rank 2 vs the only token.
        assert default.token_id == 2
        assert tf.token_id == 0
        assert tf.diagnostics["truncate_first"] is True

    def test_matches_softmax_log_pkept_over_t_reference(self, selector: TokenSelector) -> None:
        """Pinned against the V6 reference math: softmax(log(p_kept) / T)."""
        logits = np.array([2.0, 1.5, 1.0, 0.0, -1.0, -2.0])
        temperature = 1.7
        min_p = 0.2

        # Reference: raw softmax -> min-p mask -> renormalise -> log/T -> softmax.
        raw = np.exp(logits - logits.max())
        raw = raw / raw.sum()
        mask = raw >= min_p * raw.max()
        p_kept = np.where(mask, raw, 0.0)
        p_kept = p_kept / p_kept.sum()
        with np.errstate(divide="ignore"):
            log_kept = np.log(p_kept)
        scaled = np.exp(log_kept / temperature - np.max(log_kept / temperature))
        scaled[~mask] = 0.0
        expected = scaled / scaled.sum()

        # Walk the CDF with several u values and check each selected token's
        # probability equals the reference distribution's value.
        for u in (0.05, 0.3, 0.6, 0.9, 0.999):
            result = selector.select(
                logits,
                temperature=temperature,
                top_k=0,
                top_p=1.0,
                u=u,
                min_p=min_p,
                truncate_first=True,
            )
            assert result.token_prob == pytest.approx(expected[result.token_id], rel=1e-9)

    def test_zero_min_p_truncate_first_matches_default_selection(
        self, selector: TokenSelector
    ) -> None:
        """With min_p=0 both orders reduce to plain temperature sampling."""
        logits = np.array([4.0, 2.0, 1.0, 0.5])
        for u in (0.01, 0.5, 0.99):
            default = selector.select(logits, temperature=1.3, top_k=0, top_p=1.0, u=u, min_p=0.0)
            tf = selector.select(
                logits,
                temperature=1.3,
                top_k=0,
                top_p=1.0,
                u=u,
                min_p=0.0,
                truncate_first=True,
            )
            assert tf.token_id == default.token_id
            assert tf.token_prob == pytest.approx(default.token_prob, rel=1e-12)

    def test_greedy_contract_preserved(self, selector: TokenSelector) -> None:
        """T <= 0 is greedy in both orders."""
        logits = np.array([1.0, 5.0, 3.0])
        result = selector.select(
            logits, temperature=0.0, top_k=0, top_p=1.0, u=0.5, min_p=0.1, truncate_first=True
        )
        assert result.token_id == 1
        assert result.diagnostics["greedy"] is True
        assert result.diagnostics["truncate_first"] is True

    def test_top_k_and_top_p_still_apply(self, selector: TokenSelector) -> None:
        """top-k precedes the raw softmax; top-p follows temperature."""
        logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = selector.select(
            logits,
            temperature=1.0,
            top_k=3,
            top_p=0.99,
            u=0.5,
            min_p=0.01,
            truncate_first=True,
        )
        assert result.diagnostics["effective_top_k"] == 3
        assert result.num_candidates <= 3


class TestTruncateFirstConfigWiring:
    """qr_truncate_first is a per-request key resolving onto the config."""

    def test_field_is_per_request(self) -> None:
        assert "truncate_first" in PER_REQUEST_FIELDS

    def test_default_is_false(self) -> None:
        config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.truncate_first is False

    def test_qr_truncate_first_resolves(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        resolved = resolve_config(defaults, {"qr_truncate_first": True})
        assert resolved.truncate_first is True
        # Defaults are never mutated.
        assert defaults.truncate_first is False
