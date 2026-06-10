"""Tests for iter-55: selector CDF fast path + per-stage perf telemetry.

The fast path's contract is EXACT equivalence with the original
full-sort CDF selection (escalating to the full sort whenever ``u`` is
not strictly covered by the head's nonzero cumulative mass), so the
core test compares the two implementations across distribution shapes
and u-draws.
"""

from __future__ import annotations

import numpy as np
import pytest

import qr_sampler.selection.selector as selector_module
from qr_sampler.entropy.status_file import read_perf_status, write_perf_status
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.selection.selector import TokenSelector


def _record(**overrides) -> TokenSamplingRecord:
    """Minimal valid record with iter-55 stage timings."""
    base = dict(
        timestamp_ns=0,
        entropy_fetch_ms=1.0,
        total_sampling_ms=10.0,
        entropy_source_used="system",
        entropy_is_fallback=False,
        sample_mean=127.5,
        z_score=0.0,
        u_value=0.5,
        temperature_strategy="fixed",
        shannon_entropy=5.0,
        temperature_used=1.0,
        token_id=1,
        token_rank=0,
        token_prob=0.5,
        num_candidates=10,
        config_hash="deadbeefdeadbeef",
        temperature_ms=2.0,
        amplify_ms=0.1,
        select_ms=3.0,
    )
    base.update(overrides)
    return TokenSamplingRecord(**base)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


class TestCdfFastPathEquivalence:
    """Fast path must select the identical token to the full sort."""

    U_DRAWS = (1e-12, 0.01, 0.25, 0.5, 0.75, 0.9, 0.99, 0.9999, 1.0 - 1e-12)

    def _compare_all_draws(self, probs: np.ndarray) -> None:
        original = selector_module._CDF_FAST_MIN_VOCAB
        for u in self.U_DRAWS:
            fast = TokenSelector._cdf_select(probs, u)
            selector_module._CDF_FAST_MIN_VOCAB = 10**12  # force full sort
            try:
                slow = TokenSelector._cdf_select(probs, u)
            finally:
                selector_module._CDF_FAST_MIN_VOCAB = original
            assert fast == slow, f"divergence at u={u}: fast={fast} slow={slow}"

    def test_peaked_distribution(self) -> None:
        """Typical LLM shape: mass concentrated in the head."""
        rng = np.random.default_rng(42)
        logits = rng.normal(size=50_000) * 4.0
        self._compare_all_draws(_softmax(logits))

    def test_flat_distribution_escalates(self) -> None:
        """Near-uniform probs: head mass is tiny, every draw escalates."""
        rng = np.random.default_rng(7)
        logits = rng.normal(size=50_000) * 0.01
        probs = _softmax(logits)
        # Sanity: the top-512 mass really is below the larger u draws.
        head_mass = np.sort(probs)[-512:].sum()
        assert head_mass < 0.5
        self._compare_all_draws(probs)

    def test_truncated_distribution(self) -> None:
        """min-p/top-k style: few nonzero candidates inside the head."""
        rng = np.random.default_rng(3)
        probs = np.zeros(50_000)
        live = rng.choice(50_000, size=97, replace=False)
        weights = rng.random(97)
        probs[live] = weights / weights.sum()
        self._compare_all_draws(probs)

    def test_small_vocab_skips_fast_path(self) -> None:
        """Below the threshold the original path runs unconditionally."""
        rng = np.random.default_rng(11)
        probs = _softmax(rng.normal(size=256))
        self._compare_all_draws(probs)

    def test_select_end_to_end_equivalence(self) -> None:
        """Full select() pipeline agrees fast-vs-slow (creative-preset shape)."""
        rng = np.random.default_rng(99)
        logits = (rng.normal(size=50_000) * 3.0).astype(np.float32)
        selector = TokenSelector()
        original = selector_module._CDF_FAST_MIN_VOCAB
        for u in self.U_DRAWS:
            fast = selector.select(logits, 1.35, 0, 1.0, u, min_p=0.025)
            selector_module._CDF_FAST_MIN_VOCAB = 10**12
            try:
                slow = selector.select(logits, 1.35, 0, 1.0, u, min_p=0.025)
            finally:
                selector_module._CDF_FAST_MIN_VOCAB = original
            assert fast.token_id == slow.token_id
            assert fast.token_rank == slow.token_rank
            assert fast.num_candidates == slow.num_candidates


class TestPerfStatusFile:
    def test_roundtrip(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "perf.json"))
        assert write_perf_status({"window_tokens": 5}) is True
        data = read_perf_status()
        assert data is not None
        assert data["window_tokens"] == 5
        assert "updated_at" in data

    def test_disabled_channel(self, monkeypatch) -> None:
        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", "")
        assert write_perf_status({"x": 1}) is False
        assert read_perf_status() is None


class TestPerfAggregator:
    def test_snapshot_shape_and_ratios(self) -> None:
        from qr_sampler.engines.vllm import _PerfAggregator

        agg = _PerfAggregator()
        agg.PUBLISH_EVERY_TOKENS = 10**9  # no publication during this test
        agg.note(_record(entropy_prefetch_hit=True, entropy_echo_verified=True), 0.5, 0.1)
        agg.note(_record(entropy_prefetch_hit=True, entropy_echo_verified=False), 0.5, 0.1)
        agg.note(_record(entropy_prefetch_hit=False), 0.5, 0.1)
        agg.note(_record(entropy_is_fallback=True), 0.5, 0.1)

        snap = agg.snapshot()
        assert snap["window_tokens"] == 4
        assert snap["tokens_total"] == 4
        assert snap["prefetch"]["hits"] == 2
        assert snap["prefetch"]["misses"] == 1
        assert snap["prefetch"]["hit_ratio"] == pytest.approx(2 / 3, abs=1e-3)
        assert snap["prefetch"]["echo_verified_ratio"] == pytest.approx(0.5)
        assert snap["fallback_tokens_total"] == 1
        stages = ("to_numpy", "temperature", "entropy_wait", "amplify", "select", "onehot", "total")
        for stage in stages:
            assert stage in snap["stage_ms"]
            assert snap["stage_ms"][stage]["avg"] > 0

    def test_publishes_to_status_file(self, tmp_path, monkeypatch) -> None:
        from qr_sampler.engines.vllm import _PerfAggregator

        monkeypatch.setenv("QR_SAMPLER_PERF_FILE", str(tmp_path / "perf.json"))
        agg = _PerfAggregator()
        agg.PUBLISH_EVERY_TOKENS = 2
        agg.PUBLISH_MIN_INTERVAL_S = 10**9  # only the token trigger fires
        agg.note(_record(), 0.5, 0.1)
        assert read_perf_status() is None  # not yet due
        agg.note(_record(), 0.5, 0.1)
        data = read_perf_status()
        assert data is not None
        assert data["window_tokens"] == 2
