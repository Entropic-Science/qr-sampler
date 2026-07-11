"""Tests for the adapter's batched conversion, batched one-hot, and
parallel per-row sampling (perf tranche 2026-07).

Uses a deterministic constant-byte entropy source so serial and parallel
apply() runs are byte-for-byte comparable regardless of thread scheduling
(the amplified ``u`` depends only on the bytes, never on fetch order).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry
from tests.test_engines.test_vllm_adapter import (
    MockBatchUpdate,
    MockSamplingParams,
    _make_adapter,
)


@EntropySourceRegistry.register("const_bytes_test")
class _ConstantByteSource(EntropySource):
    """Deterministic source: every fetch returns the same byte value."""

    @property
    def name(self) -> str:
        return "const_bytes_test"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return b"\x90" * n

    def close(self) -> None:
        pass


def _apply_batch(parallel_rows: str, logits: np.ndarray) -> np.ndarray:
    """Build an adapter, add one request per row, run apply() once."""
    adapter = _make_adapter(
        vocab_size=logits.shape[1],
        entropy_source_type="const_bytes_test",
        apply_parallel_rows=parallel_rows,
    )
    try:
        added = [(i, MockSamplingParams(), None, None) for i in range(logits.shape[0])]
        adapter.update_state(MockBatchUpdate(added=added))
        out = logits.copy()
        adapter.apply(out)
        return out
    finally:
        adapter.close()


def _batch_logits(rows: int, vocab: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0.0, 2.0, size=(rows, vocab)).astype(np.float32)
    return logits


class TestParallelApplyEquivalence:
    """Parallel row sampling selects the same tokens as the serial loop."""

    def test_parallel_matches_serial(self) -> None:
        """Same deterministic entropy: serial and parallel agree per row."""
        logits = _batch_logits(rows=6, vocab=64)
        serial = _apply_batch("1", logits)
        parallel = _apply_batch("4", logits)
        assert np.array_equal(serial, parallel)

    def test_parallel_rows_are_onehot(self) -> None:
        """Every row ends up one-hot: one 0.0, rest -inf."""
        logits = _batch_logits(rows=5, vocab=32)
        out = _apply_batch("4", logits)
        for row in out:
            assert np.count_nonzero(row == 0.0) == 1
            assert np.all(np.isneginf(row[row != 0.0]))

    def test_parallel_state_bookkeeping(self) -> None:
        """tokens_generated advances once per row per apply() step."""
        adapter = _make_adapter(
            vocab_size=16,
            entropy_source_type="const_bytes_test",
            apply_parallel_rows="4",
        )
        try:
            added = [(i, MockSamplingParams(), None, None) for i in range(3)]
            adapter.update_state(MockBatchUpdate(added=added))
            logits = _batch_logits(rows=3, vocab=16)
            adapter.apply(logits.copy())
            adapter.apply(logits.copy())
            for i in range(3):
                assert adapter._request_states[i].tokens_generated == 2
        finally:
            adapter.close()

    def test_executor_created_lazily_and_closed(self) -> None:
        """The pool appears on the first multi-row batch and dies on close()."""
        adapter = _make_adapter(
            vocab_size=16,
            entropy_source_type="const_bytes_test",
            apply_parallel_rows="2",
        )
        try:
            assert adapter._row_executor is None
            adapter.apply(_batch_logits(rows=1, vocab=16))
            assert adapter._row_executor is None  # single row: no pool
            adapter.apply(_batch_logits(rows=3, vocab=16))
            assert adapter._row_executor is not None
        finally:
            adapter.close()
        assert adapter._row_executor is None

    def test_serial_config_never_creates_pool(self) -> None:
        """apply_parallel_rows=1 keeps the historical serial loop."""
        adapter = _make_adapter(
            vocab_size=16,
            entropy_source_type="const_bytes_test",
            apply_parallel_rows="1",
        )
        try:
            adapter.apply(_batch_logits(rows=4, vocab=16))
            assert adapter._row_executor is None
        finally:
            adapter.close()


def _apply_mixed_bypass_batch(
    parallel_rows: str, logits: np.ndarray, bypass_rows: set[int]
) -> np.ndarray:
    """Like ``_apply_batch`` but with ``bypass_rows`` opting into qr_bypass.

    Also asserts every request's step counter advanced exactly once — a
    compacted-index bug in the mixed-batch row mapping would advance the
    wrong states (every sampled state after the first bypass row shifts).
    """
    adapter = _make_adapter(
        vocab_size=logits.shape[1],
        entropy_source_type="const_bytes_test",
        apply_parallel_rows=parallel_rows,
    )
    try:
        added = [
            (
                i,
                MockSamplingParams(extra_args={"qr_bypass": True} if i in bypass_rows else None),
                None,
                None,
            )
            for i in range(logits.shape[0])
        ]
        adapter.update_state(MockBatchUpdate(added=added))
        out = logits.copy()
        adapter.apply(out)
        for i in range(logits.shape[0]):
            assert adapter._request_states[i].tokens_generated == 1
        return out
    finally:
        adapter.close()


class TestParallelMixedBypass:
    """Mixed bypass batches keep original row indices through the pool."""

    def test_parallel_mixed_bypass_matches_serial(self) -> None:
        """Sampled rows carry their ORIGINAL batch indices into the worker
        pool: serial and parallel mixed batches agree byte-for-byte, and
        bypass rows pass through bit-identically in both."""
        logits = _batch_logits(rows=6, vocab=64)
        bypass_rows = {1, 4}
        serial = _apply_mixed_bypass_batch("1", logits, bypass_rows)
        parallel = _apply_mixed_bypass_batch("4", logits, bypass_rows)
        assert np.array_equal(serial, parallel)
        for i in range(6):
            if i in bypass_rows:
                assert np.array_equal(serial[i], logits[i])
            else:
                assert np.count_nonzero(serial[i] == 0.0) == 1
                assert np.all(np.isneginf(serial[i][serial[i] != 0.0]))


class TestBatchedOnehot:
    """_force_onehot_batch equals the per-row one-hot loop."""

    def test_numpy_batch_matches_per_row(self) -> None:
        """numpy path: batch write equals the historical row-by-row writes."""
        adapter = _make_adapter(vocab_size=16, entropy_source_type="const_bytes_test")
        try:
            token_ids = [3, 0, 15, 7]
            batch = np.zeros((4, 16), dtype=np.float32)
            adapter._force_onehot_batch(batch, token_ids, is_numpy=True)

            reference = np.zeros((4, 16), dtype=np.float32)
            for i, token in enumerate(token_ids):
                adapter._force_onehot_row(reference, i, token, is_numpy=True)
            assert np.array_equal(batch, reference)
        finally:
            adapter.close()

    def test_torch_batch_matches_per_row(self) -> None:
        """torch path: fill_ + scatter_ equals template copy + scalar write."""
        torch = pytest.importorskip("torch")
        adapter = _make_adapter(vocab_size=16, entropy_source_type="const_bytes_test")
        try:
            token_ids = [5, 1, 9]
            batch = torch.zeros((3, 16), dtype=torch.float32)
            adapter._force_onehot_batch(batch, token_ids, is_numpy=False)

            reference = torch.zeros((3, 16), dtype=torch.float32)
            for i, token in enumerate(token_ids):
                adapter._force_onehot_row(reference, i, token, is_numpy=False)
            assert torch.equal(batch, reference)
        finally:
            adapter.close()

    def test_torch_apply_end_to_end(self) -> None:
        """apply() on a CPU torch tensor produces one-hot rows in place."""
        torch = pytest.importorskip("torch")
        adapter = _make_adapter(
            vocab_size=24,
            entropy_source_type="const_bytes_test",
            apply_parallel_rows="2",
        )
        try:
            gen = torch.Generator().manual_seed(17)
            logits = torch.randn((4, 24), generator=gen, dtype=torch.float32)
            returned = adapter.apply(logits)
            assert returned is logits
            for row in logits:
                assert int((row == 0.0).sum()) == 1
                assert bool(torch.isinf(row[row != 0.0]).all())
        finally:
            adapter.close()


class TestPinnedSlice:
    """Pinned staging buffer allocation logic."""

    def test_disabled_without_pin_memory(self) -> None:
        """is_pin_memory=False (the test default) never stages."""
        adapter = _make_adapter(vocab_size=8, entropy_source_type="const_bytes_test")
        try:
            assert adapter._pinned_slice((4, 8), None) is None
        finally:
            adapter.close()

    def test_pinned_buffer_grows_and_is_reused(self) -> None:
        """With pin memory: lazily allocated, grown, and reused by shape.

        On hosts without a pinned-memory allocator (CPU-only torch), the
        adapter must degrade gracefully: return ``None`` once and disable
        staging so the allocation is never re-attempted.
        """
        torch = pytest.importorskip("torch")
        try:
            torch.empty(1, pin_memory=True)
            pinned_available = True
        except RuntimeError:
            pinned_available = False

        adapter = _make_adapter(vocab_size=8, entropy_source_type="const_bytes_test")
        try:
            adapter._is_pin_memory = True
            first = adapter._pinned_slice((2, 8), torch.float32)
            if not pinned_available:
                assert first is None
                assert adapter._is_pin_memory is False  # staging disabled
                assert adapter._pinned_slice((2, 8), torch.float32) is None
                return
            assert first is not None and tuple(first.shape) == (2, 8)
            again = adapter._pinned_slice((2, 8), torch.float32)
            assert again.data_ptr() == first.data_ptr()  # reused
            bigger = adapter._pinned_slice((5, 8), torch.float32)
            assert tuple(bigger.shape) == (5, 8)
        finally:
            adapter.close()

    def test_1d_shape_skips_staging(self) -> None:
        """1-D tensors (test-only shape) never stage through the buffer."""
        adapter = _make_adapter(vocab_size=8, entropy_source_type="const_bytes_test")
        try:
            adapter._is_pin_memory = True
            assert adapter._pinned_slice((8,), None) is None
        finally:
            adapter.close()


class TestApplyParallelRowsConfig:
    """apply_parallel_rows config plumbing."""

    def test_env_ingestion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """QR_APPLY_PARALLEL_ROWS lands on the config field."""
        from qr_sampler.config import QRSamplerConfig

        monkeypatch.setenv("QR_APPLY_PARALLEL_ROWS", "3")
        assert QRSamplerConfig().apply_parallel_rows == 3

    def test_not_per_request(self) -> None:
        """The field is infrastructure: rejected in per-request extra_args."""
        from qr_sampler.config import validate_extra_args
        from qr_sampler.exceptions import ConfigValidationError

        with pytest.raises(ConfigValidationError):
            validate_extra_args({"qr_apply_parallel_rows": 4})

    def test_zero_resolves_to_cpu_count(self) -> None:
        """Default 0 resolves the worker cap to the machine's CPU count."""
        import os as _os

        adapter = _make_adapter(vocab_size=8, entropy_source_type="const_bytes_test")
        try:
            assert adapter._row_worker_cap == (_os.cpu_count() or 1)
        finally:
            adapter.close()


def test_constant_source_registered() -> None:
    """Sanity: the test-only source resolves through the registry."""
    cls: Any = EntropySourceRegistry.get("const_bytes_test")
    assert cls is _ConstantByteSource
