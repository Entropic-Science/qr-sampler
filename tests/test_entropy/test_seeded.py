"""Tests for the counter-based deterministic PRNG source (seeded_prng).

The property under test is the determinism.md §5.3 contract: bytes are a
pure function of ``(seed, token index)``, independent of draw ORDER —
threads, batching, prefetch, and other requests cannot change what block
*k* contains.
"""

from __future__ import annotations

import threading

import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import build_entropy_source
from qr_sampler.entropy.seeded import SeededPrngSource
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError

_BLOCK = 256  # bytes per draw in these tests


class TestSeededPrngDeterminism:
    def test_same_seed_same_step_same_bytes(self) -> None:
        """Two independent instances at the same (seed, step) agree bitwise."""
        a = SeededPrngSource(seed=42)
        b = SeededPrngSource(seed=42)
        for _ in range(8):
            assert a.get_random_bytes(_BLOCK) == b.get_random_bytes(_BLOCK)

    def test_distinct_steps_distinct_blocks(self) -> None:
        """Consecutive blocks of one stream never repeat."""
        src = SeededPrngSource(seed=7)
        blocks = [src.get_random_bytes(_BLOCK) for _ in range(16)]
        assert len(set(blocks)) == len(blocks)

    def test_distinct_seeds_distinct_streams(self) -> None:
        """Different seeds give different block 0."""
        assert SeededPrngSource(seed=1).get_random_bytes(_BLOCK) != SeededPrngSource(
            seed=2
        ).get_random_bytes(_BLOCK)

    def test_initial_step_resume_matches_uninterrupted_stream(self) -> None:
        """Rebuilding at initial_step=k (the preemption re-add path) yields
        exactly the blocks an uninterrupted instance would have served."""
        full = SeededPrngSource(seed=99)
        expected = [full.get_random_bytes(_BLOCK) for _ in range(10)]

        resumed = SeededPrngSource(seed=99, initial_step=4)
        assert [resumed.get_random_bytes(_BLOCK) for _ in range(6)] == expected[4:]

    def test_block_content_independent_of_request_size_history(self) -> None:
        """Block k does not depend on how many bytes earlier calls drew —
        each call consumes exactly one counter block."""
        a = SeededPrngSource(seed=5)
        b = SeededPrngSource(seed=5)
        a.get_random_bytes(8)  # small draw
        b.get_random_bytes(10_000)  # large draw
        assert a.get_random_bytes(_BLOCK) == b.get_random_bytes(_BLOCK)

    def test_set_step_pins_the_next_block(self) -> None:
        """set_step re-syncs the counter (the adapter's per-draw position
        sync); re-drawing a pinned step reproduces the block exactly —
        the property that makes vLLM's sampled-then-discarded
        partial-prefill rows harmless."""
        src = SeededPrngSource(seed=11)
        block0 = src.get_random_bytes(_BLOCK)  # consumed step 0
        src.set_step(0)
        assert src.get_random_bytes(_BLOCK) == block0  # re-drawn, identical
        src.set_step(5)
        assert src.next_step == 5
        assert src.get_random_bytes(_BLOCK) == SeededPrngSource(
            seed=11, initial_step=5
        ).get_random_bytes(_BLOCK)

    def test_thread_hammer_order_free_block_set(self) -> None:
        """N threads racing on ONE instance produce exactly the block set
        for steps 0..N-1 — order-free determinism, the property that makes
        the adapter's row thread pool safe to keep enabled."""
        n_threads = 16
        src = SeededPrngSource(seed=1234)
        results: list[bytes] = []
        lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def draw() -> None:
            barrier.wait()
            block = src.get_random_bytes(_BLOCK)
            with lock:
                results.append(block)

        threads = [threading.Thread(target=draw) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = {
            SeededPrngSource(seed=1234, initial_step=k).get_random_bytes(_BLOCK)
            for k in range(n_threads)
        }
        assert set(results) == expected
        assert src.next_step == n_threads


class TestSeededPrngInterface:
    def test_identity_properties(self) -> None:
        src = SeededPrngSource(seed=3, initial_step=2)
        assert src.name == "seeded_prng"
        assert src.is_available is True
        assert src.seed == 3
        assert src.next_step == 2

    def test_exact_byte_count(self) -> None:
        src = SeededPrngSource(seed=0)
        for n in (1, 7, 256, 10_000):
            assert len(src.get_random_bytes(n)) == n

    def test_no_prefetch_support(self) -> None:
        """ABC defaults deliberately kept: the serial path is always taken."""
        src = SeededPrngSource(seed=1)
        assert src.prefetch(100) is None
        assert src.prefetch_draw(100, "") is None
        with pytest.raises(EntropyUnavailableError):
            src.get_draw(0, "")

    def test_close_is_noop(self) -> None:
        SeededPrngSource(seed=1).close()


class TestSeededPrngFactoryRejection:
    def test_build_entropy_source_rejects_seeded_prng(self) -> None:
        """The factory must never build a process-level (shared-stream)
        instance — construction is the engine adapter's job, per request."""
        config = QRSamplerConfig(
            entropy_source_type="seeded_prng",
            seed=42,
            signal_amplifier_type="zscore_mean",
            zscore_calibration_samples=0,
            entropy_prefetch=False,
        )
        with pytest.raises(ConfigValidationError, match="per-request"):
            build_entropy_source(config)

    def test_instance_alias_cannot_evade_rejection(self) -> None:
        """A named instance whose type is seeded_prng is rejected at config
        construction (the _validate_entropy_source_instances rule names
        QR_ENTROPY_SOURCE_INSTANCES), so no process-level seeded source
        can ever be declared, let alone built."""
        with pytest.raises(ConfigValidationError, match="seeded_prng"):
            QRSamplerConfig(
                entropy_source_type="prng_lane",
                entropy_source_instances={"prng_lane": {"type": "seeded_prng"}},
            )
