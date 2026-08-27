"""Counter-based deterministic PRNG entropy source (per-request).

The deterministic lane's entropy layer (``docs/determinism.md`` §5.3/§6 T1):
random bytes must be a pure function of ``(request seed, token index)`` so
that batching, row-thread scheduling, prefetch timing, and concurrent
requests cannot change which bytes a given token receives. A seeded PRNG
consumed as a SHARED stream cannot provide that — stream position depends
on global consumption order — so this source is constructed PER REQUEST by
the engine adapter, never by the process-level factory
(``build_entropy_source`` rejects it; see ``core/pipeline.py``).
"""

from __future__ import annotations

import threading

import numpy as np

from qr_sampler.entropy.base import EntropySource


class SeededPrngSource(EntropySource):
    """Counter-based deterministic PRNG source (uniform bytes).

    Block *k* is ``Philox(key=seed, counter=[0, 0, 0, k])`` — a pure
    function of ``(seed, k)``, independent of draw order across requests,
    threads, and prefetch. Instances are PER-REQUEST (one per engine
    ``_RequestState``); the internal draw index makes consecutive
    ``get_random_bytes()`` calls consume consecutive counter blocks, and
    starts at ``initial_step`` so an engine preemption re-add (which
    rebuilds per-request state mid-generation) resumes at the correct
    block: the adapter passes ``initial_step=len(output_ids)``.

    Uniform bytes match the z-score amplifier's default population
    parameters (``population_mean=127.5``, ``population_std=255/sqrt(12)``)
    — the same assumption the QRNG lanes make, so the deterministic lane
    is a like-for-like control arm.

    The pipelined hooks (``prefetch``/``get_draw``) deliberately keep the
    ABC defaults: the source is local and infallible, so the serial fetch
    path is always taken and the fallback/breaker machinery never engages.

    Args:
        seed: Non-negative request seed; the Philox key. Identical seeds
            yield identical block sequences.
        initial_step: First counter block to serve (0-based token index).
    """

    def __init__(self, seed: int, initial_step: int = 0) -> None:
        self._seed = seed
        self._step = initial_step
        # The engine adapter samples batch rows on worker threads (ABC
        # thread-safety contract); the lock makes step reservation atomic.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        """Return ``'seeded_prng'``."""
        return "seeded_prng"

    @property
    def is_available(self) -> bool:
        """Always returns ``True`` — local, infallible."""
        return True

    @property
    def seed(self) -> int:
        """The request seed keying this instance's block sequence."""
        return self._seed

    @property
    def next_step(self) -> int:
        """The counter block the next ``get_random_bytes()`` call will use."""
        return self._step

    def get_random_bytes(self, n: int) -> bytes:
        """Return *n* uniform bytes from the next counter block.

        Each call consumes exactly one counter block regardless of *n*:
        a fresh ``Philox(key=seed, counter=[0, 0, 0, step])`` generator is
        built per call, so the bytes are a pure function of
        ``(seed, step, n)`` with no state carried between blocks. The step
        rides the counter's high word, leaving 2^192 draws of headroom
        inside each block — no overlap between blocks by construction.

        Args:
            n: Number of random bytes to generate.

        Returns:
            Exactly *n* bytes.
        """
        with self._lock:
            step = self._step
            self._step += 1
        gen = np.random.Generator(np.random.Philox(key=self._seed, counter=[0, 0, 0, step]))
        return gen.bytes(n)

    def close(self) -> None:
        """No-op — no resources to release."""
