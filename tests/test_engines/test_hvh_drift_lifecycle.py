"""Lifecycle tests for the HVH-Drift strategy inside VLLMAdapter.

Verifies the engine-side guarantees that the source code in
``engines/vllm.py`` already satisfies (and which spec §2.1.1 calls out
explicitly) without touching the adapter itself:

* Each request in the batch gets its own ``HVHDriftStrategy`` instance,
  so the per-sequence EMA state cannot leak across sequences.
* ``update_state(removed=...)`` drops the adapter's only reference to a
  request's strategy, satisfying the NFR-3 / NFR-8a memory bound.
* A request body carrying ``qr_preset=creative_sampling`` is resolved
  through ``resolve_config`` / preset expansion so the resulting
  ``_RequestState.strategy`` is an ``HVHDriftStrategy``.

Uses ``MockUniformSource`` as the entropy source (no gRPC, no GPU). The
``vllm_config=None`` DI escape hatch in ``VLLMAdapter`` is exercised
indirectly via the existing ``_make_adapter`` helper pattern.
"""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

from qr_sampler.engines.vllm.adapter import VLLMAdapter, _RequestState
from qr_sampler.temperature.hvh_drift import HVHDriftStrategy

# ---------------------------------------------------------------------------
# Minimal vLLM-shaped mock objects (mirroring test_vllm_adapter.py)
# ---------------------------------------------------------------------------


@dataclass
class _MockVllmConfig:
    vocab_size: int = 16


@dataclass
class _MockSamplingParams:
    extra_args: dict[str, Any] | None = None


@dataclass
class _MockAddedRequest:
    req_index: int
    sampling_params: _MockSamplingParams | None = None


@dataclass
class _MockBatchUpdate:
    removed: list[int] | None = None
    moved: list[Any] | None = None
    added: list[_MockAddedRequest] | None = None

    def __post_init__(self) -> None:
        if self.removed is None:
            self.removed = []
        if self.moved is None:
            self.moved = []
        if self.added is None:
            self.added = []


def _make_adapter(vocab_size: int = 16) -> VLLMAdapter:
    """Build a VLLMAdapter wired to MockUniformSource.

    The adapter default uses ``temperature_strategy='fixed'`` (config
    default), so requests configured with ``qr_preset=creative_sampling``
    will resolve to a *different* config and thus get a fresh
    HVHDriftStrategy per request -- exactly the cross-leak scenario we
    want to test.
    """
    env = {
        "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
        "QR_FALLBACK_MODE": "error",
        "QR_LOG_LEVEL": "none",
    }
    old: dict[str, str | None] = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        return VLLMAdapter(vllm_config=_MockVllmConfig(vocab_size=vocab_size))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _add_creative_request(adapter: VLLMAdapter, req_index: int) -> None:
    """Inject a request configured with the creative_sampling preset."""
    params = _MockSamplingParams(extra_args={"qr_preset": "creative_sampling"})
    adapter.update_state(
        _MockBatchUpdate(added=[_MockAddedRequest(req_index=req_index, sampling_params=params)])
    )


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


def test_preset_resolves_through_resolve_config() -> None:
    """A request body with ``qr_preset=creative_sampling`` produces a
    ``_RequestState`` whose strategy is an ``HVHDriftStrategy``.
    """
    adapter = _make_adapter()
    try:
        _add_creative_request(adapter, req_index=0)

        state = adapter._request_states[0]
        assert isinstance(state, _RequestState)
        assert isinstance(state.strategy, HVHDriftStrategy)
        # Preset expansion must have flipped the strategy field on the
        # resolved per-request config (sanity check on the FR-10 path).
        assert state.config.temperature_strategy == "hvh_drift"
    finally:
        adapter.close()


def test_state_is_per_request() -> None:
    """Two requests carrying the same preset evolve independent EMA state.

    Drives heterogeneous logits per batch row through ``apply()`` and
    asserts each request's ``H_ema`` diverges (no cross-leak).
    """
    adapter = _make_adapter(vocab_size=16)
    try:
        _add_creative_request(adapter, req_index=0)
        _add_creative_request(adapter, req_index=1)

        strat_a = adapter._request_states[0].strategy
        strat_b = adapter._request_states[1].strategy
        assert isinstance(strat_a, HVHDriftStrategy)
        assert isinstance(strat_b, HVHDriftStrategy)
        # Fresh instances -- not the same Python object, not the
        # pipeline's default strategy.
        assert strat_a is not strat_b
        assert strat_a is not adapter.get_pipeline().strategy
        assert strat_b is not adapter.get_pipeline().strategy

        # Row 0: peaked logits (low entropy).
        # Row 1: roughly flat logits (high entropy).
        peaked = np.array(
            [
                10.0,
                -1.0,
                -2.0,
                -3.0,
                -4.0,
                -5.0,
                -6.0,
                -7.0,
                -8.0,
                -9.0,
                -10.0,
                -11.0,
                -12.0,
                -13.0,
                -14.0,
                -15.0,
            ],
            dtype=np.float32,
        )
        flat = np.linspace(0.5, -0.5, 16, dtype=np.float32)

        for _ in range(8):
            # apply() mutates logits in-place, so rebuild each step.
            batch = np.stack([peaked.copy(), flat.copy()])
            adapter.apply(batch)

        # The per-request strategies live on the adapter's request map.
        # If state had leaked, both EMAs would have absorbed the same
        # mixed entropy stream; instead each one tracks only its own row.
        assert strat_a.H_ema != strat_b.H_ema, (
            f"H_ema state leaked across requests: a={strat_a.H_ema!r}, b={strat_b.H_ema!r}"
        )
        # The low-entropy row should have lower H_ema than the high-entropy row.
        assert strat_a.H_ema < strat_b.H_ema, (
            f"Expected peaked-row H_ema < flat-row H_ema, "
            f"got a={strat_a.H_ema:.4f}, b={strat_b.H_ema:.4f}"
        )
    finally:
        adapter.close()


def test_no_state_leak_when_default_strategy_is_hvh_drift() -> None:
    """Two requests with empty extra_args must STILL get independent strategies
    when the *default config* itself selects ``hvh_drift``.

    Regression guard: ``update_state`` previously reused
    ``self._pipeline.strategy`` whenever ``req_config is self._default_config``
    (the short-circuit path of ``resolve_config`` returns the defaults
    object identity when ``extra_args`` is empty). For a stateless ``fixed``
    or ``edt`` strategy that is harmless, but for ``hvh_drift`` it would
    mean every request sharing the SAME EMA state — a covert channel
    between concurrent users that violates CLAUDE.md invariant 17.
    """
    env = {
        "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
        "QR_FALLBACK_MODE": "error",
        "QR_LOG_LEVEL": "none",
        # Drive the pipeline default to hvh_drift via env var, NOT via preset.
        # This is the scenario the reviewer flagged as unsafe.
        "QR_TEMPERATURE_STRATEGY": "hvh_drift",
    }
    old: dict[str, str | None] = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        adapter = VLLMAdapter(vllm_config=_MockVllmConfig(vocab_size=16))
        try:
            # Both requests have EMPTY extra_args -- so resolve_config's
            # short-circuit returns the defaults object by identity.
            adapter.update_state(
                _MockBatchUpdate(
                    added=[
                        _MockAddedRequest(req_index=0, sampling_params=_MockSamplingParams()),
                        _MockAddedRequest(req_index=1, sampling_params=_MockSamplingParams()),
                    ],
                ),
            )

            strat_a = adapter._request_states[0].strategy
            strat_b = adapter._request_states[1].strategy
            assert isinstance(strat_a, HVHDriftStrategy)
            assert isinstance(strat_b, HVHDriftStrategy)
            # Each request must hold its own fresh strategy instance.
            assert strat_a is not strat_b, (
                "Default-config request reused the pipeline's shared strategy: "
                "EMA state would leak between concurrent users."
            )
            assert strat_a is not adapter.get_pipeline().strategy
            assert strat_b is not adapter.get_pipeline().strategy
        finally:
            adapter.close()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_state_evicted_on_remove() -> None:
    """``update_state(removed=...)`` drops the adapter's reference to the
    request's strategy (NFR-3 memory bound / NFR-8a).
    """
    adapter = _make_adapter()
    try:
        _add_creative_request(adapter, req_index=0)
        strategy = adapter._request_states[0].strategy
        # _RequestState has __slots__ without __weakref__, so it cannot
        # be weakref'd directly. HVHDriftStrategy can be, which is the
        # only object that carries non-trivial per-request state.
        strategy_ref = weakref.ref(strategy)

        # Drive a few tokens so the strategy holds non-trivial state.
        for _ in range(3):
            adapter.apply(np.zeros((1, adapter._vocab_size), dtype=np.float32))

        # Issue the removal.
        adapter.update_state(_MockBatchUpdate(removed=[0]))

        # The state map no longer contains entry 0.
        assert 0 not in adapter._request_states

        # Drop our local reference and run gc to confirm the adapter
        # was the last holder of the strategy.
        del strategy
        import gc

        gc.collect()
        assert strategy_ref() is None, (
            "HVHDriftStrategy still reachable after removal: adapter is leaking strategy"
        )
    finally:
        adapter.close()
