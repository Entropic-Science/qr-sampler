"""Tests for the pipelined (commit-then-fetch) entropy path.

Covers the three layers end to end:

- ``SamplingPipeline.sample_token``: redeems the current ticket, fires the
  next prefetch immediately after selection, and threads verification
  diagnostics into the record.
- ``FallbackEntropySource``: ticket redemption gets identical failover
  semantics to the serial path.
- ``VLLMAdapter``: per-request ticket lifecycle (fire on add, thread
  through apply, cancel on removal).

The causal contract under test: the entropy request for token *N* is
constructed (and its commitment nonce derived) only AFTER token *N-1* is
selected — verified here by re-deriving the nonce chain from the audit
trail exactly the way an external auditor would.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import SamplingPipeline, derive_commit_nonce
from qr_sampler.core.types import PrefetchContext
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.registry import register_entropy_source
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

VOCAB = 64


class FakeTicket:
    """Minimal stand-in for ``PrefetchTicket``."""

    def __init__(self, payload: bytes, nonce: int) -> None:
        self.payload = payload
        self.nonce = nonce
        self.hit: bool | None = None
        self.wait_ms: float | None = None
        self.echo_verified: bool | None = None
        self.server_timestamp_ns: int | None = None
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class FakeAsyncSource(EntropySource):
    """Prefetch-capable in-memory source that records every interaction."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.tickets: list[FakeTicket] = []
        self._closed = False

    @property
    def name(self) -> str:
        return "fake_async"

    @property
    def is_available(self) -> bool:
        return not self._closed

    def get_random_bytes(self, n: int) -> bytes:
        self.events.append(("serial", n))
        return os.urandom(n)

    def prefetch(self, n: int, nonce: int | None = None) -> FakeTicket:
        self.events.append(("prefetch", n, nonce))
        ticket = FakeTicket(os.urandom(n), nonce or 0)
        self.tickets.append(ticket)
        return ticket

    def get_random_bytes_with_ticket(self, n: int, ticket: Any | None) -> bytes:
        if ticket is None:
            return self.get_random_bytes(n)
        self.events.append(("redeem", n, ticket.nonce))
        ticket.hit = True
        ticket.echo_verified = True
        ticket.server_timestamp_ns = 42
        return ticket.payload

    def close(self) -> None:
        self._closed = True


def _build_pipeline(source: EntropySource, **config_overrides: Any) -> SamplingPipeline:
    config = QRSamplerConfig(_env_file=None, sample_count=128, **config_overrides)  # type: ignore[call-arg]
    return SamplingPipeline(
        entropy_source=source,
        amplifier=AmplifierRegistry.build(config),
        strategy=TemperatureStrategyRegistry.build(config, VOCAB),
        selector=TokenSelector(),
        sampling_logger=SamplingLogger(config),
        config=config,
    )


def _logits() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=VOCAB).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline layer
# ---------------------------------------------------------------------------


class TestPipelinePrefetch:
    def test_redeems_ticket_then_fires_next_after_selection(self) -> None:
        """Order matters: redeem current, select, THEN fire next."""
        source = FakeAsyncSource()
        pipeline = _build_pipeline(source)
        salt = b"\x01" * 16

        first_nonce = derive_commit_nonce(salt, 0, -1)
        ticket = source.prefetch(128, first_nonce)
        ctx = PrefetchContext(salt=salt, step=0, ticket=ticket)

        result = pipeline.sample_token(_logits(), prefetch_ctx=ctx, build_onehot=False)

        kinds = [e[0] for e in source.events]
        assert kinds == ["prefetch", "redeem", "prefetch"]
        assert result.next_ticket is source.tickets[-1]
        # The next fetch's nonce commits to the token JUST selected.
        expected = derive_commit_nonce(salt, 1, result.token_id)
        assert result.next_ticket.nonce == expected
        # Verification diagnostics flow into the record.
        assert result.record.entropy_prefetch_hit is True
        assert result.record.entropy_echo_verified is True
        assert result.record.entropy_server_timestamp_ns == 42
        assert result.record.entropy_nonce == f"{first_nonce:016x}"
        # build_onehot=False skips the vocab-size array.
        assert result.one_hot is None

    def test_serial_path_unchanged_without_context(self) -> None:
        source = FakeAsyncSource()
        pipeline = _build_pipeline(source)

        result = pipeline.sample_token(_logits())

        assert [e[0] for e in source.events] == ["serial"]
        assert result.next_ticket is None
        assert result.record.entropy_prefetch_hit is None
        assert result.record.entropy_nonce is None
        # Default keeps the one-hot for engine-agnostic callers.
        assert result.one_hot is not None
        assert result.one_hot[result.token_id] == 0.0

    def test_config_switch_disables_next_prefetch(self) -> None:
        source = FakeAsyncSource()
        pipeline = _build_pipeline(source, entropy_prefetch=False)
        ctx = PrefetchContext(salt=b"\x02" * 16, step=0, ticket=None)

        result = pipeline.sample_token(_logits(), prefetch_ctx=ctx)

        assert [e[0] for e in source.events] == ["serial"]
        assert result.next_ticket is None

    def test_nonce_chain_audit(self) -> None:
        """An auditor can re-derive the whole commitment chain.

        Each fetch's nonce must equal H(salt, step, prev_token_id) — i.e.
        it could not have been constructed before the previous token was
        selected. This is the verifiable post-selection guarantee.
        """
        source = FakeAsyncSource()
        pipeline = _build_pipeline(source)
        salt = os.urandom(16)

        audit: list[tuple[int, int, int]] = []  # (step, prev_token, nonce)
        ticket = source.prefetch(128, derive_commit_nonce(salt, 0, -1))
        audit.append((0, -1, ticket.nonce))

        prev_token = -1
        for step in range(3):
            ctx = PrefetchContext(salt=salt, step=step, ticket=ticket)
            result = pipeline.sample_token(
                _logits(), prefetch_ctx=ctx, build_onehot=False
            )
            prev_token = result.token_id
            ticket = result.next_ticket
            audit.append((step + 1, prev_token, ticket.nonce))

        for step, prev_token_id, nonce in audit:
            assert nonce == derive_commit_nonce(salt, step, prev_token_id)

    def test_nonce_is_nonzero_63_bit(self) -> None:
        for step in range(50):
            nonce = derive_commit_nonce(b"salt", step, step * 31 - 1)
            assert 0 < nonce <= 0x7FFFFFFFFFFFFFFF


# ---------------------------------------------------------------------------
# Fallback layer
# ---------------------------------------------------------------------------


class _UnavailableOnRedeem(FakeAsyncSource):
    def get_random_bytes_with_ticket(self, n: int, ticket: Any | None) -> bytes:
        raise EntropyUnavailableError("primary down at redeem time")

    def get_random_bytes(self, n: int) -> bytes:
        raise EntropyUnavailableError("primary down")


class TestFallbackTicketPath:
    def test_ticket_redeem_failure_engages_fallback(self) -> None:
        from qr_sampler.entropy.system import SystemEntropySource

        primary = _UnavailableOnRedeem()
        wrapper = FallbackEntropySource(primary, SystemEntropySource())

        ticket = wrapper.prefetch(64, nonce=5)
        data = wrapper.get_random_bytes_with_ticket(64, ticket)

        assert len(data) == 64
        assert wrapper.fallback_count == 1
        assert wrapper.currently_degraded is True
        assert wrapper.last_source_used == "system"

    def test_prefetch_delegation_never_raises(self) -> None:
        from qr_sampler.entropy.system import SystemEntropySource

        class _ExplodingPrefetch(FakeAsyncSource):
            def prefetch(self, n: int, nonce: int | None = None) -> FakeTicket:
                raise RuntimeError("boom")

        wrapper = FallbackEntropySource(_ExplodingPrefetch(), SystemEntropySource())
        assert wrapper.prefetch(64, nonce=5) is None

    def test_system_primary_has_no_prefetch(self) -> None:
        from qr_sampler.entropy.system import SystemEntropySource

        wrapper = FallbackEntropySource(SystemEntropySource(), SystemEntropySource())
        assert wrapper.prefetch(64) is None
        # And the redeem path with None ticket is the plain serial fetch.
        assert len(wrapper.get_random_bytes_with_ticket(64, None)) == 64


# ---------------------------------------------------------------------------
# vLLM adapter layer
# ---------------------------------------------------------------------------

# Register once at import: the registry is module-global.
register_entropy_source("fake_async_test")(FakeAsyncSource)


@pytest.fixture()
def adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("QR_ENTROPY_SOURCE_TYPE", "fake_async_test")
    monkeypatch.setenv("QR_PREINIT_ENTROPY_SOURCES", "fake_async_test")
    monkeypatch.setenv("QR_SAMPLE_COUNT", "128")

    from qr_sampler.engines.vllm import VLLMAdapter

    adapter = VLLMAdapter(None, None, False)
    yield adapter
    adapter.close()


def _batch_add(req_idx: int, extra_args: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(
        removed=[],
        moved=[],
        added=[
            SimpleNamespace(
                req_index=req_idx,
                sampling_params=SimpleNamespace(extra_args=extra_args or {}),
            )
        ],
    )


class TestAdapterTicketLifecycle:
    def _primary(self, adapter: Any, req_idx: int) -> FakeAsyncSource:
        state = adapter._request_states[req_idx]
        return state.pipeline.entropy_source._primary  # type: ignore[no-any-return]

    def test_add_fires_first_prefetch_with_sentinel_commitment(
        self, adapter: Any
    ) -> None:
        adapter.update_state(_batch_add(0))
        state = adapter._request_states[0]
        assert state.entropy_ticket is not None
        primary = self._primary(adapter, 0)
        kind, n, nonce = primary.events[-1]
        assert kind == "prefetch"
        assert n == 128
        assert nonce == derive_commit_nonce(state.prefetch_salt, 0, -1)

    def test_apply_redeems_and_rearms(self, adapter: Any) -> None:
        adapter.update_state(_batch_add(0))
        primary = self._primary(adapter, 0)
        first_ticket = adapter._request_states[0].entropy_ticket

        logits = np.random.default_rng(3).normal(size=(1, VOCAB)).astype(np.float32)
        adapter.apply(logits)

        state = adapter._request_states[0]
        assert state.entropy_ticket is not None
        assert state.entropy_ticket is not first_ticket
        kinds = [e[0] for e in primary.events]
        assert kinds == ["prefetch", "redeem", "prefetch"]
        # The one-hot was forced directly on the engine array.
        row = logits[0]
        assert row[np.argmax(row)] == 0.0
        assert np.sum(np.isneginf(row)) == VOCAB - 1

    def test_removal_cancels_inflight_ticket(self, adapter: Any) -> None:
        adapter.update_state(_batch_add(0))
        ticket = adapter._request_states[0].entropy_ticket
        adapter.update_state(SimpleNamespace(removed=[0], moved=[], added=[]))
        assert ticket.cancelled is True
        assert 0 not in adapter._request_states

    def test_prefetch_opt_out_per_request(self, adapter: Any) -> None:
        adapter.update_state(_batch_add(0, {"qr_entropy_prefetch": False}))
        assert adapter._request_states[0].entropy_ticket is None
