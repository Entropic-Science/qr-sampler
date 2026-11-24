"""Entropy conditioning: selection, wire metadata, echo check, records.

The transform runs on the entropy server. These tests pin what this
package owns: validating the selection, asking for it in gRPC metadata,
refusing bytes the server did not confirm it conditioned, and writing
what was applied onto the per-token record.

The end-to-end cases run the real ``QuantumGrpcSource`` against a live
localhost ``EntropyService`` stub that implements the server contract
(read the selection from invocation metadata, hash block by block, keep
the head of each digest, echo what it applied in initial metadata).
"""

from __future__ import annotations

import hashlib
from concurrent import futures
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
import pytest

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config
from qr_sampler.config.conditioning import (
    APPLIED_HEADER,
    RAW,
    Conditioning,
    conditioning_of,
    conditioning_problem,
)
from qr_sampler.core.pipeline import SamplingPipeline, conditioning_kwargs
from qr_sampler.core.types import PrefetchContext
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.entropy.qgrpc import QuantumGrpcSource
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.token_log import sober_row
from qr_sampler.proto.entropy_service_pb2 import EntropyRequest, EntropyResponse
from qr_sampler.proto.entropy_service_pb2_grpc import (
    EntropyServiceServicer,
    add_EntropyServiceServicer_to_server,
)
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

VOCAB = 64
HASHED = Conditioning("hash_truncate", "sha256", 128, 16)


def _config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


# ── selection ────────────────────────────────────────────────────────────


class TestSelection:
    def test_raw_is_default_and_sends_nothing(self) -> None:
        config = _config()
        assert config.conditioning_mode == "raw"
        assert RAW.metadata() == ()
        assert RAW.canonical() == "raw"
        assert RAW.raw_bytes_needed(10_000) == 10_000

    def test_hash_truncate_metadata_and_cost(self) -> None:
        assert HASHED.canonical() == "hash_truncate;sha256;128;16"
        assert HASHED.metadata() == (
            ("qr-conditioning", "hash_truncate"),
            ("qr-conditioning-hash", "sha256"),
            ("qr-conditioning-in-bytes", "128"),
            ("qr-conditioning-keep-bytes", "16"),
        )
        assert HASHED.raw_bytes_needed(10_000) == 80_000

    def test_raw_drops_unused_parameters(self) -> None:
        assert conditioning_of("raw", "sha512", 999, 7) is RAW

    @pytest.mark.parametrize(
        ("args", "match"),
        [
            (("nope", "sha256", 128, 16), "conditioning_mode"),
            (("hash_truncate", "md5", 128, 16), "conditioning_hash"),
            (("hash_truncate", "sha256", 128, 17), "at least half"),
            (("hash_truncate", "sha512", 128, 33), "at least half"),
            (("hash_truncate", "sha256", 31, 16), "at least 2"),
        ],
    )
    def test_problems(self, args: tuple[str, str, int, int], match: str) -> None:
        problem = conditioning_problem(*args)
        assert problem is not None and match in problem

    def test_raw_ignores_parameter_values(self) -> None:
        assert conditioning_problem("raw", "anything", 1, 99) is None


class TestConfigValidation:
    def test_valid_selection_on_a_grpc_source(self) -> None:
        config = _config(entropy_source_type="quantum_grpc", conditioning_mode="hash_truncate")
        assert config.conditioning_keep_bytes == 16

    def test_per_request_keys_resolve(self) -> None:
        defaults = _config(entropy_source_type="quantum_grpc")
        config = resolve_config(
            defaults,
            {
                "qr_conditioning_mode": "hash_truncate",
                "qr_conditioning_hash": "sha512",
                "qr_conditioning_in_bytes": 256,
                "qr_conditioning_keep_bytes": 32,
            },
        )
        assert conditioning_kwargs(config, _Supports())["conditioning"] == Conditioning(
            "hash_truncate", "sha512", 256, 32
        )

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"conditioning_mode": "sha256"}, "conditioning_mode"),
            ({"conditioning_mode": "hash_truncate", "conditioning_keep_bytes": 20}, "half"),
            # The local sources are not an entropy server.
            ({"conditioning_mode": "hash_truncate", "entropy_source_type": "system"}, "gRPC"),
            # Server draws integrate raw blocks.
            ({"conditioning_mode": "hash_truncate", "signal_amplifier_type": "server"}, "byte"),
            # Calibration reads raw blocks; uniform bytes need none.
            (
                {"conditioning_mode": "hash_truncate", "zscore_calibration_samples": 10},
                "calibration",
            ),
        ],
    )
    def test_rejections(self, overrides: dict[str, Any], match: str) -> None:
        base = {"entropy_source_type": "quantum_grpc"}
        with pytest.raises(ConfigValidationError, match=match):
            _config(**{**base, **overrides})

    def test_changes_the_config_hash(self) -> None:
        from qr_sampler.core.pipeline import config_hash

        raw = _config(entropy_source_type="quantum_grpc")
        hashed = _config(entropy_source_type="quantum_grpc", conditioning_mode="hash_truncate")
        assert config_hash(raw) != config_hash(hashed)


class _Supports:
    name = "stub"
    supports_conditioning = True


class _DoesNot:
    name = "third-party"
    supports_conditioning = False


class TestKwargs:
    def test_raw_passes_nothing_to_any_source(self) -> None:
        assert conditioning_kwargs(_config(), _DoesNot()) == {}  # type: ignore[arg-type]

    def test_unsupporting_source_is_unavailable_not_silently_raw(self) -> None:
        config = _config(entropy_source_type="quantum_grpc", conditioning_mode="hash_truncate")
        with pytest.raises(EntropyUnavailableError, match="cannot apply conditioning"):
            conditioning_kwargs(config, _DoesNot())  # type: ignore[arg-type]


# ── a stub entropy server implementing the conditioning contract ────────


class ConditioningServicer(EntropyServiceServicer):
    """Serves a known raw stream; honors (or ignores) the conditioning headers."""

    def __init__(self, *, honor: bool = True, echo: str | None = None) -> None:
        self.honor = honor
        self.echo = echo  # override the echoed string (a lying server)
        self.calls: list[dict[str, str]] = []

    @staticmethod
    def raw(n: int) -> bytes:
        return bytes(i % 251 for i in range(n))

    def _serve(self, request: EntropyRequest, context: Any, echo: bool = True) -> EntropyResponse:
        metadata = {k: v for k, v in context.invocation_metadata()}
        self.calls.append(metadata)
        n = request.bytes_needed
        if self.honor and metadata.get("qr-conditioning") == "hash_truncate":
            name = metadata["qr-conditioning-hash"]
            in_bytes = int(metadata["qr-conditioning-in-bytes"])
            keep = int(metadata["qr-conditioning-keep-bytes"])
            blocks = -(-n // keep)
            raw = self.raw(blocks * in_bytes)
            data = b"".join(
                hashlib.new(name, raw[i * in_bytes : (i + 1) * in_bytes]).digest()[:keep]
                for i in range(blocks)
            )[:n]
            if echo:
                applied = self.echo or f"hash_truncate;{name};{in_bytes};{keep}"
                context.send_initial_metadata(((APPLIED_HEADER, applied),))
        else:
            data = self.raw(n)
        return EntropyResponse(data=data, sequence_id=request.sequence_id)

    def GetEntropy(self, request: EntropyRequest, context: Any) -> EntropyResponse:  # noqa: N802
        return self._serve(request, context)

    def StreamEntropy(self, request_iterator: Any, context: Any) -> Any:  # noqa: N802
        # Initial metadata goes out once per stream, before the first reply.
        for index, request in enumerate(request_iterator):
            yield self._serve(request, context, echo=index == 0)


def _serve(servicer: ConditioningServicer) -> Iterator[tuple[str, ConditioningServicer]]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_EntropyServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}", servicer
    finally:
        server.stop(grace=None)


@pytest.fixture()
def service() -> Iterator[tuple[str, ConditioningServicer]]:
    yield from _serve(ConditioningServicer())


@pytest.fixture()
def old_service() -> Iterator[tuple[str, ConditioningServicer]]:
    """A server that predates conditioning: ignores the headers, no echo."""
    yield from _serve(ConditioningServicer(honor=False))


@pytest.fixture()
def lying_service() -> Iterator[tuple[str, ConditioningServicer]]:
    yield from _serve(ConditioningServicer(echo="hash_truncate;sha256;64;16"))


def _source(address: str, **overrides: Any) -> QuantumGrpcSource:
    return QuantumGrpcSource(
        _config(
            entropy_source_type="quantum_grpc",
            grpc_server_address=address,
            grpc_api_key="k",
            grpc_retry_count=0,
            **overrides,
        )
    )


def _expected(n: int, c: Conditioning = HASHED) -> bytes:
    blocks = -(-n // c.keep_bytes)
    raw = ConditioningServicer.raw(blocks * c.in_bytes)
    return b"".join(
        hashlib.new(c.hash, raw[i * c.in_bytes : (i + 1) * c.in_bytes]).digest()[: c.keep_bytes]
        for i in range(blocks)
    )[:n]


class TestWire:
    def test_raw_fetch_sends_no_conditioning_metadata(
        self, service: tuple[str, ConditioningServicer]
    ) -> None:
        address, servicer = service
        source = _source(address)
        try:
            assert source.get_random_bytes(100) == ConditioningServicer.raw(100)
        finally:
            source.close()
        assert not any(k.startswith("qr-conditioning") for k in servicer.calls[-1])
        assert servicer.calls[-1]["api-key"] == "k"

    @pytest.mark.parametrize("mode", ["unary", "server_streaming", "bidi_streaming"])
    def test_conditioned_fetch_in_every_transport_mode(
        self, service: tuple[str, ConditioningServicer], mode: str
    ) -> None:
        address, servicer = service
        source = _source(address, grpc_mode=mode)
        try:
            for _ in range(3):  # the bidi stream is reused; the echo is checked once
                assert source.get_random_bytes(100, HASHED) == _expected(100)
            # The same source still serves raw alongside.
            assert source.get_random_bytes(40) == ConditioningServicer.raw(40)
        finally:
            source.close()
        hashed_calls = [c for c in servicer.calls if "qr-conditioning" in c]
        assert len(hashed_calls) == 3
        assert hashed_calls[0]["qr-conditioning-keep-bytes"] == "16"

    @pytest.mark.parametrize("mode", ["unary", "server_streaming", "bidi_streaming"])
    def test_unconfirmed_bytes_are_refused(
        self, old_service: tuple[str, ConditioningServicer], mode: str
    ) -> None:
        """An old server answers with RAW bytes: they must never be used as hashed."""
        address, _ = old_service
        source = _source(address, grpc_mode=mode)
        try:
            with pytest.raises(EntropyUnavailableError, match="did not apply conditioning"):
                source.get_random_bytes(100, HASHED)
            assert source.get_random_bytes(10) == ConditioningServicer.raw(10)
        finally:
            source.close()

    def test_mismatched_echo_is_refused(
        self, lying_service: tuple[str, ConditioningServicer]
    ) -> None:
        address, _ = lying_service
        source = _source(address)
        try:
            with pytest.raises(EntropyUnavailableError, match="hash_truncate;sha256;64;16"):
                source.get_random_bytes(100, HASHED)
        finally:
            source.close()

    def test_prefetch_carries_conditioning(self, service: tuple[str, ConditioningServicer]) -> None:
        address, _ = service
        source = _source(address)
        try:
            ticket = source.prefetch(64, 12345, HASHED)
            assert ticket is not None
            assert source.get_random_bytes_with_ticket(64, ticket, HASHED) == _expected(64)
            assert ticket.echo_verified is True
        finally:
            source.close()

    def test_failed_prefetch_refetches_conditioned(
        self, old_service: tuple[str, ConditioningServicer]
    ) -> None:
        """The serial retry after a bad ticket keeps asking for hashed bytes."""
        address, servicer = old_service
        source = _source(address)
        try:
            ticket = source.prefetch(64, 7, HASHED)
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes_with_ticket(64, ticket, HASHED)
        finally:
            source.close()
        assert all("qr-conditioning" in c for c in servicer.calls)


# ── pipeline + records ───────────────────────────────────────────────────


def _pipeline(source: Any, config: QRSamplerConfig) -> SamplingPipeline:
    return SamplingPipeline(
        entropy_source=source,
        amplifier=AmplifierRegistry.build(config),
        strategy=TemperatureStrategyRegistry.build(config, VOCAB),
        selector=TokenSelector(),
        sampling_logger=SamplingLogger(config),
        config=config,
    )


def _logits() -> np.ndarray:
    return np.random.default_rng(17).normal(size=VOCAB).astype(np.float32)


def _hashed_config(address: str, **overrides: Any) -> QRSamplerConfig:
    return _config(
        entropy_source_type="quantum_grpc",
        grpc_server_address=address,
        grpc_retry_count=0,
        conditioning_mode="hash_truncate",
        sample_count=1000,
        **overrides,
    )


class TestPipeline:
    def test_record_and_token_log_carry_the_applied_conditioning(
        self, service: tuple[str, ConditioningServicer]
    ) -> None:
        address, servicer = service
        config = _hashed_config(address)
        source = QuantumGrpcSource(config)
        try:
            pipeline = _pipeline(source, config)
            ctx = PrefetchContext(salt=b"s" * 16)
            first = pipeline.sample_token(_logits(), prefetch_ctx=ctx)
            ctx.step, ctx.ticket = 1, first.next_ticket
            second = pipeline.sample_token(_logits(), prefetch_ctx=ctx)
        finally:
            source.close()
        for result in (first, second):
            assert result.record.conditioning == "hash_truncate;sha256;128;16"
            assert result.record.entropy_is_fallback is False
        assert second.record.entropy_prefetch_hit is True
        assert sober_row(0, first.record)["conditioning"] == "hash_truncate;sha256;128;16"
        # Every fetch of the request, prefetched ones included, asked for it.
        assert all(c.get("qr-conditioning") == "hash_truncate" for c in servicer.calls)
        # The amplifier saw exactly the server's conditioned bytes.
        expected_mean = float(np.frombuffer(_expected(1000), np.uint8).mean())
        assert first.record.sample_mean == pytest.approx(expected_mean)

    def test_raw_request_record_is_unchanged(
        self, service: tuple[str, ConditioningServicer]
    ) -> None:
        address, _ = service
        config = _config(
            entropy_source_type="quantum_grpc", grpc_server_address=address, sample_count=1000
        )
        source = QuantumGrpcSource(config)
        try:
            record = _pipeline(source, config).sample_token(_logits()).record
        finally:
            source.close()
        assert record.conditioning is None
        assert "conditioning" not in sober_row(0, record)

    def test_fallback_token_is_not_labelled_conditioned(
        self, old_service: tuple[str, ConditioningServicer]
    ) -> None:
        """Server will not condition -> fallback bytes, flagged, no conditioning label."""
        address, _ = old_service
        config = _hashed_config(address)
        primary = QuantumGrpcSource(config)
        source = FallbackEntropySource(primary, MockUniformSource(seed=3))
        assert source.supports_conditioning is True
        try:
            record = _pipeline(source, config).sample_token(_logits()).record
        finally:
            source.close()
        assert record.entropy_is_fallback is True
        assert record.conditioning is None
        assert "conditioning" not in sober_row(0, record)
