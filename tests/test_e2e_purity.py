"""End-to-end acceptance for the ``qthought_purity`` preset (spec FR-S5, Amendment 1).

Runs the REAL client stack — ``resolve_config`` preset expansion,
``QuantumGrpcSource`` over a live localhost gRPC channel, the hand-written
``qr_purity`` wire codec, ``ServerDrawAmplifier``, ``CoherenceGateStrategy``
and the full ``SamplingPipeline`` — against a stub PurityService built from
this package's own ``add_PurityServiceServicer_to_server``.

Pins the acceptance criteria for the cross-repo increment:

- A ``qthought_purity``-shaped config samples tokens and logs ``DrawMeta``
  (server u consumed verbatim, purity label / integrator / coherence triple
  on the record).
- Temperature gates when the stub reports ``z_c`` above threshold: with
  ``r = 1`` and ``ema_alpha = 1.0``, ``T_pre == T_base + 0.5`` EXACTLY
  (the Amendment 1 saturation pin, here across a real wire round trip).
- Temperature does NOT gate when ``coherence_valid=false``.
"""

from __future__ import annotations

import dataclasses
from concurrent import futures
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
import pytest

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.amplification.server_side import ServerDrawAmplifier
from qr_sampler.config import PRESET_QTHOUGHT_PURITY, QRSamplerConfig, resolve_config
from qr_sampler.core.pipeline import SamplingPipeline
from qr_sampler.entropy.qgrpc import QuantumGrpcSource
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.proto.purity_service_pb2 import DrawRequest, DrawResponse
from qr_sampler.proto.purity_service_pb2_grpc import (
    PurityServiceServicer,
    add_PurityServiceServicer_to_server,
)
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.coherence_gate import CoherenceGateStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

VOCAB = 64

#: Stub response with r = 1 coherence above the preset threshold (3.5) —
#: the gate saturates at exactly ``coherence_t_boost_max`` once the EMA
#: converges (immediately with ``ema_alpha = 1.0``).
_GATED_RESPONSE = DrawResponse(
    u=0.734,
    z=1.75,
    generation_timestamp_ns=1_234_567,
    source_id="qrng-a",
    coherence_z=4.2,
    coherence_valid=True,
    purity_label="quantum/intact/raw/qf:device",
    integrated_bytes=2_097_152,
    integrator="bit_z",
    coherence_r=1.0,
)


class ScriptedPurityServicer(PurityServiceServicer):
    """Stub PurityService returning a scripted response template.

    The request's ``sequence_id`` is echoed verbatim (the commitment-nonce
    contract) and every received request is recorded for assertions.
    """

    def __init__(self, template: DrawResponse) -> None:
        self.template = template
        self.requests: list[DrawRequest] = []

    def GetDraw(self, request: DrawRequest, context: Any) -> DrawResponse:  # noqa: N802
        self.requests.append(request)
        return dataclasses.replace(self.template, sequence_id=request.sequence_id)


@pytest.fixture()
def gated_service() -> Iterator[tuple[str, ScriptedPurityServicer]]:
    """A live stub PurityService whose coherence gates (z above threshold)."""
    yield from _serve(_GATED_RESPONSE)


@pytest.fixture()
def invalid_coherence_service() -> Iterator[tuple[str, ScriptedPurityServicer]]:
    """A live stub whose coherence statistic is flagged not-valid."""
    template = dataclasses.replace(_GATED_RESPONSE, coherence_valid=False)
    yield from _serve(template)


def _serve(template: DrawResponse) -> Iterator[tuple[str, ScriptedPurityServicer]]:
    servicer = ScriptedPurityServicer(template)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    add_PurityServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield f"127.0.0.1:{port}", servicer
    finally:
        server.stop(grace=None)


def _build_pipeline(
    address: str, **extra_qr_args: Any
) -> tuple[SamplingPipeline, QuantumGrpcSource]:
    """The real preset-driven stack: resolve_config -> registries -> pipeline."""
    defaults = QRSamplerConfig(_env_file=None, grpc_server_address=address)  # type: ignore[call-arg]
    config = resolve_config(defaults, {"qr_preset": PRESET_QTHOUGHT_PURITY, **extra_qr_args})

    source = QuantumGrpcSource(config)
    amplifier = AmplifierRegistry.build(config)
    strategy = TemperatureStrategyRegistry.build(config, VOCAB)
    assert isinstance(amplifier, ServerDrawAmplifier)
    assert isinstance(strategy, CoherenceGateStrategy)

    pipeline = SamplingPipeline(
        entropy_source=source,
        amplifier=amplifier,
        strategy=strategy,
        selector=TokenSelector(),
        sampling_logger=SamplingLogger(config),
        config=config,
    )
    return pipeline, source


def _logits() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(size=VOCAB).astype(np.float32)


class TestQthoughtPurityEndToEnd:
    def test_samples_tokens_and_logs_draw_meta(
        self, gated_service: tuple[str, ScriptedPurityServicer]
    ) -> None:
        """One real wire round trip: server u consumed, DrawMeta on the record."""
        address, servicer = gated_service
        pipeline, source = _build_pipeline(address)
        try:
            result = pipeline.sample_token(_logits(), build_onehot=False)
        finally:
            source.close()

        assert 0 <= result.token_id < VOCAB
        assert result.draw_meta is not None
        record = result.record
        # u and z travel as fixed64 doubles — exact across the wire.
        assert record.u_value == 0.734
        assert record.z_score == 1.75
        assert record.draw_z == 1.75
        assert record.draw_coherence_z == 4.2
        assert record.draw_coherence_valid is True
        assert record.draw_coherence_r == 1.0
        assert record.purity_label == "quantum/intact/raw/qf:device"
        assert record.integrated_bytes == 2_097_152
        assert record.integrator == "bit_z"
        assert record.draw_source_id == "qrng-a"
        assert record.entropy_is_fallback is False
        # The preset's draw shape reached the server: source_id defers to the
        # key binding (""), block_bytes is the 100 KiB the qthought lanes pin.
        assert servicer.requests[0].source_id == ""
        assert servicer.requests[0].block_bytes == 102400

    def test_gate_boosts_exactly_t_base_plus_half(
        self, gated_service: tuple[str, ScriptedPurityServicer]
    ) -> None:
        """Amendment 1 pin: r=1 above threshold, alpha=1.0 => T_pre = T_base + 0.5.

        The gate lags by one draw: token 1 samples at exactly T_base (no
        meta yet); token 2 sees token 1's coherence and boosts by exactly
        ``coherence_t_boost_max * r = 0.5``.
        """
        address, _servicer = gated_service
        pipeline, source = _build_pipeline(address, qr_coherence_ema_alpha=1.0)
        try:
            first = pipeline.sample_token(_logits(), build_onehot=False)
            second = pipeline.sample_token(_logits(), build_onehot=False)
        finally:
            source.close()

        assert first.record.temperature_used == 1.0  # T_base, lag-by-one
        assert first.record.gate_open is False
        assert first.record.gate_boost == 0.0

        assert second.record.temperature_used == 1.0 + 0.5  # exact saturation
        assert second.record.gate_open is True
        assert second.record.gate_boost == 0.5

    def test_no_gate_when_coherence_invalid(
        self, invalid_coherence_service: tuple[str, ScriptedPurityServicer]
    ) -> None:
        """coherence_valid=false => the gate never opens; T stays T_base."""
        address, _servicer = invalid_coherence_service
        pipeline, source = _build_pipeline(address, qr_coherence_ema_alpha=1.0)
        try:
            first = pipeline.sample_token(_logits(), build_onehot=False)
            second = pipeline.sample_token(_logits(), build_onehot=False)
        finally:
            source.close()

        for result in (first, second):
            record = result.record
            assert record.temperature_used == 1.0
            assert record.gate_open is False
            assert record.gate_boost == 0.0
        # The draws themselves still succeeded — only the gate stayed shut.
        assert second.draw_meta is not None
        assert second.record.draw_coherence_valid is False
