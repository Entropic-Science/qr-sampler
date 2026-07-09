"""Tests for the qthought roller.

Covers amplifier parity, fallback-flag propagation, the typed
arbitrary-arity decisions the case-frame grammar needs (``choose`` /
``choose_weighted`` / ``coin`` / ``bind_int``), the just-in-time
distinct-fetch contract (one fresh fetch per decision, invariant 4), the
three lockstep presets, and the optional, fallback-safe thought-level
aggregate protocol.
"""

from __future__ import annotations

import pytest

from qr_sampler.amplification.zscore import ZScoreMeanAmplifier
from qr_sampler.config import BUILTIN_PRESETS, QRSamplerConfig
from qr_sampler.entropy.base import DrawMeta, EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.qthought import (
    BindSpec,
    ChoiceProvenance,
    IntRange,
    QthoughtRoller,
)


class _FixedSource(EntropySource):
    """Returns a deterministic ramp buffer for parity checks (mean == 127.5)."""

    @property
    def name(self) -> str:
        return "fixed"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return bytes(i % 256 for i in range(n))

    def close(self) -> None:
        pass


class _ConstSource(EntropySource):
    """Returns a constant byte value — a deterministic, biased entropy stream."""

    def __init__(self, value: int) -> None:
        self._value = value

    @property
    def name(self) -> str:
        return "const"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return bytes([self._value]) * n

    def close(self) -> None:
        pass


class _CountingSource(EntropySource):
    """Wraps a source and counts ``get_random_bytes`` calls (JIT introspection)."""

    def __init__(self, inner: EntropySource) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        self.calls += 1
        return self._inner.get_random_bytes(n)

    def close(self) -> None:
        self._inner.close()


class _FailingSource(EntropySource):
    """Primary that is always unavailable — drives the fallback leg."""

    @property
    def name(self) -> str:
        return "failing_primary"

    @property
    def is_available(self) -> bool:
        return False

    def get_random_bytes(self, n: int) -> bytes:
        raise EntropyUnavailableError("primary down (test)")

    def close(self) -> None:
        pass


@pytest.fixture()
def mock_config() -> QRSamplerConfig:
    """Config running the roller against the mock source, small fetches."""
    return QRSamplerConfig(
        entropy_source_type="mock_uniform",
        sample_count=1024,
    )


def _amplified_u(config: QRSamplerConfig, source: EntropySource) -> float:
    """Recompute the uniform a deterministic source yields via a fresh amplifier."""
    buffer = source.get_random_bytes(config.sample_count)
    return ZScoreMeanAmplifier(config).amplify(buffer).u


# --------------------------------------------------------------------------- #
# Range + basic decision behaviour under the mock source
# --------------------------------------------------------------------------- #


def test_choose_index_in_range(mock_config: QRSamplerConfig) -> None:
    """Every choose lands in [0, k-1] with provenance fields populated."""
    roller = QthoughtRoller(mock_config)
    try:
        for _ in range(50):
            value = roller.choose(7)
            assert 0 <= value <= 6
        provenance = roller.drain()
        assert len(provenance) == 50
        for prov in provenance:
            assert isinstance(prov, ChoiceProvenance)
            assert prov.kind == "choose"
            assert 0.0 < prov.u < 1.0
            assert prov.source == "mock_uniform"
            assert prov.is_fallback is False
            assert prov.latency_ms >= 0.0
            assert prov.generation_timestamp > 0.0
            assert prov.thought_aggregate is None
    finally:
        roller.close()


def test_biased_device_choose_spreads_after_calibration() -> None:
    """The 'acorn' regression: a statically biased device must not pin every choose to 0.

    Without baseline calibration a real device's byte-mean offset saturates
    every amplified u into the CDF clamp, so choose(k) returns index 0 on
    every draw — decoding the same lexicon entry ('acorn', 'give', 'want')
    into every proto-thought in every channel mode. With the qthought preset's
    zscore_calibration_samples, the roller calibrates at build time and the
    indices spread over the pool again.
    """
    from qr_sampler.entropy.mock import MockUniformSource

    config = QRSamplerConfig(
        entropy_source_type="mock_uniform",
        sample_count=10000,
        zscore_calibration_samples=100,
    )
    biased = MockUniformSource(mean=122.0, seed=13)  # -5.5 static byte offset
    roller = QthoughtRoller(config, entropy_source=biased)
    try:
        values = [roller.choose(56) for _ in range(60)]
        assert len(set(values)) >= 10  # spread over the THING pool, not pinned
        assert not all(v == 0 for v in values)
    finally:
        roller.close()


def test_biased_device_choose_pins_without_calibration() -> None:
    """Control for the regression above: uncalibrated, the same device pins to 0."""
    from qr_sampler.entropy.mock import MockUniformSource

    config = QRSamplerConfig(
        entropy_source_type="mock_uniform",
        sample_count=10000,
    )
    biased = MockUniformSource(mean=122.0, seed=13)
    roller = QthoughtRoller(config, entropy_source=biased)
    try:
        values = [roller.choose(56) for _ in range(20)]
        assert all(v == 0 for v in values)
    finally:
        roller.close()


def test_coin_returns_bool(mock_config: QRSamplerConfig) -> None:
    """coin returns a bool and records it as the provenance value."""
    roller = QthoughtRoller(mock_config)
    try:
        result = roller.coin(0.5)
        assert isinstance(result, bool)
        (prov,) = roller.drain()
        assert prov.kind == "coin"
        assert prov.value is result
    finally:
        roller.close()


def test_bind_int_modes_in_range(mock_config: QRSamplerConfig) -> None:
    """The three named bind modes always produce an in-domain integer."""
    roller = QthoughtRoller(mock_config)
    try:
        for _ in range(40):
            assert 0 <= roller.bind_int(BindSpec.for_time()) <= 23
            assert 0 <= roller.bind_int(BindSpec.for_age()) <= 99
            assert 1900 <= roller.bind_int(BindSpec.for_year()) <= 2099
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Deterministic parity with the amplifier math on a fixed buffer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("source", [_FixedSource(), _ConstSource(60), _ConstSource(200)])
def test_choose_parity_with_amplifier(mock_config: QRSamplerConfig, source: EntropySource) -> None:
    """choose's index matches min(int(u*k), k-1) recomputed from the amplifier."""
    roller = QthoughtRoller(mock_config)
    roller._source = source
    try:
        value = roller.choose(10)
        expected_u = _amplified_u(mock_config, source)
        expected = min(int(expected_u * 10), 9)
        (prov,) = roller.drain()
        assert prov.u == pytest.approx(expected_u)
        assert value == expected
        assert prov.value == expected
    finally:
        roller.close()


def test_coin_parity_with_amplifier(mock_config: QRSamplerConfig) -> None:
    """coin's verdict matches u < p recomputed from the amplifier."""
    roller = QthoughtRoller(mock_config)
    roller._source = _ConstSource(130)
    try:
        expected_u = _amplified_u(mock_config, _ConstSource(130))
        # mean 130 > 127.5 → u well above 0.5: below 0.95, above 0.5.
        assert roller.coin(0.95) is True
        assert roller.coin(0.5) is False
        provenance = roller.drain()
        assert all(prov.u == pytest.approx(expected_u) for prov in provenance)
    finally:
        roller.close()


def test_choose_weighted_parity(mock_config: QRSamplerConfig) -> None:
    """choose_weighted buckets by the cumulative-weight CDF over u."""
    roller = QthoughtRoller(mock_config)
    weights = [1.0, 3.0]  # boundary at u == 0.25 (target 1.0 of total 4.0)
    try:
        roller._source = _ConstSource(60)  # u ≈ 0 → bucket 0
        assert roller.choose_weighted(weights) == 0
        roller._source = _ConstSource(200)  # u ≈ 1 → bucket 1
        assert roller.choose_weighted(weights) == 1
        kinds = {prov.kind for prov in roller.drain()}
        assert kinds == {"choose_weighted"}
    finally:
        roller.close()


def test_bind_int_single_range_parity(mock_config: QRSamplerConfig) -> None:
    """A single-range bind reduces to min(int(u*span), span-1) + low."""
    roller = QthoughtRoller(mock_config)
    roller._source = _ConstSource(130)
    spec = BindSpec(mode="test", ranges=(IntRange(0, 9),))
    try:
        value = roller.bind_int(spec)
        expected_u = _amplified_u(mock_config, _ConstSource(130))
        expected = min(int(expected_u * 10), 9)
        (prov,) = roller.drain()
        assert prov.u == pytest.approx(expected_u)
        assert value == expected
        assert prov.value == expected
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Just-in-time: exactly one fresh fetch per decision (invariant 4)
# --------------------------------------------------------------------------- #


def test_jit_one_fetch_per_decision(mock_config: QRSamplerConfig) -> None:
    """Each choose/coin/bind_int/choose_weighted triggers exactly one fetch."""
    counter = _CountingSource(_ConstSource(140))
    roller = QthoughtRoller(mock_config)
    roller._source = counter
    try:
        roller.choose(5)
        assert counter.calls == 1
        roller.coin(0.5)
        assert counter.calls == 2
        roller.bind_int(BindSpec.for_age())
        assert counter.calls == 3
        roller.choose_weighted([1.0, 1.0, 1.0])
        assert counter.calls == 4
    finally:
        roller.close()


def test_drain_returns_and_clears(mock_config: QRSamplerConfig) -> None:
    """drain hands back the buffered provenance once, then is empty."""
    roller = QthoughtRoller(mock_config)
    try:
        roller.choose(3)
        roller.coin(0.5)
        first = roller.drain()
        assert len(first) == 2
        assert roller.drain() == ()
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# draw_u / draw_index — raw draws, returned directly, never buffered
# --------------------------------------------------------------------------- #


def test_draw_u_in_range_and_not_buffered(mock_config: QRSamplerConfig) -> None:
    """draw_u returns a provenance directly and leaves the drain buffer untouched."""
    roller = QthoughtRoller(mock_config)
    try:
        prov = roller.draw_u()
        assert isinstance(prov, ChoiceProvenance)
        assert prov.kind == "draw_u"
        assert 0.0 < prov.u < 1.0
        assert prov.source == "mock_uniform"
        assert prov.is_fallback is False
        assert prov.latency_ms >= 0.0
        assert prov.generation_timestamp > 0.0
        assert prov.thought_aggregate is None
        assert roller.drain() == ()  # never appended to the buffer
    finally:
        roller.close()


def test_draw_index_in_range_and_not_buffered(mock_config: QRSamplerConfig) -> None:
    """draw_index returns a provenance directly, in-range value, never buffered."""
    roller = QthoughtRoller(mock_config)
    try:
        for _ in range(20):
            prov = roller.draw_index(9)
            assert prov.kind == "draw_index"
            assert isinstance(prov.value, int)
            assert 0 <= prov.value <= 8
        assert roller.drain() == ()  # 20 draws, none buffered
    finally:
        roller.close()


def test_draw_u_parity_with_amplifier(mock_config: QRSamplerConfig) -> None:
    """draw_u's u matches the amplifier recomputed from the same deterministic buffer."""
    roller = QthoughtRoller(mock_config)
    roller._source = _ConstSource(160)
    try:
        prov = roller.draw_u()
        expected_u = _amplified_u(mock_config, _ConstSource(160))
        assert prov.u == pytest.approx(expected_u)
    finally:
        roller.close()


def test_draw_index_parity_with_choose(mock_config: QRSamplerConfig) -> None:
    """draw_index(k) maps u to an index exactly like choose(k)'s min(int(u*k), k-1)."""
    roller = QthoughtRoller(mock_config)
    roller._source = _ConstSource(160)
    try:
        prov = roller.draw_index(10)
        expected_u = _amplified_u(mock_config, _ConstSource(160))
        expected = min(int(expected_u * 10), 9)
        assert prov.value == expected
    finally:
        roller.close()


def test_draw_index_rejects_non_positive_k(mock_config: QRSamplerConfig) -> None:
    roller = QthoughtRoller(mock_config)
    try:
        with pytest.raises(ValueError, match="k >= 1"):
            roller.draw_index(0)
    finally:
        roller.close()


def test_draw_u_jit_one_fetch_per_draw(mock_config: QRSamplerConfig) -> None:
    """Each draw_u/draw_index triggers exactly one fresh fetch (invariant 4)."""
    counter = _CountingSource(_ConstSource(140))
    roller = QthoughtRoller(mock_config)
    roller._source = counter
    try:
        roller.draw_u()
        assert counter.calls == 1
        roller.draw_index(5)
        assert counter.calls == 2
    finally:
        roller.close()


def test_draw_u_fallback_flag_propagates(mock_config: QRSamplerConfig) -> None:
    """A primary EntropyUnavailableError surfaces on draw_u as is_fallback=True."""
    roller = QthoughtRoller(mock_config)
    roller._source = FallbackEntropySource(_FailingSource(), _FixedSource())
    try:
        prov = roller.draw_u()
        assert prov.is_fallback is True
        assert prov.source == "fixed"
    finally:
        roller.close()


def test_draw_u_does_not_open_or_close_thought_scope(mock_config: QRSamplerConfig) -> None:
    """draw_u/draw_index never toggle the buffered-decision thought-scope flag."""
    roller = QthoughtRoller(mock_config)
    try:
        assert roller._thought_active is False
        roller.draw_u()
        assert roller._thought_active is False
        roller.draw_index(4)
        assert roller._thought_active is False
    finally:
        roller.close()


def test_roller_accepts_entropy_source_ctor_kwarg() -> None:
    """The entropy_source= ctor seam injects a source without a private-attribute poke."""
    source = _ConstSource(160)
    config = QRSamplerConfig(entropy_source_type="mock_uniform", sample_count=1024)
    roller = QthoughtRoller(config, entropy_source=source)
    try:
        prov = roller.draw_u()
        expected_u = _amplified_u(config, source)
        assert prov.u == pytest.approx(expected_u)
        assert prov.source == "const"
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Fallback labelling + status
# --------------------------------------------------------------------------- #


def test_fallback_flag_propagates(mock_config: QRSamplerConfig) -> None:
    """A primary EntropyUnavailableError surfaces as is_fallback=True."""
    roller = QthoughtRoller(mock_config)
    roller._source = FallbackEntropySource(_FailingSource(), _FixedSource())
    try:
        value = roller.choose(8)
        assert 0 <= value <= 7
        (prov,) = roller.drain()
        assert prov.is_fallback is True
        assert prov.source == "fixed"

        status = roller.status()
        assert status["currently_degraded"] is True
        assert status["fallback_count"] == 1
    finally:
        roller.close()


def test_status_without_fallback_wrapper(mock_config: QRSamplerConfig) -> None:
    """fallback_mode='error' builds a bare source; status reports clean."""
    config = mock_config.model_copy(update={"fallback_mode": "error"})
    roller = QthoughtRoller(config)
    try:
        status = roller.status()
        assert status["currently_degraded"] is False
        assert status["fallback_count"] == 0
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Optional, fallback-safe thought-level aggregate protocol
# --------------------------------------------------------------------------- #


def test_thought_aggregate_folded_when_amplifier_supports_it() -> None:
    """The zscore_thought amplifier folds a thought aggregate on drain.

    The default qthought preset now uses server draws (no byte accumulator),
    so this pins the aggregate protocol on an explicit zscore_thought config —
    the amplifier that still carries it for any byte-lane caller.
    """
    config = QRSamplerConfig(
        entropy_source_type="mock_uniform",
        signal_amplifier_type="zscore_thought",
        sample_count=1024,
    )
    roller = QthoughtRoller(config)
    roller._source = _ConstSource(160)
    try:
        assert hasattr(roller._amplifier, "begin_thought")
        roller.begin_thought()
        roller.choose(6)
        roller.coin(0.5)
        provenance = roller.drain()
        assert len(provenance) == 2
        folded = 2 * roller.config.sample_count
        for prov in provenance:
            assert prov.thought_aggregate is not None
            assert prov.thought_aggregate["sample_count"] == folded
            assert prov.thought_aggregate["bias"] > 0.0  # const 160 > 127.5
            assert set(prov.thought_aggregate) >= {"z_score", "bias", "u", "sample_count"}
    finally:
        roller.close()


def test_thought_aggregate_omitted_for_zscore_mean(mock_config: QRSamplerConfig) -> None:
    """With zscore_mean the thought protocol is absent and the aggregate is omitted."""
    roller = QthoughtRoller(mock_config)
    try:
        assert not hasattr(roller._amplifier, "begin_thought")
        roller.begin_thought()  # no-op beyond opening the scope
        roller.choose(4)
        (prov,) = roller.drain()
        assert prov.thought_aggregate is None
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_choose_rejects_non_positive_k(mock_config: QRSamplerConfig) -> None:
    roller = QthoughtRoller(mock_config)
    try:
        with pytest.raises(ValueError, match="k >= 1"):
            roller.choose(0)
    finally:
        roller.close()


def test_coin_rejects_out_of_range_p(mock_config: QRSamplerConfig) -> None:
    roller = QthoughtRoller(mock_config)
    try:
        with pytest.raises(ValueError, match="0 <= p <= 1"):
            roller.coin(1.5)
    finally:
        roller.close()


def test_choose_weighted_rejects_bad_weights(mock_config: QRSamplerConfig) -> None:
    roller = QthoughtRoller(mock_config)
    try:
        with pytest.raises(ValueError, match="at least one weight"):
            roller.choose_weighted([])
        with pytest.raises(ValueError, match="non-negative"):
            roller.choose_weighted([1.0, -1.0])
        with pytest.raises(ValueError, match="positive total"):
            roller.choose_weighted([0.0, 0.0])
    finally:
        roller.close()


def test_bind_int_rejects_bad_spec(mock_config: QRSamplerConfig) -> None:
    roller = QthoughtRoller(mock_config)
    try:
        with pytest.raises(ValueError, match="at least one range"):
            roller.bind_int(BindSpec(mode="empty", ranges=()))
        with pytest.raises(ValueError, match="high"):
            roller.bind_int(BindSpec(mode="bad", ranges=(IntRange(9, 0),)))
        with pytest.raises(ValueError, match="weight must be positive"):
            roller.bind_int(BindSpec(mode="bad", ranges=(IntRange(0, 9, 0.0),)))
    finally:
        roller.close()


# --------------------------------------------------------------------------- #
# Presets + default construction + registry lazy resolution
# --------------------------------------------------------------------------- #


def test_qthought_presets_registered() -> None:
    """The three qthought lanes pin server-integrated draws (1 MiB blocks); the
    byte-fetch fields (sample_count / calibration) survive as degrade-fallback."""
    assert BUILTIN_PRESETS["qthought"] == {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    }
    assert BUILTIN_PRESETS["qthought_think"] == {
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 6000,
        "zscore_calibration_samples": 200,
    }
    assert BUILTIN_PRESETS["qthought_voice"] == {
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 0,
        "top_p": 1.0,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    }


def test_default_construction_resolves_qthought_preset() -> None:
    """No-arg construction resolves the qthought preset (quantum + server draws)."""
    roller = QthoughtRoller()
    try:
        assert roller.config.signal_amplifier_type == "server"
        assert roller.config.draw_block_bytes == 1048576
        assert roller.config.entropy_source_type == "quantum_grpc"
    finally:
        roller.close()


def test_quantum_grpc_resolves_via_builtin_table() -> None:
    """quantum_grpc resolves through the registry's lazy builtin table.

    Nothing else needs to have imported the qgrpc source module —
    this is the property that made the roller's old import-nudge helper
    deletable.
    """
    from qr_sampler.entropy.registry import EntropySourceRegistry

    assert EntropySourceRegistry.get("quantum_grpc") is not None


# --------------------------------------------------------------------------- #
# Server-draw decode path (signal_amplifier_type="server")
# --------------------------------------------------------------------------- #


class _FakeDrawSource(EntropySource):
    """Draw-capable source returning a scripted ``(u, meta)`` and counting calls."""

    supports_server_draw = True

    def __init__(self, u: float = 0.734, z: float = 1.5) -> None:
        self._u = u
        self._z = z
        self.draw_calls: list[tuple[int, str]] = []
        self.byte_calls = 0

    @property
    def name(self) -> str:
        return "fake_draw"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        self.byte_calls += 1
        return bytes(i % 256 for i in range(n))

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: object | None = None
    ) -> tuple[float, DrawMeta]:
        self.draw_calls.append((block_bytes, source_id))
        return self._u, DrawMeta(
            z=self._z,
            coherence_z=0.0,
            coherence_valid=False,
            coherence_r=0.0,
            purity_label="quantum/intact/raw",
            integrated_bytes=block_bytes,
            integrator="bit_z",
            source_id="dragonfly-0",
        )

    def close(self) -> None:
        pass


class _DrawlessSource(_FakeDrawSource):
    """Server-draw capable on paper; every draw fails (PurityService down)."""

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: object | None = None
    ) -> tuple[float, DrawMeta]:
        self.draw_calls.append((block_bytes, source_id))
        raise EntropyUnavailableError("PurityService down")


def _server_config(**over: object) -> QRSamplerConfig:
    return QRSamplerConfig(
        entropy_source_type="mock_uniform",
        signal_amplifier_type="server",
        draw_block_bytes=1048576,
        sample_count=256,
        **over,  # type: ignore[arg-type]
    )


def test_server_draw_mode_uses_get_draw_per_decision() -> None:
    """With the server amplifier, each decision is one ``get_draw`` (1 MiB block),
    no byte fetch, provenance from ``DrawMeta`` (z from the server, bias 0)."""
    src = _FakeDrawSource(u=0.734, z=1.5)
    roller = QthoughtRoller(_server_config(), entropy_source=src)
    try:
        for _ in range(5):
            roller.choose(10)
        roller.coin(0.5)
        roller.bind_int(BindSpec.for_time())
        prov = roller.drain()
        assert len(prov) == 7
        assert len(src.draw_calls) == 7  # exactly one server draw per decision
        assert src.byte_calls == 0  # the happy path never fetches bytes
        assert all(block == 1048576 for block, _ in src.draw_calls)  # 1 MiB
        for p in prov:
            assert p.is_fallback is False
            assert p.z_score == 1.5
            assert p.bias == 0.0  # baseline correction happened server-side
            assert 0.0 < p.u < 1.0
    finally:
        roller.close()


def test_server_draw_u_and_draw_index_use_get_draw() -> None:
    """The raw-draw methods (dispose gate + persona seed) also draw server-side."""
    src = _FakeDrawSource(u=0.42, z=-0.8)
    roller = QthoughtRoller(_server_config(), entropy_source=src)
    try:
        prov_u = roller.draw_u()
        assert prov_u.kind == "draw_u"
        assert prov_u.u == 0.42
        assert prov_u.is_fallback is False
        prov_idx = roller.draw_index(8)
        assert prov_idx.kind == "draw_index"
        assert 0 <= prov_idx.value <= 7
        assert len(src.draw_calls) == 2
        assert src.byte_calls == 0
    finally:
        roller.close()


def test_server_draw_degrades_to_local_labelled_on_failure() -> None:
    """A failed draw degrades to local bytes + a calibrated fallback amplifier,
    labelled ``is_fallback`` — a decision is never muted."""
    src = _DrawlessSource()
    roller = QthoughtRoller(_server_config(zscore_calibration_samples=0), entropy_source=src)
    try:
        value = roller.choose(6)
        assert 0 <= value <= 5
        (p,) = roller.drain()
        assert p.is_fallback is True
        assert len(src.draw_calls) == 1  # the draw was attempted
        assert src.byte_calls >= 1  # then it fell back to a byte fetch
    finally:
        roller.close()


def test_server_draw_failure_raises_under_fallback_error() -> None:
    """``fallback_mode="error"`` re-raises a failed draw instead of degrading."""
    src = _DrawlessSource()
    roller = QthoughtRoller(_server_config(fallback_mode="error"), entropy_source=src)
    try:
        with pytest.raises(EntropyUnavailableError):
            roller.choose(4)
    finally:
        roller.close()
