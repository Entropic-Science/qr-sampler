"""Tests for the coherence-gated temperature strategy (spec FR-S4).

Covers, per Amendment 1.2:

- registry wiring and per-request config fields;
- the structural one-draw lag (meta observed at token t affects token t+1);
- the significance gate (``coherence_valid`` AND ``coherence_z >= threshold``);
- exact EMA arithmetic and the ``r=1, alpha=1.0 => T_base + 0.5`` saturation pin;
- composition with every builtin inner strategy (incl. hvh's ``min_p``
  diagnostics passthrough) via differential twins;
- the upstream-of-truncation effect (a token outside the base-T top-p
  nucleus survives under the boosted T);
- every fail-safe branch returning exactly ``T_base`` with no exception;
- ``gate_open``/``gate_boost`` landing on pipeline ``TokenSamplingRecord``s.
"""

from __future__ import annotations

import os
from typing import Any, ClassVar

import numpy as np

from qr_sampler.amplification.server_side import ServerDrawAmplifier
from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import SamplingPipeline
from qr_sampler.entropy.base import DrawMeta, EntropySource
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.base import TemperatureResult
from qr_sampler.temperature.coherence_gate import CoherenceGateStrategy
from qr_sampler.temperature.edt import EDTTemperatureStrategy
from qr_sampler.temperature.hvh_drift import HVHDriftStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

VOCAB = 64


def _meta(**overrides: Any) -> DrawMeta:
    """A gate-opening DrawMeta (z above default threshold) with overrides."""
    fields: dict[str, Any] = {
        "z": 1.75,
        "coherence_z": 4.2,
        "coherence_valid": True,
        "coherence_r": 0.31,
        "purity_label": "quantum/intact/raw/qf:device",
        "integrated_bytes": 2_097_152,
        "integrator": "bit_z",
        "source_id": "qrng-a",
    }
    fields.update(overrides)
    return DrawMeta(**fields)


def _config(**overrides: Any) -> QRSamplerConfig:
    overrides.setdefault("temperature_strategy", "coherence_gate")
    return QRSamplerConfig(_env_file=None, sample_count=128, **overrides)  # type: ignore[call-arg]


def _logits() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(size=VOCAB).astype(np.float32)


def _gate(config: QRSamplerConfig) -> CoherenceGateStrategy:
    strategy = TemperatureStrategyRegistry.build(config, VOCAB)
    assert isinstance(strategy, CoherenceGateStrategy)
    return strategy


# ---------------------------------------------------------------------------
# Registry + config wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_registered_builtin(self) -> None:
        assert "coherence_gate" in TemperatureStrategyRegistry.list_registered()
        assert TemperatureStrategyRegistry.get("coherence_gate") is CoherenceGateStrategy

    def test_build_from_config_takes_vocab_size(self) -> None:
        strategy = _gate(_config())
        assert strategy._vocab_size == VOCAB

    def test_config_defaults(self) -> None:
        config = _config()
        assert config.coherence_threshold == 3.5
        assert config.coherence_t_boost_max == 0.5
        assert config.coherence_ema_alpha == 0.3
        assert config.coherence_inner_strategy == "fixed"


# ---------------------------------------------------------------------------
# Lag, significance gate, EMA arithmetic
# ---------------------------------------------------------------------------


class TestGateBehavior:
    def test_first_token_is_exactly_t_base(self) -> None:
        config = _config(fixed_temperature=0.9)
        result = _gate(config).compute_temperature(_logits(), config)
        assert result.temperature == 0.9
        assert result.diagnostics["gate_open"] is False
        assert result.diagnostics["gate_boost"] == 0.0
        assert result.diagnostics["coherence_z"] is None
        assert result.diagnostics["coherence_valid"] is False
        assert result.diagnostics["strategy"] == "coherence_gate"
        assert result.diagnostics["inner"] == "fixed"

    def test_lag_by_one_meta_affects_next_token_only(self) -> None:
        """Meta observed after token t's compute first boosts token t+1."""
        config = _config()
        strategy = _gate(config)
        first = strategy.compute_temperature(_logits(), config)
        assert first.temperature == config.fixed_temperature

        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        second = strategy.compute_temperature(_logits(), config)
        assert second.temperature > config.fixed_temperature
        assert second.diagnostics["gate_open"] is True

    def test_significance_gate_below_threshold_boost_exactly_zero(self) -> None:
        config = _config()
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_z=3.49, coherence_r=1.0))
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["gate_boost"] == 0.0
        assert result.diagnostics["gate_open"] is False
        # The meta itself is still surfaced for the record.
        assert result.diagnostics["coherence_z"] == 3.49
        assert result.diagnostics["coherence_valid"] is True

    def test_invalid_coherence_never_opens_gate(self) -> None:
        config = _config()
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_valid=False, coherence_z=99.0, coherence_r=1.0))
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["gate_open"] is False
        assert result.diagnostics["coherence_valid"] is False

    def test_negative_r_clamped_to_zero_boost(self) -> None:
        config = _config()
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_r=-0.8))
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["gate_boost"] == 0.0

    def test_saturation_pin_r1_alpha1_is_exactly_t_base_plus_half(self) -> None:
        """Amendment 1 pin: r=1, z above threshold, alpha=1.0 => T_base + 0.5."""
        config = _config(coherence_ema_alpha=1.0, fixed_temperature=1.0)
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == 1.0 + 0.5
        assert result.diagnostics["gate_boost"] == 0.5
        assert result.diagnostics["gate_open"] is True

    def test_ema_arithmetic_exact(self) -> None:
        """b_ema <- alpha*b + (1-alpha)*b_ema, hand-tracked over three tokens."""
        alpha = 0.3
        config = _config(coherence_ema_alpha=alpha)
        strategy = _gate(config)
        expected = 0.0

        # Token 1: gate open at r=1 -> b = 0.5.
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        expected = alpha * 0.5 + (1.0 - alpha) * expected
        r1 = strategy.compute_temperature(_logits(), config)
        assert r1.diagnostics["gate_boost"] == expected
        assert r1.temperature == config.fixed_temperature + expected

        # Token 2: gate open at r=0.4 -> b = 0.2.
        strategy.observe_draw_meta(_meta(coherence_r=0.4))
        expected = alpha * 0.2 + (1.0 - alpha) * expected
        r2 = strategy.compute_temperature(_logits(), config)
        assert r2.diagnostics["gate_boost"] == expected

        # Token 3: below threshold -> b = 0, EMA decays (gate stays open
        # while residual boost > 0).
        strategy.observe_draw_meta(_meta(coherence_z=1.0, coherence_r=1.0))
        expected = (1.0 - alpha) * expected
        r3 = strategy.compute_temperature(_logits(), config)
        assert r3.diagnostics["gate_boost"] == expected
        assert r3.diagnostics["gate_open"] is True
        assert r3.temperature == config.fixed_temperature + expected

    def test_ema_snaps_to_zero_and_gate_recloses(self) -> None:
        """Geometric decay alone never reaches 0.0 — the floor snap must.

        One open event, then a long run of below-threshold tokens: the EMA
        must eventually become exactly 0.0 (not ~1e-30) so ``gate_open``
        genuinely re-closes instead of reporting an open gate forever.
        """
        config = _config(coherence_ema_alpha=0.3)
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        strategy.compute_temperature(_logits(), config)  # gate opens

        result = None
        for _ in range(60):  # 0.15 * 0.7^n < 1e-6 well before 60 tokens
            strategy.observe_draw_meta(_meta(coherence_z=1.0))  # below threshold
            result = strategy.compute_temperature(_logits(), config)
        assert result is not None
        assert result.diagnostics["gate_boost"] == 0.0  # exactly, via the snap
        assert result.diagnostics["gate_open"] is False
        assert result.temperature == config.fixed_temperature

    def test_none_meta_clears_stale_evidence(self) -> None:
        """``observe_draw_meta(None)`` (degraded draw) hard-resets the gate."""
        config = _config(coherence_ema_alpha=1.0)
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        boosted = strategy.compute_temperature(_logits(), config)
        assert boosted.diagnostics["gate_open"] is True

        strategy.observe_draw_meta(None)  # the pipeline's degraded-draw signal
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["gate_open"] is False
        assert result.diagnostics["gate_boost"] == 0.0


# ---------------------------------------------------------------------------
# Composition with builtin inner strategies
# ---------------------------------------------------------------------------


class TestComposition:
    def _boosted(self, config: QRSamplerConfig) -> CoherenceGateStrategy:
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        return strategy

    def test_fixed_inner_pre_shift(self) -> None:
        config = _config(coherence_ema_alpha=1.0, fixed_temperature=0.8)
        result = self._boosted(config).compute_temperature(_logits(), config)
        assert result.temperature == 0.8 + 0.5
        assert result.diagnostics["inner"] == "fixed"

    def test_edt_inner_differential_twin(self) -> None:
        """The boost shifts edt_base_temp INSIDE the EDT formula (pre-clamp)."""
        config = _config(coherence_inner_strategy="edt", coherence_ema_alpha=1.0)
        logits = _logits()
        result = self._boosted(config).compute_temperature(logits, config)

        twin_cfg = config.model_copy(update={"edt_base_temp": config.edt_base_temp + 0.5})
        twin = EDTTemperatureStrategy(VOCAB).compute_temperature(logits, twin_cfg)
        assert result.temperature == twin.temperature
        assert result.shannon_entropy == twin.shannon_entropy
        assert result.diagnostics["inner"] == "edt"

    def test_hvh_inner_differential_twin_and_min_p_passthrough(self) -> None:
        """HVH state parity across two tokens; per-token min_p survives merge."""
        config = _config(coherence_inner_strategy="hvh_drift", coherence_ema_alpha=1.0)
        strategy = _gate(config)
        twin = HVHDriftStrategy(VOCAB)
        logits_1, logits_2 = _logits(), _logits() * 1.7

        first = strategy.compute_temperature(logits_1, config)
        twin_first = twin.compute_temperature(logits_1, config)
        assert first.temperature == twin_first.temperature
        assert first.diagnostics["min_p"] == twin_first.diagnostics["min_p"]

        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        second = strategy.compute_temperature(logits_2, config)
        twin_cfg = config.model_copy(update={"hvh_t_base": config.hvh_t_base + 0.5})
        twin_second = twin.compute_temperature(logits_2, twin_cfg)
        assert second.temperature == twin_second.temperature
        # hvh diagnostics pass through the merge (min_p is what the
        # pipeline forwards to the selector).
        assert second.diagnostics["min_p"] == twin_second.diagnostics["min_p"]
        assert second.diagnostics["varentropy"] == twin_second.diagnostics["varentropy"]
        assert second.diagnostics["strategy"] == "coherence_gate"
        assert second.diagnostics["inner"] == "hvh_drift"

    def test_unknown_base_field_inner_post_adds_boost(self) -> None:
        """A registered-but-unmapped inner gets the boost post-added."""
        config = _config(coherence_ema_alpha=1.0)
        logits = _logits()
        strategy = self._boosted(config)
        # Inject a third-party-style inner the _BASE_FIELD table doesn't know.
        strategy._inner = EDTTemperatureStrategy(VOCAB)
        strategy._inner_name = "custom_edt"
        result = strategy.compute_temperature(logits, config)

        base = EDTTemperatureStrategy(VOCAB).compute_temperature(logits, config)
        assert result.temperature == base.temperature + 0.5
        assert result.diagnostics["inner"] == "custom_edt"


# ---------------------------------------------------------------------------
# Upstream-of-truncation differential
# ---------------------------------------------------------------------------


class TestTruncationDifferential:
    def test_token_outside_base_top_p_survives_under_boost(self) -> None:
        """The boost widens the top-p nucleus because temperature division is
        selector stage 1, upstream of truncation. With strictly descending
        logits the nucleus is always the prefix 0..k-1, so num_candidates
        identifies exactly which token ids survive."""
        logits = np.linspace(4.0, 0.0, 16).astype(np.float64)
        config = _config(coherence_ema_alpha=1.0, fixed_temperature=1.0)
        strategy = _gate(config)
        selector = TokenSelector()

        t_base = strategy.compute_temperature(logits, config).temperature
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        t_boosted = strategy.compute_temperature(logits, config).temperature
        assert t_base == 1.0
        assert t_boosted == 1.5

        base_sel = selector.select(logits, t_base, top_k=0, top_p=0.8, u=0.5)
        boosted_sel = selector.select(logits, t_boosted, top_k=0, top_p=0.8, u=0.5)
        # Token id base_n is outside the base-T nucleus (prefix 0..base_n-1)
        # but inside the boosted-T one.
        assert boosted_sel.num_candidates > base_sel.num_candidates


# ---------------------------------------------------------------------------
# Fail-safe branches — every one returns exactly T_base, nothing escapes
# ---------------------------------------------------------------------------


class _ExplodingMeta:
    """Duck-typed meta whose attribute access blows up mid-gate."""

    @property
    def coherence_valid(self) -> bool:
        raise RuntimeError("boom")

    coherence_z = 4.2
    coherence_r = 1.0


class TestFailSafe:
    def test_exploding_meta_yields_exactly_t_base_and_resets_boost(self) -> None:
        config = _config(coherence_ema_alpha=1.0, fixed_temperature=1.1)
        strategy = _gate(config)
        # Accumulate a real boost first, then poison the next observation.
        strategy.observe_draw_meta(_meta(coherence_r=1.0))
        assert strategy.compute_temperature(_logits(), config).temperature == 1.1 + 0.5

        strategy.observe_draw_meta(_ExplodingMeta())
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == 1.1
        assert result.diagnostics["gate_open"] is False
        assert result.diagnostics["gate_boost"] == 0.0
        assert result.diagnostics["coherence_z"] is None
        assert result.diagnostics["coherence_valid"] is False

    def test_non_numeric_meta_fields_yield_exactly_t_base(self) -> None:
        config = _config()
        strategy = _gate(config)
        strategy.observe_draw_meta(_meta(coherence_z="not-a-number"))
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["gate_boost"] == 0.0

    def test_unknown_inner_strategy_falls_back_to_fixed(self) -> None:
        config = _config(coherence_inner_strategy="no_such_strategy")
        result = _gate(config).compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["inner"] == "fixed"

    def test_self_referential_inner_falls_back_to_fixed(self) -> None:
        config = _config(coherence_inner_strategy="coherence_gate")
        strategy = _gate(config)
        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == config.fixed_temperature
        assert result.diagnostics["inner"] == "fixed"

    def test_boosted_inner_failure_falls_back_to_unmodified_inner(self) -> None:
        """If the BOOSTED inner run raises, the token gets exactly T_base."""

        class _BoostAllergicInner:
            def compute_temperature(
                self, logits: np.ndarray, config: QRSamplerConfig
            ) -> TemperatureResult:
                if config.fixed_temperature != 1.0:
                    raise RuntimeError("cannot handle a shifted base")
                return TemperatureResult(
                    temperature=config.fixed_temperature,
                    shannon_entropy=0.0,
                    diagnostics={"strategy": "fixed"},
                )

        config = _config(coherence_ema_alpha=1.0, fixed_temperature=1.0)
        strategy = _gate(config)
        strategy._inner = _BoostAllergicInner()  # type: ignore[assignment]
        strategy._inner_name = "fixed"
        strategy.observe_draw_meta(_meta(coherence_r=1.0))

        result = strategy.compute_temperature(_logits(), config)
        assert result.temperature == 1.0
        assert result.diagnostics["gate_open"] is False
        assert result.diagnostics["gate_boost"] == 0.0


# ---------------------------------------------------------------------------
# Pipeline records carry the gate flags
# ---------------------------------------------------------------------------


class FakeDrawSource(EntropySource):
    """Draw-capable in-memory source returning a scripted (u, meta)."""

    supports_server_draw: ClassVar[bool] = True

    def __init__(self, u: float = 0.734, meta: DrawMeta | None = None) -> None:
        self.u = u
        self.meta = meta if meta is not None else _meta()

    @property
    def name(self) -> str:
        return "fake_draw"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return os.urandom(n)

    def get_draw(
        self, block_bytes: int, source_id: str, ticket: Any | None = None
    ) -> tuple[float, DrawMeta]:
        return self.u, self.meta

    def close(self) -> None:
        pass


class TestPipelineGateRecords:
    def _pipeline(self, config: QRSamplerConfig, meta: DrawMeta) -> SamplingPipeline:
        return SamplingPipeline(
            entropy_source=FakeDrawSource(meta=meta),
            amplifier=ServerDrawAmplifier(config),
            strategy=TemperatureStrategyRegistry.build(config, VOCAB),
            selector=TokenSelector(),
            sampling_logger=SamplingLogger(config),
            config=config,
        )

    def test_gate_open_and_boost_land_on_records_with_one_draw_lag(self) -> None:
        config = _config(signal_amplifier_type="server", coherence_ema_alpha=1.0)
        pipeline = self._pipeline(config, _meta(coherence_r=1.0))

        first = pipeline.sample_token(_logits(), build_onehot=False).record
        assert first.gate_open is False
        assert first.gate_boost == 0.0

        second = pipeline.sample_token(_logits(), build_onehot=False).record
        assert second.gate_open is True
        assert second.gate_boost == 0.5

    def test_below_threshold_records_stay_closed(self) -> None:
        config = _config(signal_amplifier_type="server")
        pipeline = self._pipeline(config, _meta(coherence_z=0.5))
        pipeline.sample_token(_logits(), build_onehot=False)
        record = pipeline.sample_token(_logits(), build_onehot=False).record
        assert record.gate_open is False
        assert record.gate_boost == 0.0
        assert record.draw_coherence_z == 0.5

    def test_gate_state_published_to_status_file_on_change(self, tmp_path, monkeypatch) -> None:
        """FR-T3 enabler: gate transitions land in the cross-process file."""
        from qr_sampler.telemetry import status_file

        path = tmp_path / "qr_entropy_status.json"
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(path))

        config = _config(signal_amplifier_type="server", coherence_ema_alpha=1.0)
        pipeline = self._pipeline(config, _meta(coherence_r=1.0))

        pipeline.sample_token(_logits(), build_onehot=False)  # gate closed
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["gate_open"] is False
        assert data["gate_boost"] == 0.0
        assert data["coherence_valid"] is True

        pipeline.sample_token(_logits(), build_onehot=False)  # gate opens
        data = status_file.read_entropy_status()
        assert data is not None
        assert data["gate_open"] is True
        assert data["gate_boost"] == 0.5

        # Steady state: same gate values do not rewrite the file.
        before = path.stat().st_mtime_ns
        pipeline.sample_token(_logits(), build_onehot=False)
        assert path.stat().st_mtime_ns == before

    def test_degraded_draw_resets_gate_instead_of_replaying_stale_meta(self) -> None:
        """A PurityService outage must hard-reset the gate, not replay evidence.

        Token 1 draws fine (gate-opening meta observed). Token 2's draw fails
        (degraded) — its own temperature legitimately still uses token 1's
        meta (one-draw lag), but the failure must clear the stored evidence so
        token 3 is exactly T_base with a closed gate.
        """

        class _OutageAfterFirstDraw(FakeDrawSource):
            def __init__(self) -> None:
                super().__init__(meta=_meta(coherence_r=1.0))
                self.calls = 0

            def get_draw(
                self, block_bytes: int, source_id: str, ticket: Any | None = None
            ) -> tuple[float, DrawMeta]:
                self.calls += 1
                if self.calls > 1:
                    raise EntropyUnavailableError("purity service down")
                return self.u, self.meta

        config = _config(signal_amplifier_type="server", coherence_ema_alpha=1.0)
        pipeline = SamplingPipeline(
            entropy_source=_OutageAfterFirstDraw(),
            amplifier=ServerDrawAmplifier(config),
            strategy=TemperatureStrategyRegistry.build(config, VOCAB),
            selector=TokenSelector(),
            sampling_logger=SamplingLogger(config),
            config=config,
        )

        first = pipeline.sample_token(_logits(), build_onehot=False).record
        assert first.gate_open is False  # one-draw lag: nothing observed yet

        second = pipeline.sample_token(_logits(), build_onehot=False).record
        assert second.gate_open is True  # token 1's meta, legitimately
        assert second.entropy_is_fallback is True  # but this draw degraded

        third = pipeline.sample_token(_logits(), build_onehot=False).record
        assert third.gate_open is False  # outage cleared the stale evidence
        assert third.gate_boost == 0.0

    def test_no_gate_diag_no_status_write(self, tmp_path, monkeypatch) -> None:
        """Non-gate strategies never touch the status file from the pipeline."""
        from qr_sampler.telemetry import status_file

        path = tmp_path / "qr_entropy_status.json"
        monkeypatch.setenv("QR_ENTROPY_STATUS_FILE", str(path))

        config = _config(signal_amplifier_type="server", temperature_strategy="fixed")
        pipeline = self._pipeline(config, _meta())
        pipeline.sample_token(_logits(), build_onehot=False)
        assert status_file.read_entropy_status() is None
