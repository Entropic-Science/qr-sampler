"""Tests for qr_sampler.config.presets resolution helpers.

Covers BUILTIN_PRESETS shape, resolve_preset merge semantics, and the
expand_extra_args entry point invoked by resolve_config().
"""

from __future__ import annotations

import pytest

from qr_sampler.config import (
    BUILTIN_PRESETS,
    QRSamplerConfig,
    expand_extra_args,
    resolve_config,
    resolve_preset,
)
from qr_sampler.exceptions import ConfigValidationError

# Expected V6_HVD_R01_01 winner values (spec §2.3).
_CREATIVE_EXPECTED: dict[str, object] = {
    "qr_temperature_strategy": "hvh_drift",
    "qr_hvh_t_base": 1.35,
    "qr_hvh_alpha_h": 0.3,
    "qr_hvh_alpha_vh": -0.2,
    "qr_hvh_gamma_dh": 1.0,
    "qr_hvh_delta_dvh": 0.5,
    "qr_hvh_lambda_ema": 0.02,
    "qr_hvh_min_p_base": 0.025,
    "qr_hvh_kappa_h": 0.03,
    "qr_hvh_nu_dh": 0.02,
    "qr_top_k": 0,
    "qr_top_p": 1.0,
}

_NORMAL_T1_EXPECTED: dict[str, object] = {
    "qr_temperature_strategy": "fixed",
    "qr_fixed_temperature": 1.0,
    "qr_top_k": 0,
    "qr_top_p": 1.0,
}


class TestResolvePreset:
    """resolve_preset(): preset -> qr_* dict with caller-wins merge."""

    def test_creative_sampling_expands_to_v6_winner_values(self) -> None:
        result = resolve_preset("creative_sampling", {})
        assert result == _CREATIVE_EXPECTED

    def test_normal_t1_expands_to_baseline(self) -> None:
        result = resolve_preset("normal_t1", {})
        assert result == _NORMAL_T1_EXPECTED
        # Exactly 4 keys; no HVH-Drift hyperparameters leak through.
        assert len(result) == 4
        assert not any(key.startswith("qr_hvh_") for key in result)

    def test_per_request_override_wins_over_preset(self) -> None:
        result = expand_extra_args(
            {"qr_preset": "creative_sampling", "qr_hvh_t_base": 0.7},
            QRSamplerConfig(_env_file=None),  # type: ignore[call-arg]
        )
        assert result["qr_hvh_t_base"] == 0.7
        # Other preset keys still applied.
        assert result["qr_temperature_strategy"] == "hvh_drift"
        assert result["qr_hvh_alpha_h"] == 0.3

    def test_unknown_preset_raises_config_error(self) -> None:
        with pytest.raises(ConfigValidationError) as exc_info:
            resolve_preset("does_not_exist", {})
        message = str(exc_info.value)
        assert "Unknown preset" in message
        # Known presets listed for discoverability (FR-11).
        for known in BUILTIN_PRESETS:
            assert known in message

    def test_qr_preset_key_stripped_from_output(self) -> None:
        result = resolve_preset("creative_sampling", {"qr_preset": "creative_sampling"})
        assert "qr_preset" not in result

    def test_resolve_preset_does_not_mutate_caller_args(self) -> None:
        caller = {"qr_hvh_t_base": 0.7}
        resolve_preset("creative_sampling", caller)
        assert caller == {"qr_hvh_t_base": 0.7}

    def test_non_qr_keys_passed_through(self) -> None:
        result = resolve_preset("creative_sampling", {"other_key": 42})
        assert result["other_key"] == 42


class TestExpandExtraArgs:
    """expand_extra_args(): preset resolution prioritization."""

    def test_no_preset_returns_extra_args_unchanged(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        extra = {"qr_top_k": 50}
        result = expand_extra_args(extra, defaults)
        assert result == {"qr_top_k": 50}

    def test_none_extra_args_with_no_preset_returns_empty(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert expand_extra_args(None, defaults) == {}

    def test_empty_string_preset_means_no_preset(self) -> None:
        """``QR_PRESET=""`` ingests as ``""`` -- must mean "no preset", not
        an ``Unknown preset ''`` error on every request (review fix)."""
        defaults = QRSamplerConfig(_env_file=None, preset="")  # type: ignore[call-arg]
        extra = {"qr_top_k": 50}
        assert expand_extra_args(extra, defaults) == {"qr_top_k": 50}
        assert expand_extra_args(None, defaults) == {}

    def test_env_var_preset_picked_up_via_default_config(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = expand_extra_args({}, defaults)
        assert result == _CREATIVE_EXPECTED

    def test_env_var_preset_applied_when_extra_args_is_none(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = expand_extra_args(None, defaults)
        assert result == _CREATIVE_EXPECTED

    def test_per_request_preset_beats_env_var(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = expand_extra_args({"qr_preset": "normal_t1"}, defaults)
        assert result == _NORMAL_T1_EXPECTED

    def test_unknown_preset_in_extra_args_raises(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ConfigValidationError, match="Unknown preset"):
            expand_extra_args({"qr_preset": "bogus"}, defaults)

    def test_env_var_preset_combined_with_extra_qr_args(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = expand_extra_args({"qr_top_k": 25}, defaults)
        # Preset's top_k (0) overridden by caller's qr_top_k=25 (FR-10).
        assert result["qr_top_k"] == 25
        assert result["qr_temperature_strategy"] == "hvh_drift"


class TestResolveConfigIntegration:
    """resolve_config() must invoke expand_extra_args before validation."""

    def test_resolve_config_invokes_expand_extra_args(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        result = resolve_config(defaults, {"qr_preset": "creative_sampling"})
        assert result.temperature_strategy == "hvh_drift"
        assert result.hvh_t_base == 1.35
        assert result.hvh_alpha_h == 0.3
        assert result.hvh_alpha_vh == -0.2
        assert result.hvh_gamma_dh == 1.0
        assert result.hvh_delta_dvh == 0.5
        assert result.hvh_lambda_ema == 0.02
        assert result.hvh_min_p_base == 0.025
        assert result.hvh_kappa_h == 0.03
        assert result.hvh_nu_dh == 0.02
        assert result.top_k == 0
        assert result.top_p == 1.0

    def test_resolve_config_normal_t1_preset(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        result = resolve_config(defaults, {"qr_preset": "normal_t1"})
        assert result.temperature_strategy == "fixed"
        assert result.fixed_temperature == 1.0
        assert result.top_k == 0
        assert result.top_p == 1.0

    def test_resolve_config_env_var_preset(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = resolve_config(defaults, None)
        assert result.temperature_strategy == "hvh_drift"
        assert result.hvh_t_base == 1.35

    def test_resolve_config_per_request_preset_overrides_env_var(self) -> None:
        defaults = QRSamplerConfig(  # type: ignore[call-arg]
            _env_file=None,
            preset="creative_sampling",
        )
        result = resolve_config(defaults, {"qr_preset": "normal_t1"})
        assert result.temperature_strategy == "fixed"
        assert result.fixed_temperature == 1.0

    def test_resolve_config_caller_override_wins_over_preset(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        result = resolve_config(
            defaults,
            {"qr_preset": "creative_sampling", "qr_hvh_t_base": 0.7},
        )
        assert result.temperature_strategy == "hvh_drift"
        assert result.hvh_t_base == 0.7

    def test_resolve_config_unknown_preset_raises(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ConfigValidationError, match="Unknown preset"):
            resolve_config(defaults, {"qr_preset": "bogus"})


class TestBuiltinPresetsShape:
    """Structural invariants for BUILTIN_PRESETS."""

    def test_has_creative_sampling_and_normal_t1(self) -> None:
        assert "creative_sampling" in BUILTIN_PRESETS
        assert "normal_t1" in BUILTIN_PRESETS

    def test_preset_keys_are_field_names_without_qr_prefix(self) -> None:
        for preset_name, overrides in BUILTIN_PRESETS.items():
            for key in overrides:
                assert not key.startswith("qr_"), (
                    f"BUILTIN_PRESETS[{preset_name!r}] key {key!r} must not "
                    f"include the qr_ prefix (it is added at resolution time)"
                )

    def test_all_preset_keys_are_valid_config_fields(self) -> None:
        valid_fields = set(QRSamplerConfig.model_fields.keys())
        for preset_name, overrides in BUILTIN_PRESETS.items():
            for key in overrides:
                assert key in valid_fields, (
                    f"BUILTIN_PRESETS[{preset_name!r}] references unknown field {key!r}"
                )


class TestChatLightPreset:
    """The lighter owui / external-caller lane (spec §4.1).

    Pins the lighter-lane contract: fresh quantum entropy into the sampler
    with NO coherence gate — the cross-device coherence statistic is a
    qthought scientific-lineage concern, not a general chatbot one.
    """

    def test_chat_light_is_a_builtin_preset(self) -> None:
        assert "chat_light" in BUILTIN_PRESETS

    def test_chat_light_resolves_and_uses_quantum_entropy(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        result = resolve_config(defaults, {"qr_preset": "chat_light"})
        assert result.entropy_source_type == "quantum_grpc"

    def test_chat_light_temperature_strategy_is_not_coherence_gate(self) -> None:
        """The lighter lane must never ride the coherence gate."""
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        result = resolve_config(defaults, {"qr_preset": "chat_light"})
        assert result.temperature_strategy != "coherence_gate"
        assert result.temperature_strategy == "fixed"

    def test_chat_light_uses_plain_local_amplifier_not_server_draw(self) -> None:
        """A plain amplifier (local byte fetch + z-score), not the
        server-integrated qr_purity draw the qthought* lanes use."""
        overrides = BUILTIN_PRESETS["chat_light"]
        assert overrides["signal_amplifier_type"] == "zscore_mean"
        # No server-draw / coherence keys leak into the lighter lane.
        assert "coherence_inner_strategy" not in overrides
        assert not any(key.startswith("coherence_") for key in overrides)
