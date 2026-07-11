"""Tests for qr_sampler.config module.

Covers: default values, env var loading, per-request resolution,
non-overridable field rejection, validate_extra_args, type coercion,
and invalid key detection.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from qr_sampler.config import (
    ALL_FIELDS,
    PER_REQUEST_FIELDS,
    QRSamplerConfig,
    resolve_config,
    validate_extra_args,
)
from qr_sampler.exceptions import ConfigValidationError

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """Verify all fields have the expected default values."""

    def test_infrastructure_defaults(self, default_config: QRSamplerConfig) -> None:
        assert default_config.grpc_server_address == "localhost:50051"
        assert default_config.grpc_timeout_ms == 5000.0
        assert default_config.grpc_retry_count == 2
        assert default_config.grpc_mode == "unary"
        assert default_config.grpc_method_path == "/qr_entropy.EntropyService/GetEntropy"
        assert default_config.grpc_stream_method_path == "/qr_entropy.EntropyService/StreamEntropy"
        assert default_config.grpc_api_key == ""
        assert default_config.grpc_api_key_header == "api-key"
        assert default_config.fallback_mode == "system"
        assert default_config.entropy_source_type == "system"

    def test_amplification_defaults(self, default_config: QRSamplerConfig) -> None:
        assert default_config.signal_amplifier_type == "zscore_mean"
        assert default_config.sample_count == 10000  # iter-48 default
        assert default_config.population_mean == 127.5
        assert default_config.population_std == pytest.approx(73.61215932167728)
        assert default_config.uniform_clamp_epsilon == 1e-10

    def test_temperature_defaults(self, default_config: QRSamplerConfig) -> None:
        assert default_config.temperature_strategy == "fixed"
        # 1.0 = the true no-scaling quantum baseline (see config.py — the
        # earlier 0.7 default sharpened the distribution).
        assert default_config.fixed_temperature == 1.0
        assert default_config.edt_base_temp == 0.8
        assert default_config.edt_exponent == 0.5
        assert default_config.edt_min_temp == 0.1
        assert default_config.edt_max_temp == 2.0

    def test_selection_defaults(self, default_config: QRSamplerConfig) -> None:
        assert default_config.top_k == 0
        assert default_config.top_p == 1.0

    def test_logging_defaults(self, default_config: QRSamplerConfig) -> None:
        assert default_config.log_level == "summary"
        assert default_config.diagnostic_mode is False

    def test_hvh_fields_have_v6_winner_defaults(self, default_config: QRSamplerConfig) -> None:
        """All hvh_* fields default to the V6_HVD_R01_01 winning configuration."""
        assert default_config.hvh_t_base == 1.35
        assert default_config.hvh_alpha_h == 0.3
        assert default_config.hvh_alpha_vh == -0.2
        assert default_config.hvh_gamma_dh == 1.0
        assert default_config.hvh_delta_dvh == 0.5
        assert default_config.hvh_lambda_ema == 0.02
        assert default_config.hvh_min_p_base == 0.025
        assert default_config.hvh_kappa_h == 0.03
        assert default_config.hvh_nu_dh == 0.02

    def test_min_p_base_defaults_to_zero(self, default_config: QRSamplerConfig) -> None:
        """min_p_base defaults to 0.0 so the selector remains a no-op (NFR-7)."""
        assert default_config.min_p_base == 0.0

    def test_preset_defaults_to_none(self, default_config: QRSamplerConfig) -> None:
        """preset defaults to None so no preset is applied unless QR_PRESET is set."""
        assert default_config.preset is None


# ---------------------------------------------------------------------------
# Environment variable loading
# ---------------------------------------------------------------------------


class TestEnvVarLoading:
    """Verify that QR_* env vars are picked up correctly."""

    def test_string_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_SERVER_ADDRESS": "myhost:9999"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_server_address == "myhost:9999"

    def test_float_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_TIMEOUT_MS": "1234.5"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_timeout_ms == 1234.5

    def test_int_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_SAMPLE_COUNT": "100"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.sample_count == 100

    def test_bool_env_var_true(self) -> None:
        with patch.dict(os.environ, {"QR_DIAGNOSTIC_MODE": "true"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.diagnostic_mode is True

    def test_bool_env_var_false(self) -> None:
        with patch.dict(os.environ, {"QR_DIAGNOSTIC_MODE": "false"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.diagnostic_mode is False

    def test_multiple_env_vars(self) -> None:
        env = {
            "QR_TOP_K": "100",
            "QR_TOP_P": "0.95",
            "QR_TEMPERATURE_STRATEGY": "edt",
        }
        with patch.dict(os.environ, env):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.top_k == 100
        assert config.top_p == 0.95
        assert config.temperature_strategy == "edt"

    def test_grpc_method_path_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_METHOD_PATH": "/qrng.QuantumRNG/GetRandomBytes"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_method_path == "/qrng.QuantumRNG/GetRandomBytes"

    def test_grpc_stream_method_path_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_STREAM_METHOD_PATH": ""}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_stream_method_path == ""

    def test_grpc_api_key_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_API_KEY": "test-key-123"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_api_key == "test-key-123"

    def test_grpc_api_key_header_env_var(self) -> None:
        with patch.dict(os.environ, {"QR_GRPC_API_KEY_HEADER": "authorization"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.grpc_api_key_header == "authorization"

    def test_non_qr_env_vars_ignored(self) -> None:
        with patch.dict(os.environ, {"OTHER_TOP_K": "999"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.top_k == 0  # Unchanged default

    def test_qr_hvh_t_base_env_var(self) -> None:
        """QR_HVH_T_BASE is auto-bound by pydantic-settings."""
        with patch.dict(os.environ, {"QR_HVH_T_BASE": "2.0"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.hvh_t_base == 2.0

    def test_qr_preset_env_var(self) -> None:
        """QR_PRESET populates the preset field on the config."""
        with patch.dict(os.environ, {"QR_PRESET": "creative_sampling"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.preset == "creative_sampling"


# ---------------------------------------------------------------------------
# Per-request config resolution
# ---------------------------------------------------------------------------


class TestResolveConfig:
    """Verify resolve_config creates correct new configs from extra_args."""

    def test_no_extra_args_returns_same(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(default_config, None)
        assert result is default_config

    def test_empty_extra_args_returns_same(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(default_config, {})
        assert result is default_config

    def test_non_qr_keys_ignored(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(default_config, {"other_key": 42})
        assert result is default_config

    def test_single_override(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(default_config, {"qr_top_k": 100})
        assert result.top_k == 100
        assert result is not default_config

    def test_multiple_overrides(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(
            default_config,
            {
                "qr_top_k": 100,
                "qr_top_p": 0.95,
                "qr_fixed_temperature": 1.0,
            },
        )
        assert result.top_k == 100
        assert result.top_p == 0.95
        assert result.fixed_temperature == 1.0

    def test_original_unchanged(self, default_config: QRSamplerConfig) -> None:
        resolve_config(default_config, {"qr_top_k": 100})
        assert default_config.top_k == 0  # Original unchanged

    def test_mixed_qr_and_non_qr_keys(self, default_config: QRSamplerConfig) -> None:
        result = resolve_config(
            default_config,
            {
                "qr_top_k": 100,
                "other_key": 42,
            },
        )
        assert result.top_k == 100

    def test_all_per_request_fields_overridable(self, default_config: QRSamplerConfig) -> None:
        """Every field in PER_REQUEST_FIELDS should be overridable."""
        for field_name in PER_REQUEST_FIELDS:
            key = f"qr_{field_name}"
            # Use a value that's different from default
            field_info = QRSamplerConfig.model_fields[field_name]
            default_val = field_info.default
            if isinstance(default_val, bool):
                override_val: Any = not default_val
            elif isinstance(default_val, int):
                override_val = default_val + 1
            elif isinstance(default_val, float):
                # +0.1 can cross a field's declared upper bound (e.g. the
                # gdt_t_peak <= 1.5 coherence-cliff cap); fall back to -0.1.
                override_val = default_val + 0.1
                try:
                    QRSamplerConfig.model_validate(
                        {**default_config.model_dump(), field_name: override_val}
                    )
                except Exception:
                    override_val = default_val - 0.1
            elif isinstance(default_val, str):
                override_val = default_val + "_test"
            else:
                continue

            result = resolve_config(default_config, {key: override_val})
            assert getattr(result, field_name) == override_val, f"Failed to override {field_name}"

    def test_type_coercion_string_to_int(self, default_config: QRSamplerConfig) -> None:
        """Pydantic should coerce '100' to int 100."""
        result = resolve_config(default_config, {"qr_top_k": "100"})
        assert result.top_k == 100
        assert isinstance(result.top_k, int)

    def test_type_coercion_string_to_float(self, default_config: QRSamplerConfig) -> None:
        """Pydantic should coerce '0.95' to float 0.95."""
        result = resolve_config(default_config, {"qr_top_p": "0.95"})
        assert result.top_p == 0.95

    def test_type_coercion_string_to_bool(self, default_config: QRSamplerConfig) -> None:
        """Pydantic should coerce 'true' to True."""
        result = resolve_config(default_config, {"qr_diagnostic_mode": "true"})
        assert result.diagnostic_mode is True

    def test_type_coercion_string_to_bool_bypass(self, default_config: QRSamplerConfig) -> None:
        """Pydantic should coerce 'true' to True for qr_bypass."""
        result = resolve_config(default_config, {"qr_bypass": "true"})
        assert result.bypass is True

    def test_bypass_defaults_false_and_overridable(self, default_config: QRSamplerConfig) -> None:
        """bypass defaults False (bare requests never bypass) and is
        per-request overridable via qr_bypass."""
        assert default_config.bypass is False
        result = resolve_config(default_config, {"qr_bypass": True})
        assert result.bypass is True
        assert default_config.bypass is False  # defaults unchanged

    def test_per_request_override_hvh_field(self, default_config: QRSamplerConfig) -> None:
        """An hvh_* hyperparameter is overridable per-request and defaults stay clean."""
        result = resolve_config(default_config, {"qr_hvh_t_base": 1.5})
        assert result.hvh_t_base == 1.5
        assert default_config.hvh_t_base == 1.35  # defaults unchanged

    def test_min_p_base_per_request_override(self, default_config: QRSamplerConfig) -> None:
        """min_p_base is overridable per-request via qr_min_p_base."""
        result = resolve_config(default_config, {"qr_min_p_base": 0.05})
        assert result.min_p_base == 0.05
        assert default_config.min_p_base == 0.0  # defaults unchanged

    def test_preset_not_in_per_request_fields(self, default_config: QRSamplerConfig) -> None:
        """preset is NOT per-request overridable via the normal field-merge path.

        ``preset`` itself is an infrastructure-only field (set by QR_PRESET),
        so it must never appear in ``PER_REQUEST_FIELDS``. The selection
        surface uses ``qr_preset`` as a special key that ``resolve_config``
        and ``validate_extra_args`` both recognize via the preset-resolution
        layer (it expands into concrete qr_* overrides before merging).
        """
        assert "preset" not in PER_REQUEST_FIELDS
        # Known preset names must be accepted (the validation hook needs to
        # let qr_preset through so the vLLM per-request preset flow works).
        validate_extra_args({"qr_preset": "creative_sampling"})
        validate_extra_args({"qr_preset": "normal_t1"})
        # Unknown preset names must be rejected with a helpful message.
        with pytest.raises(ConfigValidationError, match="Unknown preset"):
            validate_extra_args({"qr_preset": "not_a_real_preset"})


# ---------------------------------------------------------------------------
# Non-overridable field rejection
# ---------------------------------------------------------------------------


class TestNonOverridableFields:
    """Verify that infrastructure fields cannot be overridden per-request."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "grpc_server_address",
            "grpc_timeout_ms",
            "grpc_retry_count",
            "grpc_mode",
            "grpc_method_path",
            "grpc_stream_method_path",
            "grpc_api_key",
            "grpc_api_key_header",
            "fallback_mode",
            "oe_conditioning",
            "oe_sources",
            "oe_parallel",
            "oe_timeout",
        ],
    )
    def test_infrastructure_field_rejected(
        self, default_config: QRSamplerConfig, field_name: str
    ) -> None:
        key = f"qr_{field_name}"
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            resolve_config(default_config, {key: "some_value"})

    @pytest.mark.parametrize(
        "field_name",
        [
            "grpc_server_address",
            "grpc_timeout_ms",
            "grpc_retry_count",
            "grpc_mode",
            "grpc_method_path",
            "grpc_stream_method_path",
            "grpc_api_key",
            "grpc_api_key_header",
            "fallback_mode",
            "oe_conditioning",
            "oe_sources",
            "oe_parallel",
            "oe_timeout",
        ],
    )
    def test_infrastructure_field_rejected_in_validate(self, field_name: str) -> None:
        key = f"qr_{field_name}"
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({key: "some_value"})


# ---------------------------------------------------------------------------
# validate_extra_args
# ---------------------------------------------------------------------------


class TestValidateExtraArgs:
    """Test the standalone validation function."""

    def test_valid_keys_pass(self) -> None:
        validate_extra_args({"qr_top_k": 100, "qr_top_p": 0.95})

    def test_non_qr_keys_ignored(self) -> None:
        validate_extra_args({"other_key": 42, "another": "value"})

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ConfigValidationError, match="Unknown config field"):
            validate_extra_args({"qr_nonexistent_field": 42})

    def test_empty_args_pass(self) -> None:
        validate_extra_args({})

    def test_mixed_valid_and_non_qr_keys(self) -> None:
        validate_extra_args({"qr_top_k": 100, "other": 42})


# ---------------------------------------------------------------------------
# Field set consistency
# ---------------------------------------------------------------------------


class TestFieldSets:
    """Verify internal field sets are consistent."""

    def test_per_request_fields_are_subset_of_all(self) -> None:
        assert PER_REQUEST_FIELDS <= ALL_FIELDS

    def test_infrastructure_fields_not_in_per_request(self) -> None:
        infra_fields = ALL_FIELDS - PER_REQUEST_FIELDS
        assert "grpc_server_address" in infra_fields
        assert "grpc_timeout_ms" in infra_fields
        assert "grpc_retry_count" in infra_fields
        assert "grpc_mode" in infra_fields
        assert "grpc_method_path" in infra_fields
        assert "grpc_stream_method_path" in infra_fields
        assert "grpc_api_key" in infra_fields
        assert "grpc_api_key_header" in infra_fields
        assert "fallback_mode" in infra_fields

    def test_entropy_source_type_is_per_request(self) -> None:
        """entropy_source_type is per-request overridable so comparison mode
        can route requests to different entropy sources on the same engine."""
        assert "entropy_source_type" in PER_REQUEST_FIELDS

    def test_all_fields_populated(self) -> None:
        """ALL_FIELDS should contain every model field."""
        assert frozenset(QRSamplerConfig.model_fields.keys()) == ALL_FIELDS

    def test_per_request_fields_derived_set_is_exactly_the_intended_set(self) -> None:
        """PER_REQUEST_FIELDS is derived from field metadata; this pin makes
        a metadata mistake (missing/extra ``per_request`` marker) fail loudly."""
        intended = frozenset(
            {
                "signal_amplifier_type",
                "sample_count",
                "population_mean",
                "population_std",
                "uniform_clamp_epsilon",
                "zscore_calibration_samples",
                "temperature_strategy",
                "fixed_temperature",
                "edt_base_temp",
                "edt_exponent",
                "edt_min_temp",
                "edt_max_temp",
                "top_k",
                "top_p",
                "log_level",
                "diagnostic_mode",
                "entropy_source_type",
                "entropy_prefetch",
                "hvh_t_base",
                "hvh_alpha_h",
                "hvh_alpha_vh",
                "hvh_gamma_dh",
                "hvh_delta_dvh",
                "hvh_lambda_ema",
                "hvh_min_p_base",
                "hvh_kappa_h",
                "hvh_nu_dh",
                "min_p_base",
                "draw_source_id",
                "draw_block_bytes",
                "coherence_threshold",
                "coherence_t_boost_max",
                "coherence_ema_alpha",
                "coherence_inner_strategy",
                "tt_t_base",
                "tt_gamma",
                "tt_min_p_base",
                "tt_min_p_scale",
                "evdt_t_base",
                "evdt_alpha",
                "evdt_beta",
                "evdt_min_p_base",
                "evdt_min_p_scale",
                "evdt_min_p_vh",
                "gdt_t_base",
                "gdt_t_peak",
                "gdt_mu",
                "gdt_sigma",
                "gdt_alpha",
                "gdt_lambda_vh",
                "gdt_min_p_base",
                "gdt_min_p_scale",
                "dynatemp_t_center",
                "dynatemp_t_range",
                "dynatemp_exponent",
                "dynatemp_min_p",
                "belltemp_t_base",
                "belltemp_t_peak",
                "belltemp_mu",
                "belltemp_sigma",
                "belltemp_vh_weight",
                "belltemp_lambda_vh",
                "belltemp_min_p_base",
                "belltemp_min_p_scale",
                "mix_t_cool",
                "mix_t_hot",
                "mix_gate_a",
                "mix_gate_b",
                "mix_gate_c",
                "mix_gate_d",
                "mix_min_p",
                "rba_buffer_n",
                "rba_lam",
                "rba_threshold",
                "rba_t",
                "rba_min_p",
                "truncate_first",
                "bypass",
            }
        )
        assert intended == PER_REQUEST_FIELDS

    def test_qrng_quota_defaults(self, default_config: QRSamplerConfig) -> None:
        """QRNG service quota limits are config fields with the documented
        service limits as defaults (deploy config, not code constants)."""
        assert default_config.qrng_max_bytes_per_request == 35_200
        assert default_config.qrng_max_requests_per_minute == 500
        assert default_config.qrng_max_bytes_per_day == 500 * 1024 * 1024
        assert "qrng_max_bytes_per_request" not in PER_REQUEST_FIELDS


# ---------------------------------------------------------------------------
# Init kwargs override
# ---------------------------------------------------------------------------


class TestInitKwargs:
    """Verify that constructor kwargs take highest priority."""

    def test_init_kwargs_override_defaults(self) -> None:
        config = QRSamplerConfig(top_k=200, _env_file=None)  # type: ignore[call-arg]
        assert config.top_k == 200

    def test_init_kwargs_override_env_vars(self) -> None:
        with patch.dict(os.environ, {"QR_TOP_K": "100"}):
            config = QRSamplerConfig(top_k=200, _env_file=None)  # type: ignore[call-arg]
        assert config.top_k == 200


# ---------------------------------------------------------------------------
# Extra="ignore" behavior
# ---------------------------------------------------------------------------


class TestExtraIgnore:
    """Verify that unknown fields in constructor are ignored."""

    def test_unknown_kwargs_ignored(self) -> None:
        config = QRSamplerConfig(
            unknown_field="value",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert not hasattr(config, "unknown_field")


# ---------------------------------------------------------------------------
# Model copy behavior
# ---------------------------------------------------------------------------


class TestModelCopy:
    """Verify that model_copy creates independent instances."""

    def test_model_copy_creates_new_instance(self, default_config: QRSamplerConfig) -> None:
        copy = default_config.model_copy(update={"top_k": 200})
        assert copy is not default_config
        assert copy.top_k == 200
        assert default_config.top_k == 0

    def test_model_copy_preserves_unmodified(self, default_config: QRSamplerConfig) -> None:
        copy = default_config.model_copy(update={"top_k": 200})
        assert copy.top_p == default_config.top_p
        assert copy.fixed_temperature == default_config.fixed_temperature
        assert copy.grpc_server_address == default_config.grpc_server_address


# ---------------------------------------------------------------------------
# OpenEntropy config fields
# ---------------------------------------------------------------------------


class TestOpenEntropyConfigFields:
    """Verify OpenEntropy config fields have correct defaults and overrides."""

    def test_oe_defaults(self, default_config: QRSamplerConfig) -> None:
        """Verify OpenEntropy field defaults."""
        assert default_config.oe_conditioning == "raw"
        assert default_config.oe_sources == ""
        assert default_config.oe_parallel is True
        assert default_config.oe_timeout == 5.0

    def test_oe_conditioning_env_var(self) -> None:
        """Verify QR_OE_CONDITIONING env var is loaded."""
        with patch.dict(os.environ, {"QR_OE_CONDITIONING": "sha256"}):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.oe_conditioning == "sha256"

    def test_oe_conditioning_infra_locked(self) -> None:
        """oe_conditioning is read only at source construction — a per-request
        override was a silent no-op falsifying config_hash provenance, so it
        is rejected outright (behavior-change ledger #1)."""
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_oe_conditioning": "vonneumann"})

    def test_oe_sources_infra_locked(self) -> None:
        """Verify oe_sources cannot be overridden per-request."""
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_oe_sources": "clock_jitter"})

    def test_oe_parallel_infra_locked(self) -> None:
        """Verify oe_parallel cannot be overridden per-request."""
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_oe_parallel": "false"})

    def test_oe_timeout_infra_locked(self) -> None:
        """Verify oe_timeout cannot be overridden per-request."""
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_oe_timeout": "10.0"})


# ---------------------------------------------------------------------------
# Named entropy-source instances (entropy_source_instances)
# ---------------------------------------------------------------------------


class TestEntropySourceInstances:
    """Validation of the named entropy-source instances infrastructure field."""

    def test_default_is_empty(self, default_config: QRSamplerConfig) -> None:
        assert default_config.entropy_source_instances == {}

    def test_valid_instances_accepted(self) -> None:
        config = QRSamplerConfig(
            entropy_source_instances={
                "qbert_prng_uniform": {
                    "type": "quantum_grpc",
                    "grpc_api_key": "key-a",
                },
                "qbert_prng_markov": {
                    "type": "quantum_grpc",
                    "grpc_api_key": "key-b",
                    "grpc_server_address": "unix:///run/qbert0g/qbert0g.sock",
                },
            },
            _env_file=None,  # type: ignore[call-arg]
        )
        assert set(config.entropy_source_instances) == {
            "qbert_prng_uniform",
            "qbert_prng_markov",
        }

    def test_env_var_json_loading(self) -> None:
        env = {
            "QR_ENTROPY_SOURCE_INSTANCES": (
                '{"qbert_prng_uniform":{"type":"quantum_grpc","grpc_api_key":"k"}}'
            )
        }
        with patch.dict(os.environ, env):
            config = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        assert config.entropy_source_instances == {
            "qbert_prng_uniform": {"type": "quantum_grpc", "grpc_api_key": "k"}
        }

    def test_unknown_override_key_rejected(self) -> None:
        """Only the conservative infrastructure allowlist may be overridden."""
        with pytest.raises(ConfigValidationError, match="outside the allowlist"):
            QRSamplerConfig(
                entropy_source_instances={"lane": {"type": "quantum_grpc", "sample_count": 5000}},
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_sampling_field_never_allowlisted(self) -> None:
        """The allowlist stays infrastructure-only (transport + timeout/retry)."""
        from qr_sampler.config import ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST

        assert {
            "grpc_server_address",
            "grpc_api_key",
            "grpc_mode",
            "grpc_timeout_ms",
            "grpc_retry_count",
        } == ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST

    def test_instance_name_shadowing_builtin_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="shadows a registered"):
            QRSamplerConfig(
                entropy_source_instances={"system": {"type": "quantum_grpc"}},
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_missing_type_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="must declare 'type'"):
            QRSamplerConfig(
                entropy_source_instances={"lane": {"grpc_api_key": "k"}},
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ConfigValidationError, match="must declare 'type'"):
            QRSamplerConfig(
                entropy_source_instances={"lane": {"type": "no_such_source"}},
                _env_file=None,  # type: ignore[call-arg]
            )

    def test_not_per_request_overridable(self) -> None:
        """Instances are infrastructure — rejected in per-request extra_args."""
        assert "entropy_source_instances" not in PER_REQUEST_FIELDS
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_entropy_source_instances": {"lane": {"type": "system"}}})

    def test_instance_name_is_valid_per_request_source_type(self) -> None:
        """qr_entropy_source_type accepts instance names (a plain string
        field); the adapter constrains it to pre-initialised names."""
        defaults = QRSamplerConfig(
            entropy_source_instances={"qbert_prng_uniform": {"type": "quantum_grpc"}},
            _env_file=None,  # type: ignore[call-arg]
        )
        resolved = resolve_config(defaults, {"qr_entropy_source_type": "qbert_prng_uniform"})
        assert resolved.entropy_source_type == "qbert_prng_uniform"
        # The instances declaration itself rides along unchanged.
        assert resolved.entropy_source_instances == defaults.entropy_source_instances
