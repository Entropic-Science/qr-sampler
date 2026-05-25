"""Configuration system for qr-sampler.

Uses pydantic-settings for declarative, layered configuration:
init kwargs -> environment variables (QR_*) -> .env file -> field defaults.

Per-request overrides are applied via resolve_config() which creates a new
config instance without mutating the defaults. Infrastructure fields are
protected from per-request override.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qr_sampler.exceptions import ConfigValidationError

# Fields that can be overridden per-request via SamplingParams.extra_args.
# Infrastructure fields (server address, timeout, retry, fallback mode, etc.)
# are deliberately excluded — they cannot change per-request.
_PER_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        "signal_amplifier_type",
        "sample_count",
        "population_mean",
        "population_std",
        "uniform_clamp_epsilon",
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
        "oe_conditioning",
        # Per-request switchable so comparison mode can fan out two requests
        # to the same engine instance with different entropy sources. The
        # engine adapter additionally constrains the allowed values at startup
        # to the set of entropy sources it has pre-initialised.
        "entropy_source_type",
        # HVH-Drift hyperparameters (V6_HVD_R01_01 winner from createmp-evalsuite).
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
    }
)

# All known config field names (populated after class definition).
_ALL_FIELDS: frozenset[str] = frozenset()


class QRSamplerConfig(BaseSettings):
    """Configuration for qr-sampler.

    Resolution order: init kwargs -> env vars (QR_*) -> .env file -> defaults.

    Fields are divided into two groups:
    - **Infrastructure**: Server addresses, timeouts, transport mode — NOT
      overridable per-request.
    - **Sampling parameters**: Amplification, temperature, selection, logging
      — overridable per-request via SamplingParams.extra_args with qr_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="QR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Infrastructure (NOT per-request overridable) ---

    grpc_server_address: str = Field(
        default="localhost:50051",
        description="gRPC entropy server address (host:port or unix:///path)",
    )
    grpc_timeout_ms: float = Field(
        default=5000.0,
        description="gRPC call timeout in milliseconds",
    )
    grpc_retry_count: int = Field(
        default=2,
        description="Number of retries after gRPC failure",
    )
    grpc_mode: str = Field(
        default="unary",
        description="gRPC transport mode: 'unary', 'server_streaming', 'bidi_streaming'",
    )
    grpc_method_path: str = Field(
        default="/qr_entropy.EntropyService/GetEntropy",
        description="gRPC method path for unary RPC (e.g. '/qrng.QuantumRNG/GetRandomBytes')",
    )
    grpc_stream_method_path: str = Field(
        default="/qr_entropy.EntropyService/StreamEntropy",
        description="gRPC method path for streaming RPC (empty string disables streaming modes)",
    )
    grpc_api_key: str = Field(
        default="",
        description="API key sent via gRPC metadata (empty = no auth)",
    )
    grpc_api_key_header: str = Field(
        default="api-key",
        description="gRPC metadata header name for the API key",
    )
    fallback_mode: str = Field(
        default="system",
        description="Fallback entropy source: 'error', 'system', 'mock_uniform'",
    )

    @field_validator("fallback_mode")
    @classmethod
    def _coerce_fallback_mode(cls, v: str) -> str:
        # Coerce unknown values (typos in QR_FALLBACK_MODE secret, etc.) to
        # 'system' and emit ONE warning at config-load time. Without this,
        # build_entropy_source warns per-pipeline at every preinit (the
        # QR_PREINIT_ENTROPY_SOURCES expansion builds N pipelines).
        if v not in {"error", "system", "mock_uniform"}:
            import warnings

            warnings.warn(
                f"Unknown fallback_mode {v!r}; coerced to 'system'",
                stacklevel=2,
            )
            return "system"
        return v

    entropy_source_type: str = Field(
        default="system",
        description="Primary entropy source identifier",
    )

    # --- Circuit Breaker (NOT per-request overridable) ---

    cb_window_size: int = Field(
        default=100,
        description="Rolling latency window size for P99 computation",
    )
    cb_min_timeout_ms: float = Field(
        default=5.0,
        description="Minimum adaptive timeout in milliseconds",
    )
    cb_timeout_multiplier: float = Field(
        default=1.5,
        description="Multiplier applied to P99 latency for adaptive timeout",
    )
    cb_recovery_window_s: float = Field(
        default=10.0,
        description="Seconds to wait before half-open retry after circuit opens",
    )
    cb_max_consecutive_failures: int = Field(
        default=3,
        description="Consecutive failures before circuit breaker opens",
    )

    # --- Signal Amplification (per-request overridable) ---

    signal_amplifier_type: str = Field(
        default="zscore_mean",
        description="Signal amplification algorithm",
    )
    sample_count: int = Field(
        # iter-48 (2026-05-25): halved from 20480 → 10000 to reduce
        # per-token QRNG bandwidth. The full 20-kB budget exceeded what
        # the amplifier needed to converge on z-score-mean at typical
        # vocabulary sizes; 10 kB still produces statistically clean
        # amplified signals while halving the per-token gRPC payload
        # (and the colocation-bound backbone RTT cost — see
        # ``qrng_colocation_constraint`` auto-memory).
        default=10000,
        description="Number of entropy bytes to fetch per token",
    )
    population_mean: float = Field(
        default=127.5,
        description="Null-hypothesis mean of byte values {0..255}",
    )
    population_std: float = Field(
        default=73.61215932167728,
        description="Population std for continuous uniform [0, 255]",
    )
    uniform_clamp_epsilon: float = Field(
        default=1e-10,
        description="Clamp u to (epsilon, 1-epsilon) to avoid degenerate CDF",
    )
    ecdf_calibration_samples: int = Field(
        default=2000,
        ge=100,
        description="Samples for ECDF calibration",
    )

    # --- Temperature Strategy (per-request overridable) ---

    temperature_strategy: str = Field(
        default="fixed",
        description="Temperature strategy: 'fixed' or 'edt'",
    )
    fixed_temperature: float = Field(
        default=1.0,
        description=(
            "Constant temperature for fixed strategy. Default is 1.0 — the "
            "true 'no temperature scaling' baseline (logits used as-is for "
            "quantum-driven softmax selection). The `creative_sampling` "
            "preset overrides this with the HVH-Drift dynamic strategy "
            "(temperature_strategy=hvh_drift, base 1.35 with per-token "
            "entropy/varentropy drift). Earlier default of 0.7 was inherited "
            "from generic LLM-sampling conventions but is wrong for a "
            "quantum-entropy baseline — sharpening the distribution makes "
            "the QRNG signal less load-bearing."
        ),
    )
    edt_base_temp: float = Field(
        default=0.8,
        description="Base coefficient for EDT",
    )
    edt_exponent: float = Field(
        default=0.5,
        description="Power-law exponent for EDT",
    )
    edt_min_temp: float = Field(
        default=0.1,
        description="EDT temperature floor",
    )
    edt_max_temp: float = Field(
        default=2.0,
        description="EDT temperature ceiling",
    )

    # --- HVH-Drift Temperature Strategy (per-request overridable) ---
    # Defaults pinned to V6_HVD_R01_01 winning configuration from
    # createmp-evalsuite (results/v6/round_final). Dormant unless
    # temperature_strategy = "hvh_drift" is explicitly set.

    hvh_t_base: float = Field(
        default=1.35,
        description="HVH-Drift base temperature (V6_HVD_R01_01 winner)",
    )
    hvh_alpha_h: float = Field(
        default=0.3,
        description="HVH-Drift entropy coefficient (V6_HVD_R01_01 winner)",
    )
    hvh_alpha_vh: float = Field(
        default=-0.2,
        description="HVH-Drift varentropy coefficient (V6_HVD_R01_01 winner)",
    )
    hvh_gamma_dh: float = Field(
        default=1.0,
        description="HVH-Drift entropy-drift coefficient (V6_HVD_R01_01 winner)",
    )
    hvh_delta_dvh: float = Field(
        default=0.5,
        description="HVH-Drift varentropy-drift coefficient (V6_HVD_R01_01 winner)",
    )
    hvh_lambda_ema: float = Field(
        default=0.02,
        description="HVH-Drift EMA decay rate for H/VH state (V6_HVD_R01_01 winner)",
    )
    hvh_min_p_base: float = Field(
        default=0.025,
        description="HVH-Drift min-p base term (V6_HVD_R01_01 winner)",
    )
    hvh_kappa_h: float = Field(
        default=0.03,
        description="HVH-Drift min-p entropy coefficient (V6_HVD_R01_01 winner)",
    )
    hvh_nu_dh: float = Field(
        default=0.02,
        description="HVH-Drift min-p entropy-drift coefficient (V6_HVD_R01_01 winner)",
    )

    # --- Token Selection (per-request overridable) ---

    top_k: int = Field(
        default=0,
        description="Top-k filtering (<=0 disables)",
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling threshold (1.0 disables)",
    )
    min_p_base: float = Field(
        default=0.0,
        description=(
            "Default min-p truncation threshold used by the selector when the "
            "active temperature strategy does not emit a per-token min_p. "
            "Default 0.0 disables min-p (preserves prior behavior; NFR-7)."
        ),
    )

    # --- Logging (per-request overridable) ---

    log_level: str = Field(
        default="summary",
        description="Logging verbosity: 'none', 'summary', 'full'",
    )
    diagnostic_mode: bool = Field(
        default=False,
        description="Store all token records in memory for analysis",
    )

    # --- Preset (env-var ingestion only; NOT per-request overridable) ---
    # Resolved by qr_sampler.presets.expand_extra_args, not via the normal
    # _PER_REQUEST_FIELDS merge path. Per-request callers use the
    # `qr_preset` key in extra_args directly.

    preset: str | None = Field(
        default=None,
        description=(
            "Optional preset name (e.g. 'creative_sampling', 'normal_t1') "
            "loaded from QR_PRESET. Expanded into field overrides at "
            "resolve_config() time; not a runtime sampling parameter."
        ),
    )

    # --- OpenEntropy (oe_conditioning per-request, others infrastructure) ---

    oe_conditioning: str = Field(
        default="raw",
        description="OpenEntropy conditioning mode: raw, sha256, vonneumann",
    )
    oe_sources: str = Field(
        default="",
        description="Comma-separated OpenEntropy source names. Empty = all available.",
    )
    oe_parallel: bool = Field(
        default=True,
        description="Collect OpenEntropy sources in parallel",
    )
    oe_timeout: float = Field(
        default=5.0,
        description="OpenEntropy collection timeout in seconds",
    )


# Populate _ALL_FIELDS now that the class is defined.
_ALL_FIELDS = frozenset(QRSamplerConfig.model_fields.keys())


def _strip_prefix(key: str) -> str:
    """Strip the 'qr_' prefix from an extra_args key.

    Args:
        key: The key with or without 'qr_' prefix.

    Returns:
        The key with 'qr_' prefix removed if present.
    """
    if key.startswith("qr_"):
        return key[3:]
    return key


def validate_extra_args(extra_args: dict[str, Any]) -> None:
    """Validate all qr_* keys in extra_args without creating a config.

    This is called by validate_params() at request creation time to
    reject bad keys early, before the request enters the batch.

    ``qr_preset`` is accepted here as a special case (the preset itself
    is not a per-request-overridable field, but selecting a preset *by
    name* is the supported per-request surface; resolve_config()
    expands it into concrete overrides before merging). The preset name
    is validated against ``BUILTIN_PRESETS`` so unknown names fail at
    the same point as unknown qr_* keys.

    Args:
        extra_args: Dictionary of extra arguments, potentially with qr_ prefix.

    Raises:
        ConfigValidationError: If any qr_* key is unknown or non-overridable,
            or if ``qr_preset`` names an unknown preset.
    """
    # Imported lazily to mirror resolve_config's import-cycle workaround.
    from qr_sampler.presets import BUILTIN_PRESETS

    for key in extra_args:
        if not key.startswith("qr_"):
            continue
        if key == "qr_preset":
            preset_name = extra_args[key]
            if preset_name not in BUILTIN_PRESETS:
                raise ConfigValidationError(
                    f"Unknown preset {preset_name!r}; known: {sorted(BUILTIN_PRESETS)}"
                )
            continue
        field_name = _strip_prefix(key)
        if field_name not in _ALL_FIELDS:
            raise ConfigValidationError(
                f"Unknown config field: '{key}' (no field '{field_name}' exists)"
            )
        if field_name not in _PER_REQUEST_FIELDS:
            raise ConfigValidationError(
                f"Field '{field_name}' is an infrastructure field and cannot be "
                f"overridden per-request via extra_args"
            )


def resolve_config(
    defaults: QRSamplerConfig,
    extra_args: dict[str, Any] | None,
) -> QRSamplerConfig:
    """Create a new config instance merging defaults with per-request overrides.

    The extra_args keys use 'qr_' prefix (e.g., 'qr_top_k': 100).
    Only fields in _PER_REQUEST_FIELDS are overridable. Keys without the
    'qr_' prefix are silently ignored (they belong to other processors).

    Preset expansion runs first: ``qr_preset`` in extra_args (or
    ``defaults.preset`` from QR_PRESET) is expanded into concrete
    ``qr_*`` overrides before the normal field-merge path.

    Args:
        defaults: The base configuration loaded from environment.
        extra_args: Per-request overrides from SamplingParams.extra_args.

    Returns:
        A new QRSamplerConfig with overrides applied.

    Raises:
        ConfigValidationError: If any qr_* key is unknown or non-overridable.
    """
    # Imported here to avoid a circular import (presets -> config).
    from qr_sampler.presets import expand_extra_args

    extra_args = expand_extra_args(extra_args, defaults)
    if not extra_args:
        return defaults

    # Validate all qr_* keys first.
    validate_extra_args(extra_args)

    # Extract and apply valid overrides.
    overrides: dict[str, Any] = {}
    for key, value in extra_args.items():
        if not key.startswith("qr_"):
            continue
        field_name = _strip_prefix(key)
        overrides[field_name] = value

    if not overrides:
        return defaults

    # Use model_validate on a merged dict to ensure type coercion.
    # model_copy(update=...) skips validation, so string "100" would not
    # be coerced to int 100. model_validate runs the full validator.
    merged = defaults.model_dump()
    merged.update(overrides)
    return QRSamplerConfig.model_validate(merged)
