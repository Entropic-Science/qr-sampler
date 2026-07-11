"""The declarative configuration model for qr-sampler.

Uses pydantic-settings for layered configuration:
init kwargs -> environment variables (QR_*) -> .env file -> field defaults.

Fields are divided into two groups:

- **Infrastructure**: server addresses, timeouts, transport mode, quotas —
  NOT overridable per-request.
- **Sampling parameters**: amplification, temperature, selection, logging —
  overridable per-request via ``SamplingParams.extra_args`` with a ``qr_``
  prefix. A field is per-request if and only if it is declared with
  ``Field(json_schema_extra={"per_request": True})``; the
  :data:`PER_REQUEST_FIELDS` set is DERIVED from that metadata at import
  time, so the declaration on the field is the single source of truth.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from qr_sampler.exceptions import ConfigValidationError

#: Marker for per-request-overridable fields (see module docstring).
#: ``dict[str, Any]`` keeps it assignable to pydantic's invariant JsonDict.
_PER_REQUEST: dict[str, Any] = {"per_request": True}

#: Infrastructure fields a named entropy-source instance may override.
#: Deliberately conservative: transport address, credentials, transport
#: mode, and timeout/retry — nothing that changes sampling semantics.
ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "grpc_server_address",
        "grpc_api_key",
        "grpc_mode",
        "grpc_timeout_ms",
        "grpc_retry_count",
    }
)


class QRSamplerConfig(BaseSettings):
    """Configuration for qr-sampler.

    Resolution order: init kwargs -> env vars (QR_*) -> .env file -> defaults.

    Per-request overrides are applied via ``resolve_config()`` which creates
    a new config instance without mutating the defaults. Infrastructure
    fields are protected from per-request override.
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
        description=(
            "Number of retries after gRPC failure (1 + grpc_retry_count "
            "total attempts). Retries operate on the already-warmed-up "
            "channel established by ``QuantumGrpcSource.warmup()`` at "
            "engine startup, so each retry is a single round-trip on a "
            "known-good connection — no channel-establishment cost is "
            "paid mid-fetch. The circuit breaker still trips on "
            "consecutive failures and the TCP pre-probe fast-fails when "
            "the tunnel to the entropy server is down."
        ),
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
    grpc_draw_method_path: str = Field(
        default="/qr_purity.PurityService/GetDraw",
        description=(
            "gRPC method path for the unary server-integrated draw RPC "
            "(empty string disables the draw handle)"
        ),
    )
    grpc_draw_stream_method_path: str = Field(
        default="/qr_purity.PurityService/StreamDraws",
        description=(
            "gRPC method path for the bidi-streaming server-integrated draw "
            "RPC (empty string disables the draw stream handle)"
        ),
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

    entropy_source_instances: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Named entropy-source instances: instance_name -> {'type': "
            "<builtin source type>, <infrastructure overrides>}. Lets one "
            "engine pre-initialise several pipelines of the SAME source "
            "type with different transport config (e.g. two quantum_grpc "
            "lanes whose API keys bind to different Qbert0G devices), each "
            "selectable per-request via qr_entropy_source_type. Override "
            "keys are restricted to ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST. "
            "Env: QR_ENTROPY_SOURCE_INSTANCES (JSON). Infrastructure — "
            "NOT per-request overridable."
        ),
    )

    @field_validator("entropy_source_instances")
    @classmethod
    def _validate_entropy_source_instances(
        cls, v: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Validate instance declarations loudly at config-construction time.

        Raises :class:`~qr_sampler.exceptions.ConfigValidationError` (NOT a
        pydantic ``ValidationError``) so a bad ``QR_ENTROPY_SOURCE_INSTANCES``
        fails startup with the same exception type the rest of the config
        surface uses. Checks:

        - instance names must not shadow a registered source type;
        - every instance must declare ``type`` naming a registered source;
        - override keys are restricted to
          :data:`ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST`.
        """
        if not v:
            return v
        # Lazy import: the registry module is import-light (its builtin
        # table is resolved on demand), and importing it here rather than
        # at module top keeps config.model free of an entropy-package
        # import edge at import time.
        from qr_sampler.entropy.registry import EntropySourceRegistry

        known_types = set(EntropySourceRegistry.list_available())
        for name, spec in v.items():
            if name in known_types:
                raise ConfigValidationError(
                    f"Entropy-source instance name {name!r} shadows a registered "
                    f"source type. Instance names must be distinct from source "
                    f"types: {sorted(known_types)}"
                )
            source_type = spec.get("type")
            if not isinstance(source_type, str) or source_type not in known_types:
                raise ConfigValidationError(
                    f"Entropy-source instance {name!r} must declare 'type' naming "
                    f"a registered source type (got {source_type!r}). "
                    f"Available: {sorted(known_types)}"
                )
            bad_keys = sorted(set(spec) - {"type"} - ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST)
            if bad_keys:
                raise ConfigValidationError(
                    f"Entropy-source instance {name!r} carries override keys "
                    f"outside the allowlist: {bad_keys}. Allowed overrides: "
                    f"{sorted(ENTROPY_INSTANCE_OVERRIDE_ALLOWLIST)}"
                )
        return v

    entropy_source_type: str = Field(
        default="system",
        description=(
            "Primary entropy source identifier — a registered source type "
            "or a declared entropy_source_instances name. Per-request "
            "switchable so comparison mode can fan out two requests to the "
            "same engine instance with different entropy sources. The "
            "engine adapter additionally constrains the allowed values at "
            "startup to the set of entropy sources it has pre-initialised."
        ),
        json_schema_extra=_PER_REQUEST,
    )

    entropy_prefetch: bool = Field(
        default=True,
        description=(
            "Pipeline the per-token entropy fetch (commit-then-fetch): the "
            "gRPC request for token N+1 is fired the instant token N is "
            "selected, so the network round trip overlaps the engine's next "
            "forward pass instead of serializing behind it. The causal "
            "contract is preserved — physical generation still happens "
            "strictly AFTER the previous token's selection, and the request "
            "carries a commitment nonce derived from that token (echoed by "
            "the server via sequence_id) so the ordering is externally "
            "verifiable. Set QR_ENTROPY_PREFETCH=0 to restore the "
            "strictly-serial fetch-after-logits timing. Timing-only switch "
            "(does not affect the sampled distribution): per-request "
            "override lets an operator A/B the pipelined vs serial fetch "
            "latency on a live deployment."
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- QRNG service quotas (NOT per-request overridable) ---
    # Documented limits of the QRNG service for our API key (QRNG team
    # README, 2026-06-10; adjustable on request to the QRNG team).
    # Exceeding any of them returns gRPC RESOURCE_EXHAUSTED — a quota
    # verdict, not a connectivity one; QuantumGrpcSource gives it a distinct
    # telemetry event so the operator response is "lower sample_count /
    # concurrency or ask for a bigger quota", never "go check the tunnel".
    #
    # Request-rate math worth keeping in view: the just-in-time
    # post-selection contract pins entropy fetches at exactly ONE request
    # per generated token (coalescing N tokens' bytes into one request
    # would fetch token N+1's entropy before token N is committed —
    # breaking the experiment's causal ordering, so it is deliberately not
    # an optimisation we will ever take). At 500 requests/minute that caps
    # aggregate decode throughput at ~8.3 tokens/sec across ALL concurrent
    # sequences.

    qrng_max_bytes_per_request: int = Field(
        default=35_200,
        description="QRNG service per-request byte quota (RESOURCE_EXHAUSTED beyond it)",
    )
    qrng_max_requests_per_minute: int = Field(
        default=500,
        description="QRNG service request-rate quota per minute",
    )
    qrng_max_bytes_per_day: int = Field(
        default=500 * 1024 * 1024,
        description="QRNG service daily byte quota",
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
        description=(
            "BASE seconds to wait before the FIRST half-open retry after the "
            "circuit opens. Subsequent opens without an intervening success "
            "back off exponentially (base x 2^opens) up to "
            "cb_recovery_window_max_s. Keeping the base short lets a transient "
            "post-wake stale channel recover in one cycle; the backoff stops a "
            "sustained outage (QRNG origin down for minutes) from being probed "
            "every base-seconds, which would hammer a dead server with channel "
            "resets + reconnects."
        ),
    )
    cb_recovery_window_max_s: float = Field(
        default=60.0,
        description=(
            "Ceiling for the exponentially-backed-off recovery window. Caps "
            "half-open reconnect attempts at ~one per this many seconds during "
            "a sustained QRNG outage. Trade-off: once the QRNG recovers, the "
            "circuit can stay open for up to this long before the next "
            "half-open re-engages the primary (bounded extra PRNG fallback)."
        ),
    )
    cb_max_consecutive_failures: int = Field(
        default=3,
        description="Consecutive failures before circuit breaker opens",
    )

    # --- Signal Amplification (per-request overridable) ---

    signal_amplifier_type: str = Field(
        default="zscore_mean",
        description="Signal amplification algorithm",
        json_schema_extra=_PER_REQUEST,
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
        json_schema_extra=_PER_REQUEST,
    )
    population_mean: float = Field(
        default=127.5,
        description="Null-hypothesis mean of byte values {0..255}",
        json_schema_extra=_PER_REQUEST,
    )
    population_std: float = Field(
        default=73.61215932167728,
        description="Population std for continuous uniform [0, 255]",
        json_schema_extra=_PER_REQUEST,
    )
    uniform_clamp_epsilon: float = Field(
        default=1e-10,
        description="Clamp u to (epsilon, 1-epsilon) to avoid degenerate CDF",
        json_schema_extra=_PER_REQUEST,
    )
    ecdf_calibration_samples: int = Field(
        default=2000,
        ge=100,
        description="Samples for ECDF calibration",
    )
    zscore_calibration_samples: int = Field(
        default=0,
        ge=0,
        description=(
            "Calibration blocks for the z-score amplifiers. 0 (default) keeps "
            "the ideal-population baseline (mean 127.5, uniform-byte std) — "
            "byte-identical to the historical behaviour. N > 0 draws N blocks "
            "of sample_count bytes from the entropy source at build time and "
            "replaces the baseline with the device's empirical block-mean and "
            "block-SEM, so a real device's static manufacturing bias reads as "
            "z ~ 0 instead of saturating the CDF into its clamp (the 'acorn' "
            "failure: every u pinned to an extreme selects the same lexicon "
            "index on every draw). Baseline correction, not censoring — the "
            "same rationale as the server-integrated draw path."
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- Server-integrated draws (per-request overridable) ---
    # Active only with signal_amplifier_type="server": the entropy server
    # integrates a raw block itself (qr_purity.PurityService) and returns
    # the uniform u directly, so both fields default to "defer to server".

    draw_source_id: str = Field(
        default="",
        description=(
            "Source id for server-integrated draws; '' defers to the server's API-key binding"
        ),
        json_schema_extra=_PER_REQUEST,
    )
    draw_block_bytes: int = Field(
        default=0,
        ge=0,
        description=(
            "Raw block size for server-integrated draws; 0 defers to the "
            "server default (integration.block_bytes)"
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- Temperature Strategy (per-request overridable) ---

    temperature_strategy: str = Field(
        default="fixed",
        description="Temperature strategy: 'fixed' or 'edt'",
        json_schema_extra=_PER_REQUEST,
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
        json_schema_extra=_PER_REQUEST,
    )
    edt_base_temp: float = Field(
        default=0.8,
        description="Base coefficient for EDT",
        json_schema_extra=_PER_REQUEST,
    )
    edt_exponent: float = Field(
        default=0.5,
        description="Power-law exponent for EDT",
        json_schema_extra=_PER_REQUEST,
    )
    edt_min_temp: float = Field(
        default=0.1,
        description="EDT temperature floor",
        json_schema_extra=_PER_REQUEST,
    )
    edt_max_temp: float = Field(
        default=2.0,
        description="EDT temperature ceiling",
        json_schema_extra=_PER_REQUEST,
    )

    # --- HVH-Drift Temperature Strategy (per-request overridable) ---
    # Defaults pinned to V6_HVD_R01_01 winning configuration from
    # createmp-evalsuite (results/v6/round_final). Dormant unless
    # temperature_strategy = "hvh_drift" is explicitly set.

    hvh_t_base: float = Field(
        default=1.35,
        description="HVH-Drift base temperature (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_alpha_h: float = Field(
        default=0.3,
        description="HVH-Drift entropy coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_alpha_vh: float = Field(
        default=-0.2,
        description="HVH-Drift varentropy coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_gamma_dh: float = Field(
        default=1.0,
        description="HVH-Drift entropy-drift coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_delta_dvh: float = Field(
        default=0.5,
        description="HVH-Drift varentropy-drift coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_lambda_ema: float = Field(
        default=0.02,
        description="HVH-Drift EMA decay rate for H/VH state (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_min_p_base: float = Field(
        default=0.025,
        description="HVH-Drift min-p base term (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_kappa_h: float = Field(
        default=0.03,
        description="HVH-Drift min-p entropy coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )
    hvh_nu_dh: float = Field(
        default=0.02,
        description="HVH-Drift min-p entropy-drift coefficient (V6_HVD_R01_01 winner)",
        json_schema_extra=_PER_REQUEST,
    )

    # --- TT-Entropy-Exchange Temperature Strategy (per-request overridable) ---
    # V6 research spec §7.3 predicted defaults. Dormant unless
    # temperature_strategy = "tt_exchange" is explicitly set.

    tt_t_base: float = Field(
        default=1.0,
        description="TT-Exchange base temperature (V6 §7.3 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    tt_gamma: float = Field(
        default=0.6,
        description=(
            "TT-Exchange gain on the entropy removed by truncation "
            "(T = tt_t_base + tt_gamma * max(0, H - H_kept))"
        ),
        json_schema_extra=_PER_REQUEST,
    )
    tt_min_p_base: float = Field(
        default=0.005,
        description="TT-Exchange min-p base term (V6 §7.3 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    tt_min_p_scale: float = Field(
        default=0.025,
        description="TT-Exchange min-p entropy coefficient (V6 §7.3 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )

    # --- EVDT-TT Temperature Strategy (per-request overridable) ---
    # V6 research spec §7.1 predicted defaults. Dormant unless
    # temperature_strategy = "evdt_tt" is explicitly set. The family's
    # defining truncate-before-temperature order additionally requires
    # the per-request selector flag `qr_truncate_first: true`.

    evdt_t_base: float = Field(
        default=1.25,
        description="EVDT-TT base temperature (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    evdt_alpha: float = Field(
        default=0.35,
        description="EVDT-TT entropy coefficient (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    evdt_beta: float = Field(
        default=-0.10,
        description="EVDT-TT varentropy coefficient (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    evdt_min_p_base: float = Field(
        default=0.008,
        description="EVDT-TT min-p base term (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    evdt_min_p_scale: float = Field(
        default=0.015,
        description="EVDT-TT min-p entropy coefficient (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )
    evdt_min_p_vh: float = Field(
        default=0.005,
        description="EVDT-TT min-p varentropy coefficient (V6 §7.1 predicted default)",
        json_schema_extra=_PER_REQUEST,
    )

    # --- Coherence-Gate Temperature Strategy (per-request overridable) ---
    # Dormant unless temperature_strategy = "coherence_gate" is explicitly
    # set. The gate reads the coherence triple from the PREVIOUS
    # server-integrated draw's DrawMeta (one-draw lag, structural) and
    # boosts the inner strategy's base temperature.

    coherence_threshold: float = Field(
        default=3.5,
        description=(
            "Minimum coherence_z (Fisher-transformed cross-device block "
            "correlation) for the gate to open; below it the boost is 0.0"
        ),
        json_schema_extra=_PER_REQUEST,
    )
    coherence_t_boost_max: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Maximum temperature boost at r = 1; instantaneous boost is "
            "coherence_t_boost_max * max(0, coherence_r) when the gate is open"
        ),
        json_schema_extra=_PER_REQUEST,
    )
    coherence_ema_alpha: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "EMA smoothing for the gate boost: b_ema <- alpha*b + (1-alpha)*b_ema; "
            "1.0 means the instantaneous boost is applied unsmoothed"
        ),
        json_schema_extra=_PER_REQUEST,
    )
    coherence_inner_strategy: str = Field(
        default="fixed",
        description=(
            "Temperature strategy the coherence gate composes over; the boost "
            "shifts its base-temperature field on a per-token config copy"
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- Token Selection (per-request overridable) ---

    top_k: int = Field(
        default=0,
        description="Top-k filtering (<=0 disables)",
        json_schema_extra=_PER_REQUEST,
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling threshold (1.0 disables)",
        json_schema_extra=_PER_REQUEST,
    )
    min_p_base: float = Field(
        default=0.0,
        description=(
            "Default min-p truncation threshold used by the selector when the "
            "active temperature strategy does not emit a per-token min_p. "
            "Default 0.0 disables min-p (preserves prior behavior; NFR-7)."
        ),
        json_schema_extra=_PER_REQUEST,
    )
    truncate_first: bool = Field(
        default=False,
        description=(
            "Selector-order option for truncate-first-then-temperature "
            "families (EVDT-TT): apply the min-p mask on the RAW "
            "(temperature-free) distribution, then temperature on the kept "
            "support. Default False preserves the pinned selector order "
            "(top-k -> softmax -> min-p -> top-p -> CDF; AGENTS.md "
            "invariant 15) as a strict no-op."
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- Pipeline bypass (per-request overridable) ---

    bypass: bool = Field(
        default=False,
        description=(
            "Skip qr-sampler's sampling pipeline for this request entirely: "
            "its logits rows pass through the engine adapter untouched, so "
            "the engine's native sampler (temperature/top_p/top_k/seed from "
            "the standard request params) applies. Zero entropy is drawn and "
            "no sampling records or perf telemetry are produced. Default "
            "False — bare requests never bypass."
        ),
        json_schema_extra=_PER_REQUEST,
    )

    # --- Engine adapter (infrastructure; NOT per-request overridable) ---

    apply_parallel_rows: int = Field(
        default=0,
        ge=0,
        description=(
            "Worker-thread cap for the engine adapter's per-row sampling "
            "loop (vLLM apply()). 0 (default) resolves to the machine's CPU "
            "count; 1 restores the historical single-threaded loop; N > 1 "
            "caps the pool at N. Single-row batches never use the pool. Set "
            "to 1 if a third-party entropy source plugin is not safe for "
            "concurrent get_random_bytes() calls (all builtin sources are)."
        ),
    )

    # --- Logging (per-request overridable) ---

    log_level: str = Field(
        default="summary",
        description="Logging verbosity: 'none', 'summary', 'full'",
        json_schema_extra=_PER_REQUEST,
    )
    diagnostic_mode: bool = Field(
        default=False,
        description="Store all token records in memory for analysis",
        json_schema_extra=_PER_REQUEST,
    )

    # --- Preset (env-var ingestion only; NOT per-request overridable) ---
    # Resolved by qr_sampler.config.presets.expand_extra_args, not via the
    # normal PER_REQUEST_FIELDS merge path. Per-request callers use the
    # `qr_preset` key in extra_args directly.

    preset: str | None = Field(
        default=None,
        description=(
            "Optional preset name (e.g. 'creative_sampling', 'normal_t1') "
            "loaded from QR_PRESET. Expanded into field overrides at "
            "resolve_config() time; not a runtime sampling parameter."
        ),
    )

    # --- OpenEntropy (all infrastructure) ---
    # ``oe_conditioning`` is deliberately NOT per-request: it is read only
    # when the OpenEntropy source is constructed (see entropy/openentropy.py),
    # so a per-request override would be a silent no-op that falsifies the
    # ``config_hash`` provenance. Callers sending ``qr_oe_conditioning``
    # fail fast with ConfigValidationError instead.

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


#: All known config field names.
ALL_FIELDS: frozenset[str] = frozenset(QRSamplerConfig.model_fields.keys())

#: Fields overridable per-request via SamplingParams.extra_args — DERIVED
#: from the ``per_request`` field metadata, never hand-maintained.
PER_REQUEST_FIELDS: frozenset[str] = frozenset(
    name
    for name, info in QRSamplerConfig.model_fields.items()
    if isinstance(info.json_schema_extra, dict) and info.json_schema_extra.get("per_request")
)
