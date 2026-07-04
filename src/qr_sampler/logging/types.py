"""Data types for the diagnostic logging subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TokenSamplingRecord:
    """Immutable record of a single token sampling event.

    Captures all information about one token's sampling pipeline execution
    for diagnostic analysis and weak-signal integration research.

    Attributes:
        timestamp_ns: Wall-clock time of sampling (nanoseconds since epoch).
        entropy_fetch_ms: Time to fetch entropy (milliseconds).
        total_sampling_ms: Total time for the full sampling pipeline (ms).
        entropy_source_used: Name of the entropy source that provided bytes.
        entropy_is_fallback: True if a fallback source was used.
        sample_mean: Mean of raw entropy bytes (expected ~127.5 unbiased).
            ``math.nan`` on the server-draw path — no byte mean exists
            when the server integrates the block itself.
        z_score: Z-score from signal amplification (``draw_z`` on the
            server-draw path).
        u_value: Uniform value from amplification, in (0, 1).
        temperature_strategy: Name of the temperature strategy used.
        shannon_entropy: Shannon entropy of the logit distribution (nats).
        temperature_used: Final temperature applied.
        token_id: Vocabulary index of the selected token.
        token_rank: Rank of selected token (0 = most probable).
        token_prob: Probability of the selected token.
        num_candidates: Number of tokens surviving filtering.
        config_hash: 16-char SHA-256 prefix of the active config.
        varentropy: Varentropy ``VH`` of the logit distribution (HVH-Drift only).
        min_p_used: Effective per-token min-p threshold applied by the selector.
        preset_active: Name of the active preset (env-var derived), if any.
        h_ema: Smoothed entropy EMA after the current update (HVH-Drift only).
        vh_ema: Smoothed varentropy EMA after the current update (HVH-Drift only).
        entropy_prefetch_hit: ``True`` when this token's entropy arrived via
            a pipelined prefetch fired at the previous token's selection
            (round trip overlapped the forward pass); ``False`` when a
            prefetch was attempted but redeemed via the serial fallback;
            ``None`` when no prefetch was in play (serial mode / non-async
            source).
        entropy_nonce: Hex form of the 63-bit commitment nonce carried in
            the request's ``sequence_id`` field (pipelined path only).
            Derived from the previously selected token — see
            ``qr_sampler.core.pipeline.derive_commit_nonce``.
        entropy_echo_verified: ``True`` when the server echoed the nonce
            back, cryptographically binding this entropy to a request that
            could only exist after the previous token's selection.
        entropy_server_timestamp_ns: Server-reported physical generation
            timestamp (``generation_timestamp_ns``), when provided.
        draw_z: Server-integrated draw statistic z (server-draw mode only;
            equals ``z_score`` on the draw path). ``None`` elsewhere.
        draw_coherence_z: Fisher coherence statistic reported with the
            draw; meaningless unless ``draw_coherence_valid``.
        draw_coherence_valid: Whether the server's coherence monitor had a
            fresh value for this draw (``False``/``None`` => ignore the
            coherence numbers).
        draw_coherence_r: Peak lag-scanned Pearson r behind
            ``draw_coherence_z``.
        purity_label: Canonical purity label of the serving source.
        integrated_bytes: Raw bytes the server integrated into this draw.
        integrator: Server-side integrator registry name (e.g. ``bit_z``).
        draw_source_id: The SERVING source id echoed by the server.
        gate_open: Whether the coherence gate applied a positive
            temperature boost this token (coherence_gate strategy only).
        gate_boost: The EMA-smoothed temperature boost applied pre-inner
            strategy (coherence_gate strategy only).
    """

    # Timing
    timestamp_ns: int
    entropy_fetch_ms: float
    total_sampling_ms: float

    # Entropy source
    entropy_source_used: str
    entropy_is_fallback: bool

    # Signal amplification
    sample_mean: float
    z_score: float
    u_value: float

    # Temperature
    temperature_strategy: str
    shannon_entropy: float
    temperature_used: float

    # Selection
    token_id: int
    token_rank: int
    token_prob: float
    num_candidates: int

    # Config snapshot
    config_hash: str

    # Optional HVH-Drift / preset diagnostics. ``None`` for non-HVH strategies
    # so existing call sites and downstream consumers stay backward-compatible.
    varentropy: float | None = field(default=None)
    min_p_used: float | None = field(default=None)
    preset_active: str | None = field(default=None)
    h_ema: float | None = field(default=None)
    vh_ema: float | None = field(default=None)

    # Pipelined-entropy verification diagnostics. ``None`` on the serial
    # path so existing consumers stay backward-compatible.
    entropy_prefetch_hit: bool | None = field(default=None)
    entropy_nonce: str | None = field(default=None)
    entropy_echo_verified: bool | None = field(default=None)
    entropy_server_timestamp_ns: int | None = field(default=None)

    # iter-55: per-stage timing breakdown of the sampling pipeline, used
    # by the adapter's rolling perf aggregate (/health/entropy "perf"
    # block) to attribute per-token cost. ``entropy_fetch_ms`` above
    # remains the entropy stage's share.
    temperature_ms: float | None = field(default=None)
    amplify_ms: float | None = field(default=None)
    select_ms: float | None = field(default=None)

    # Server-integrated draw diagnostics (qr_purity.PurityService). ``None``
    # on the local byte-amplification path (and on the degraded-draw path,
    # where no DrawMeta exists) so existing consumers stay
    # backward-compatible — same precedent as the prefetch block above.
    draw_z: float | None = field(default=None)
    draw_coherence_z: float | None = field(default=None)
    draw_coherence_valid: bool | None = field(default=None)
    draw_coherence_r: float | None = field(default=None)
    purity_label: str | None = field(default=None)
    integrated_bytes: int | None = field(default=None)
    integrator: str | None = field(default=None)
    draw_source_id: str | None = field(default=None)

    # Coherence-gate temperature diagnostics, copied from the strategy's
    # TemperatureResult diagnostics when present (coherence_gate only).
    gate_open: bool | None = field(default=None)
    gate_boost: float | None = field(default=None)
