"""Data types for the diagnostic logging subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TokenSamplingRecord:
    """Immutable record of a single token sampling event.

    Captures all information about one token's sampling pipeline execution
    for diagnostic analysis and consciousness-influence research.

    Attributes:
        timestamp_ns: Wall-clock time of sampling (nanoseconds since epoch).
        entropy_fetch_ms: Time to fetch entropy (milliseconds).
        total_sampling_ms: Total time for the full sampling pipeline (ms).
        entropy_source_used: Name of the entropy source that provided bytes.
        entropy_is_fallback: True if a fallback source was used.
        sample_mean: Mean of raw entropy bytes (expected ~127.5 unbiased).
        z_score: Z-score from signal amplification.
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
