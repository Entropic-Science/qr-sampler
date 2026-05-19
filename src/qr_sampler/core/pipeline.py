"""Engine-agnostic sampling pipeline.

Orchestrates the full per-token sampling flow:
    logits (numpy) -> temperature -> entropy fetch -> amplification
    -> CDF selection -> one-hot numpy -> diagnostic record.

This module has **zero** imports from ``torch``, ``vllm``, or any inference
engine package. It operates exclusively on 1-D numpy arrays.

Factory functions (``build_pipeline``, ``build_entropy_source``,
``config_hash``, ``accepts_config``) provide construction helpers
shared by all engine adapters.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.types import SamplingResult
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.registry import EntropySourceRegistry
from qr_sampler.logging.logger import SamplingLogger
from qr_sampler.logging.types import TokenSamplingRecord
from qr_sampler.selection.selector import TokenSelector
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.entropy.base import EntropySource
    from qr_sampler.temperature.base import TemperatureStrategy

logger = logging.getLogger("qr_sampler")


def config_hash(config: QRSamplerConfig) -> str:
    """Compute a short hash of the config for logging.

    Args:
        config: The sampler configuration to hash.

    Returns:
        First 16 hex characters of the SHA-256 digest of the config dump.
    """
    raw = config.model_dump_json().encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def accepts_config(cls: type) -> bool:
    """Check if a class constructor accepts a QRSamplerConfig as first arg.

    Inspects the ``__init__`` signature for a parameter annotated as
    ``QRSamplerConfig`` (or whose name is ``config``).

    Args:
        cls: The class to inspect.

    Returns:
        True if the constructor expects a config argument.
    """
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return False

    params = list(sig.parameters.values())
    # inspect.signature(cls) already strips 'self' for classes.
    for param in params:
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            if param.name == "config":
                return True
        elif annotation is QRSamplerConfig or (
            isinstance(annotation, str) and "QRSamplerConfig" in annotation
        ):
            return True
        # Only check the first non-self parameter.
        break
    return False


def build_entropy_source(config: QRSamplerConfig) -> EntropySource:
    """Build the entropy source from config, wrapping with fallback if needed.

    Args:
        config: Sampler configuration specifying source type and fallback mode.

    Returns:
        An EntropySource, potentially wrapped in FallbackEntropySource.
    """
    source_cls = EntropySourceRegistry.get(config.entropy_source_type)

    # Only pass config if the constructor expects it.
    if accepts_config(source_cls):
        primary: EntropySource = source_cls(config)  # type: ignore[call-arg]
    else:
        primary = source_cls()

    if config.fallback_mode == "error":
        return primary

    # Build fallback source.
    if config.fallback_mode == "system":
        from qr_sampler.entropy.system import SystemEntropySource

        fallback: EntropySource = SystemEntropySource()
    elif config.fallback_mode == "mock_uniform":
        from qr_sampler.entropy.mock import MockUniformSource

        fallback = MockUniformSource()
    else:
        logger.warning(
            "Unknown fallback_mode %r, using system fallback",
            config.fallback_mode,
        )
        from qr_sampler.entropy.system import SystemEntropySource

        fallback = SystemEntropySource()

    return FallbackEntropySource(primary, fallback)


def build_pipeline(config: QRSamplerConfig, vocab_size: int) -> SamplingPipeline:
    """Construct a fully-initialized SamplingPipeline from config.

    This is the primary factory function. Engine adapters call this
    to get a ready-to-use pipeline without knowing construction details.

    Construction sequence:
        1. ``build_entropy_source(config)`` — with fallback wrapping
        2. ``AmplifierRegistry.build(config)`` — from registry
        3. Calibrate amplifier if it supports calibration
        4. ``TemperatureStrategyRegistry.build(config, vocab_size)`` — from registry
        5. ``TokenSelector()``
        6. ``SamplingLogger(config)``
        7. Return ``SamplingPipeline(...)``

    Args:
        config: Sampler configuration.
        vocab_size: Vocabulary size of the model.

    Returns:
        A fully constructed and ready-to-use SamplingPipeline.
    """
    entropy_source = build_entropy_source(config)

    amplifier = AmplifierRegistry.build(config)
    # Calibrate amplifier if it supports calibration (e.g., ECDF).
    if hasattr(amplifier, "calibrate"):
        amplifier.calibrate(entropy_source, config)

    strategy = TemperatureStrategyRegistry.build(config, vocab_size)
    selector = TokenSelector()
    sampling_logger = SamplingLogger(config)

    return SamplingPipeline(
        entropy_source=entropy_source,
        amplifier=amplifier,
        strategy=strategy,
        selector=selector,
        sampling_logger=sampling_logger,
        config=config,
    )


class SamplingPipeline:
    """Engine-agnostic sampling pipeline.

    Orchestrates: temperature -> entropy fetch -> amplification -> CDF selection.
    Operates on 1-D numpy arrays. Has no dependency on any inference engine.

    All components are injected via the constructor. Use ``build_pipeline()``
    for the standard construction path.
    """

    def __init__(
        self,
        entropy_source: EntropySource,
        amplifier: SignalAmplifier,
        strategy: TemperatureStrategy,
        selector: TokenSelector,
        sampling_logger: SamplingLogger,
        config: QRSamplerConfig,
    ) -> None:
        """Initialize the pipeline with all required components.

        Args:
            entropy_source: Source of random bytes (may be a FallbackEntropySource).
            amplifier: Signal amplification algorithm.
            strategy: Temperature computation strategy.
            selector: CDF-based token selector.
            sampling_logger: Diagnostic logger.
            config: Default configuration for this pipeline.
        """
        self._entropy_source = entropy_source
        self._amplifier = amplifier
        self._strategy = strategy
        self._selector = selector
        self._sampling_logger = sampling_logger
        self._config = config
        self._default_config_hash = config_hash(config)

    def sample_token(
        self,
        logits: np.ndarray,
        config: QRSamplerConfig | None = None,
        amplifier: SignalAmplifier | None = None,
        strategy: TemperatureStrategy | None = None,
        config_hash_str: str | None = None,
    ) -> SamplingResult:
        """Sample a single token from a 1-D logit array.

        Runs the full pipeline: temperature -> entropy -> amplify -> select
        -> one-hot numpy -> diagnostic record -> log.

        Args:
            logits: 1-D numpy array of shape ``(vocab_size,)``.
            config: Per-request config override (``None`` = use default).
            amplifier: Per-request amplifier override (``None`` = use default).
            strategy: Per-request strategy override (``None`` = use default).
            config_hash_str: Pre-computed hash (``None`` = compute from config).

        Returns:
            SamplingResult with ``token_id``, ``one_hot`` numpy array,
            and ``record`` for diagnostics.
        """
        t_start_ns = time.perf_counter_ns()

        # Resolve per-request overrides.
        active_config = config if config is not None else self._config
        active_amplifier = amplifier if amplifier is not None else self._amplifier
        active_strategy = strategy if strategy is not None else self._strategy
        hash_str = config_hash_str if config_hash_str is not None else self._default_config_hash

        # --- 1. Compute temperature ---
        temp_result = active_strategy.compute_temperature(logits, active_config)

        # Per-token min-p: HVH-Drift publishes a value via diagnostics; other
        # strategies omit the key, so fall back to the config-level default.
        min_p = float(temp_result.diagnostics.get("min_p", active_config.min_p_base))

        # --- 2. Fetch entropy just-in-time ---
        t_fetch_start = time.perf_counter_ns()
        entropy_is_fallback = False
        entropy_source_name = self._entropy_source.name

        raw_bytes = self._entropy_source.get_random_bytes(active_config.sample_count)

        # Detect if fallback was used.
        if isinstance(self._entropy_source, FallbackEntropySource):
            entropy_source_name = self._entropy_source.last_source_used
            entropy_is_fallback = (
                self._entropy_source.last_source_used != self._entropy_source.primary_name
            )

        t_fetch_end = time.perf_counter_ns()
        entropy_fetch_ms = (t_fetch_end - t_fetch_start) / 1_000_000.0

        # --- 3. Amplify to uniform float ---
        amp_result = active_amplifier.amplify(raw_bytes)

        # --- 4. Select token via CDF ---
        selection = self._selector.select(
            logits,
            temp_result.temperature,
            active_config.top_k,
            active_config.top_p,
            amp_result.u,
            min_p=min_p,
        )

        # --- 5. Build one-hot numpy array ---
        vocab_size = len(logits)
        one_hot = np.full(vocab_size, float("-inf"), dtype=np.float32)
        one_hot[selection.token_id] = 0.0

        # --- 6. Build diagnostic record ---
        t_end_ns = time.perf_counter_ns()
        total_sampling_ms = (t_end_ns - t_start_ns) / 1_000_000.0

        # Optional HVH-Drift / preset diagnostics. ``.get`` returns ``None``
        # for non-HVH strategies and pre-Step-2 selectors that omit min_p_used.
        temp_diag = temp_result.diagnostics
        record = TokenSamplingRecord(
            timestamp_ns=t_start_ns,
            entropy_fetch_ms=entropy_fetch_ms,
            total_sampling_ms=total_sampling_ms,
            entropy_source_used=entropy_source_name,
            entropy_is_fallback=entropy_is_fallback,
            sample_mean=amp_result.diagnostics.get("sample_mean", 0.0),
            z_score=amp_result.diagnostics.get("z_score", 0.0),
            u_value=amp_result.u,
            temperature_strategy=active_config.temperature_strategy,
            shannon_entropy=temp_result.shannon_entropy,
            temperature_used=temp_result.temperature,
            token_id=selection.token_id,
            token_rank=selection.token_rank,
            token_prob=selection.token_prob,
            num_candidates=selection.num_candidates,
            config_hash=hash_str,
            varentropy=temp_diag.get("varentropy"),
            min_p_used=selection.diagnostics.get("min_p_used"),
            preset_active=active_config.preset,
            h_ema=temp_diag.get("h_ema"),
            vh_ema=temp_diag.get("vh_ema"),
        )

        # --- 7. Log ---
        self._sampling_logger.log_token(record)

        return SamplingResult(
            token_id=selection.token_id,
            one_hot=one_hot,
            record=record,
        )

    @property
    def entropy_source(self) -> EntropySource:
        """The active entropy source (may be a FallbackEntropySource wrapper)."""
        return self._entropy_source

    @property
    def amplifier(self) -> SignalAmplifier:
        """The default signal amplifier for this pipeline."""
        return self._amplifier

    @property
    def strategy(self) -> TemperatureStrategy:
        """The default temperature strategy for this pipeline."""
        return self._strategy

    @property
    def default_config(self) -> QRSamplerConfig:
        """The default configuration for this pipeline."""
        return self._config

    @property
    def sampling_logger(self) -> SamplingLogger:
        """The diagnostic logger for this pipeline."""
        return self._sampling_logger

    def close(self) -> None:
        """Release entropy source resources."""
        self._entropy_source.close()
