"""vLLM V1 LogitsProcessor adapter.

Implements the vLLM V1 LogitsProcessor contract:
    - ``__init__(vllm_config, device, is_pin_memory)``
    - ``apply(logits) -> logits``
    - ``update_state(batch_update) -> None``
    - ``validate_params(params) -> None``
    - ``is_argmax_invariant() -> bool``

Internally delegates all sampling to ``SamplingPipeline``.

Registered via entry point::

    [project.entry-points."vllm.logits_processors"]
    qr_sampler = "qr_sampler.engines.vllm:VLLMAdapter"

The processor applies globally to all requests in a vLLM instance. Deploy
separate instances for different sampling strategies.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.core.pipeline import (
    SamplingPipeline,
    build_pipeline,
    config_hash,
    derive_commit_nonce,
)
from qr_sampler.core.types import PrefetchContext
from qr_sampler.engines.base import EngineAdapter
from qr_sampler.engines.vllm.telemetry import _PerfAggregator
from qr_sampler.exceptions import ConfigValidationError
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

# Formal V1 LogitsProcessor base.
# vLLM 0.17.0 entry-point discovery validates plugins via issubclass against
# vllm.v1.sample.logits_processor.LogitsProcessor. Duck-typing alone passes
# attribute checks but fails the isinstance/issubclass gate. In dev / test
# environments vLLM is not installed, so we fall back to ``object`` — the
# module must still import cleanly there because EngineAdapterRegistry's
# builtin table resolves this module on demand.
try:
    from vllm.v1.sample.logits_processor import (
        LogitsProcessor as _VLLMLogitsProcessorBase,
    )
except ImportError:  # pragma: no cover - exercised where vLLM is not installed
    # The ignore is only "used" when mypy runs in an env where vLLM IS
    # installed (LogitsProcessor then resolves to a real class instead of
    # Any); warn_unused_ignores is disabled for this module in pyproject.
    _VLLMLogitsProcessorBase = object  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from qr_sampler.amplification.base import SignalAmplifier
    from qr_sampler.logging.logger import SamplingLogger
    from qr_sampler.temperature.base import TemperatureStrategy

logger = logging.getLogger("qr_sampler")

# Default vocabulary size when vllm_config does not provide one (testing).
_DEFAULT_VOCAB_SIZE = 32000

# Env var listing the entropy sources to pre-initialise at adapter startup.
# Comma-separated; whitespace tolerated. A per-request ``qr_entropy_source_type``
# override must name one of these — anything else is rejected when the request
# is added to the batch.
_PREINIT_ENV_VAR = "QR_PREINIT_ENTROPY_SOURCES"
_DEFAULT_PREINIT = "quantum_grpc,system"


class _RequestState:
    """Per-request state tracked across engine steps.

    Attributes:
        pipeline: The pre-initialised pipeline whose entropy source matches
            this request's ``entropy_source_type``. All sampling for this
            request flows through this pipeline.
        config: Resolved per-request configuration.
        amplifier: Signal amplifier for this request.
        strategy: Temperature strategy for this request.
        config_hash_str: Short hash for logging.
        source: The entropy source this request resolved to. Exposed for
            test introspection; production code should treat the pipeline
            as opaque.
        tokens_generated: Counter incremented once per ``apply()`` call
            that processed this request. Used to populate the
            ``entropy.request.completed`` event so an operator can spot
            "Modal Succeeded but 0 tokens" failure modes (K-5) from the
            log stream alone.
        dominant_source_name: Snapshot of ``pipeline.entropy_source.name``
            at routing time. For the always-primary case this is the
            actual source used; if a future change switches sources
            mid-request, this becomes the routing-time hint and the
            completion event should record the observed dominant.
        prefetch_salt: Per-request random salt for commitment-nonce
            derivation (commit-then-fetch entropy pipelining).
        entropy_ticket: In-flight prefetch ticket for this request's next
            token, or ``None``. Fired at the previous token's selection
            (or at request-add time for the first token, which overlaps
            the entire prefill) and redeemed on the next ``apply()``.
    """

    __slots__ = (
        "amplifier",
        "config",
        "config_hash_str",
        "dominant_source_name",
        "entropy_ticket",
        "pipeline",
        "prefetch_salt",
        "source",
        "strategy",
        "tokens_generated",
    )

    def __init__(
        self,
        pipeline: SamplingPipeline,
        config: QRSamplerConfig,
        amplifier: SignalAmplifier,
        strategy: TemperatureStrategy,
        config_hash_str: str,
    ) -> None:
        self.pipeline = pipeline
        self.config = config
        self.amplifier = amplifier
        self.strategy = strategy
        self.config_hash_str = config_hash_str
        self.source = pipeline.entropy_source
        self.tokens_generated = 0
        self.dominant_source_name = pipeline.entropy_source.name
        self.prefetch_salt = os.urandom(16)
        self.entropy_ticket: Any | None = None


class VLLMAdapter(EngineAdapter, _VLLMLogitsProcessorBase):
    """vLLM V1 LogitsProcessor that replaces token sampling with
    external-entropy-driven selection.

    Formally subclasses ``vllm.v1.sample.logits_processor.LogitsProcessor``
    when vLLM is importable (the Modal runtime path); falls back to a
    plain object base in dev/test environments where vLLM is unavailable.
    The dual base list keeps ``EngineAdapter``'s ABC contract first so
    its abstractmethod checks still apply.

    The adapter manages vLLM-specific concerns (batch state, tensor
    conversion, one-hot forcing) and delegates all sampling logic to
    the engine-agnostic ``SamplingPipeline``.

    Constructor signature matches vLLM V1's ``LogitsProcessor`` ABC::

        __init__(self, vllm_config, device, is_pin_memory)
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ) -> None:
        """Initialize the adapter and all subsystems.

        Args:
            vllm_config: vLLM's ``VllmConfig`` object (provides vocab_size).
                ``None`` in test environments -- uses ``_DEFAULT_VOCAB_SIZE``.
            device: ``torch.device`` for tensor operations. ``None`` in tests.
            is_pin_memory: Whether to use pinned CPU memory for transfers.
        """
        # --- Extract vocab_size ---
        self._vocab_size = self._extract_vocab_size(vllm_config)
        self._device = device
        self._is_pin_memory = is_pin_memory

        # --- Load default configuration ---
        self._default_config = QRSamplerConfig()

        # --- Pre-initialise one pipeline per allowed entropy source ---
        # The default source from QR_ENTROPY_SOURCE_TYPE is always included
        # so a request with no per-request override still resolves cleanly.
        self._preinit_sources = self._resolve_preinit_sources(
            self._default_config.entropy_source_type
        )
        self._pipelines: dict[str, SamplingPipeline] = {}
        for source_type in self._preinit_sources:
            self._pipelines[source_type] = self._build_pipeline_for_source(source_type)

        # --- Default pipeline (used when no per-request state exists) ---
        self._pipeline = self._pipelines[self._default_config.entropy_source_type]

        # iter-52d (2026-05-25): eagerly establish each entropy source's
        # connection NOW, so per-token ``get_random_bytes()`` calls land
        # on already-open, already-verified channels. The default
        # ``EntropySource.warmup()`` is a no-op; ``QuantumGrpcSource``
        # overrides it to open the gRPC channel + do a single tiny
        # verification fetch. Soft-fail by design: if QRNG is
        # unreachable at startup, ``FallbackEntropySource`` engages
        # transparently for subsequent fetches.
        for source_type, pipeline in self._pipelines.items():
            try:
                pipeline.entropy_source.warmup()
            except Exception as exc:
                logger.warning(
                    "Entropy-source warmup for %s raised (%s); fallback will engage",
                    source_type,
                    exc,
                    extra={
                        "event": "entropy.warmup.failed",
                        "source_type": source_type,
                    },
                )

        # Designate the DEFAULT pipeline's source as the owner of the
        # cross-process entropy-status file (telemetry IPC for
        # out-of-process health readers). Only the default pipeline
        # publishes: the pre-init loop also builds a system-primary
        # wrapper whose always-healthy state must not clobber the
        # quantum lane's file.
        try:
            enable = getattr(self._pipeline.entropy_source, "enable_status_publishing", None)
            if callable(enable):
                enable()
        except Exception:
            # Telemetry plumbing must never break the LogitsProcessor.
            pass

        # --- Pre-compute default state ---
        self._default_config_hash = config_hash(self._default_config)

        # --- Pre-allocate tensors ---
        self._onehot_template = self._create_onehot_template()
        self._cpu_buffer = self._create_cpu_buffer()

        # --- Per-request state ---
        # Maps request index (batch position) to its state.
        self._request_states: dict[int, _RequestState] = {}

        # --- iter-55: rolling per-stage perf telemetry ---
        self._perf = _PerfAggregator()

        logger.info(
            "VLLMAdapter initialized: vocab_size=%d, "
            "default_entropy_source=%s, preinit_sources=%s, "
            "amplifier=%s, temperature=%s",
            self._vocab_size,
            self._pipeline.entropy_source.name,
            sorted(self._pipelines.keys()),
            self._default_config.signal_amplifier_type,
            self._default_config.temperature_strategy,
        )

    @staticmethod
    def _resolve_preinit_sources(default_source_type: str) -> list[str]:
        """Parse the pre-init source list from env, deduped and ordered.

        The default source is always included so the no-override path is
        always serviceable. Order is preserved (first occurrence wins) so
        operators can document a canonical order in their env config.
        """
        raw = os.environ.get(_PREINIT_ENV_VAR, _DEFAULT_PREINIT)
        parsed = [part.strip() for part in raw.split(",") if part.strip()]
        if default_source_type not in parsed:
            parsed.append(default_source_type)
        seen: set[str] = set()
        ordered: list[str] = []
        for name in parsed:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _build_pipeline_for_source(self, source_type: str) -> SamplingPipeline:
        """Build a SamplingPipeline whose entropy source is ``source_type``.

        Mirrors ``self._default_config`` for every other field. Uses
        ``model_validate`` so type coercion runs without re-reading env.
        """
        cfg_dump = self._default_config.model_dump()
        cfg_dump["entropy_source_type"] = source_type
        cfg = QRSamplerConfig.model_validate(cfg_dump)
        return build_pipeline(cfg, self._vocab_size)

    def get_pipeline(self) -> SamplingPipeline:
        """Return the underlying SamplingPipeline.

        Returns:
            The engine-agnostic sampling pipeline used by this adapter.
        """
        return self._pipeline

    @staticmethod
    def _extract_vocab_size(vllm_config: Any) -> int:
        """Extract vocabulary size from vLLM config, with fallback.

        Args:
            vllm_config: vLLM config object, or ``None`` for tests.

        Returns:
            Vocabulary size as integer.
        """
        if vllm_config is None:
            return _DEFAULT_VOCAB_SIZE

        # vLLM V1: vllm_config.model_config.hf_text_config.vocab_size
        try:
            return int(vllm_config.model_config.hf_text_config.vocab_size)
        except AttributeError:
            pass

        # Try direct vocab_size attribute.
        try:
            return int(vllm_config.vocab_size)
        except AttributeError:
            pass

        logger.warning(
            "Could not extract vocab_size from vllm_config, using default %d",
            _DEFAULT_VOCAB_SIZE,
        )
        return _DEFAULT_VOCAB_SIZE

    def _create_onehot_template(self) -> Any:
        """Create the one-hot template tensor filled with -inf.

        Returns:
            A tensor of shape ``(vocab_size,)`` filled with ``-inf``,
            or a numpy array if torch is unavailable.
        """
        try:
            import torch

            return torch.full(
                (self._vocab_size,),
                float("-inf"),
                device=self._device,
                dtype=torch.float32,
            )
        except ImportError:
            return np.full(self._vocab_size, float("-inf"), dtype=np.float32)

    def _create_cpu_buffer(self) -> Any:
        """Create a pinned-memory CPU buffer for transfers.

        Returns:
            A pinned tensor if ``is_pin_memory`` is True and torch is available,
            otherwise ``None``.
        """
        if not self._is_pin_memory:
            return None
        try:
            import torch

            return torch.empty(self._vocab_size, dtype=torch.float32, pin_memory=True)
        except ImportError:
            return None

    def is_argmax_invariant(self) -> bool:
        """Return ``False`` -- this processor fundamentally changes token selection.

        This ensures the processor runs before penalties and temperature scaling
        in the vLLM pipeline, operating on raw logits.
        """
        return False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate ``qr_*`` keys in ``params.extra_args``."""
        extra_args = getattr(params, "extra_args", None) or {}
        if extra_args:
            validate_extra_args(extra_args)

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch composition changes.

        Must be called every engine step before ``apply()``. Processes
        changes in the required order: removed -> moved -> added.

        Args:
            batch_update: A ``BatchUpdate`` with ``removed``, ``moved``,
                and ``added`` sequences, or ``None`` if no changes.
        """
        self._update_state_impl(batch_update)

    def _update_state_impl(self, batch_update: Any | None) -> None:
        if batch_update is None:
            return

        # 1. Process removals.
        for removed in getattr(batch_update, "removed", []):
            req_idx = removed if isinstance(removed, int) else getattr(removed, "req_index", None)
            if req_idx is not None:
                state = self._request_states.pop(req_idx, None)
                if state is not None:
                    # The speculative prefetch for the never-sampled next
                    # token is abandoned — cancel best-effort so the
                    # background loop drops it on arrival.
                    if state.entropy_ticket is not None:
                        with contextlib.suppress(Exception):
                            state.entropy_ticket.cancel()
                        state.entropy_ticket = None
                    # Emit a request-boundary event so an operator can spot
                    # the K-5 failure mode (Modal-reported success with zero
                    # output) directly from the log stream.
                    logger.info(
                        "request %d completed: tokens=%d source=%s",
                        req_idx,
                        state.tokens_generated,
                        state.dominant_source_name,
                        extra={
                            "event": "entropy.request.completed",
                            "req_idx": req_idx,
                            "tokens_generated": state.tokens_generated,
                            "dominant_source": state.dominant_source_name,
                        },
                    )

        # 2. Process moves (index reassignments).
        for moved in getattr(batch_update, "moved", []):
            if hasattr(moved, "src_index") and hasattr(moved, "dst_index"):
                state = self._request_states.pop(moved.src_index, None)
                if state is not None:
                    self._request_states[moved.dst_index] = state

        # 3. Process additions.
        for added in getattr(batch_update, "added", []):
            req_idx = getattr(added, "req_index", None)
            if req_idx is None:
                continue

            extra_args = (
                getattr(
                    getattr(added, "sampling_params", None),
                    "extra_args",
                    None,
                )
                or {}
            )

            # Resolve per-request config.
            req_config = resolve_config(self._default_config, extra_args)

            # Route to the pipeline matching the (possibly-overridden) source.
            target_source_type = req_config.entropy_source_type
            target_pipeline = self._pipelines.get(target_source_type)
            if target_pipeline is None:
                raise ConfigValidationError(
                    f"Entropy source {target_source_type!r} is not pre-initialised "
                    f"for this adapter. Pre-initialised sources: "
                    f"{sorted(self._pipelines.keys())}. "
                    f"Set {_PREINIT_ENV_VAR!s} at process startup to include it."
                )

            # Always build a fresh strategy. Stateful strategies (e.g.
            # hvh_drift) carry per-request EMA state on the instance, so
            # sharing the target pipeline's default would leak distributional
            # drift between concurrent requests (CLAUDE.md invariant 17).
            # The cost is one constructor call per addition.
            strategy = TemperatureStrategyRegistry.build(req_config, self._vocab_size)

            # Amplifier is safe to share when the resolved config matches
            # the target pipeline's defaults (it carries only calibration
            # state, which depends on config). Rebuild only when the config
            # differs. We compare model_dumps so a request that only overrode
            # the entropy source type still hits the fast path.
            if req_config is self._default_config or (
                req_config.model_dump() == target_pipeline.default_config.model_dump()
            ):
                amplifier = target_pipeline.amplifier
                hash_str = config_hash(target_pipeline.default_config)
            else:
                amplifier = AmplifierRegistry.build(req_config)
                # Calibrate per-request amplifier if it supports calibration.
                if hasattr(amplifier, "calibrate"):
                    amplifier.calibrate(target_pipeline.entropy_source, req_config)
                hash_str = config_hash(req_config)

            state = _RequestState(
                pipeline=target_pipeline,
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
            )
            self._request_states[req_idx] = state

            # Commit-then-fetch, step 0: fire the FIRST token's entropy
            # request now, at request-add time. There is no previous token
            # to wait for (the -1 sentinel commits to that fact), and the
            # round trip overlaps the entire prefill instead of stalling
            # the first sampling step. ``prefetch()`` never raises and
            # returns None for non-async sources (e.g. the system/PRNG
            # comparison lane), which keeps that path untouched.
            if req_config.entropy_prefetch:
                first_nonce = derive_commit_nonce(state.prefetch_salt, 0, -1)
                state.entropy_ticket = target_pipeline.entropy_source.prefetch(
                    req_config.sample_count, first_nonce
                )

            # Emit a single per-request routing event so an operator can
            # confirm the comparison pipe's per-column extra_args
            # ("qr_entropy_source_type": "quantum_grpc" vs "system")
            # actually landed on the matching pipeline. Without this,
            # K-2-style misconfigurations (Modal Secret silently
            # overriding the default to "system") silently route every
            # request to the wrong source.
            logger.info(
                "request %d routed: requested=%s resolved=%s preset=%s",
                req_idx,
                extra_args.get("qr_entropy_source_type"),
                target_pipeline.entropy_source.name,
                extra_args.get("qr_preset"),
                extra={
                    "event": "entropy.request.routed",
                    "req_idx": req_idx,
                    "requested_source_type": extra_args.get("qr_entropy_source_type"),
                    "resolved_pipeline_source": target_pipeline.entropy_source.name,
                    "extra_args_keys": sorted(extra_args.keys()),
                    "qr_preset": extra_args.get("qr_preset"),
                },
            )

    def apply(self, logits: Any) -> Any:
        """Run the full sampling pipeline on each row of the logit tensor.

        For each request in the batch:
            1. Convert logit row to numpy
            2. Delegate to ``pipeline.sample_token()``
            3. Write one-hot result back to engine tensor

        Args:
            logits: 2-D tensor of shape ``(num_requests, vocab_size)``.
                May be a ``torch.Tensor`` or a ``numpy.ndarray``.

        Returns:
            The modified logits tensor (in-place).
        """
        return self._apply_impl(logits)

    def _apply_impl(self, logits: Any) -> Any:
        # Determine batch size.
        if hasattr(logits, "shape"):
            num_requests = logits.shape[0] if len(logits.shape) > 1 else 1
        else:
            return logits

        if num_requests == 0:
            return logits

        is_numpy = isinstance(logits, np.ndarray)
        is_1d = len(logits.shape) == 1

        for i in range(num_requests):
            # Get per-request state or fall back to defaults.
            state = self._request_states.get(i)
            prefetch_ctx: PrefetchContext | None = None
            if state is not None:
                pipeline = state.pipeline
                req_config: QRSamplerConfig | None = state.config
                amplifier: SignalAmplifier | None = state.amplifier
                strategy: TemperatureStrategy | None = state.strategy
                hash_str: str | None = state.config_hash_str
                # Step index BEFORE increment: 0-based token index, the
                # same convention derive_commit_nonce documents.
                prefetch_ctx = PrefetchContext(
                    salt=state.prefetch_salt,
                    step=state.tokens_generated,
                    ticket=state.entropy_ticket,
                )
                state.entropy_ticket = None  # consumed below, one way or another
                state.tokens_generated += 1
            else:
                pipeline = self._pipeline
                req_config = None
                amplifier = None
                strategy = None
                hash_str = None

            # --- Extract row as numpy ---
            t_stage = time.perf_counter_ns()
            if is_1d:
                row = logits if is_numpy else self._to_numpy(logits)
            else:
                row = logits[i] if is_numpy else self._to_numpy(logits[i])
            to_numpy_ms = (time.perf_counter_ns() - t_stage) / 1_000_000.0

            # --- Delegate to pipeline (routed by entropy source) ---
            # build_onehot=False: the one-hot is forced directly on the
            # engine tensor below, so the pipeline's vocab-size numpy
            # allocation + fill per token would be pure dead weight.
            result = pipeline.sample_token(
                row,
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
                prefetch_ctx=prefetch_ctx,
                build_onehot=False,
            )

            # Store the in-flight ticket for this request's NEXT token
            # (fired inside sample_token immediately after selection).
            if state is not None:
                state.entropy_ticket = result.next_ticket

            # --- Force one-hot logits using engine tensor ---
            t_stage = time.perf_counter_ns()
            if is_1d:
                self._force_onehot(logits, result.token_id, is_numpy)
            else:
                self._force_onehot_row(logits, i, result.token_id, is_numpy)
            onehot_ms = (time.perf_counter_ns() - t_stage) / 1_000_000.0

            self._perf.note(result.record, to_numpy_ms, onehot_ms)

        return logits

    @staticmethod
    def _to_numpy(tensor: Any) -> np.ndarray:
        """Convert a tensor to a numpy array with zero-copy where possible.

        Args:
            tensor: A torch.Tensor or numpy array.

        Returns:
            Numpy array view (if CPU tensor) or copy.
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        # torch.Tensor -- use .numpy() for zero-copy on CPU.
        try:
            if not tensor.is_cpu:
                result: np.ndarray = tensor.detach().cpu().numpy()
            else:
                result = tensor.detach().numpy()
            return result
        except AttributeError:
            return np.asarray(tensor)

    def _force_onehot(self, logits: Any, token_id: int, is_numpy: bool) -> None:
        """Force 1-D logits to one-hot: all -inf except token_id = 0.0.

        Args:
            logits: 1-D logit array or tensor.
            token_id: The selected token index.
            is_numpy: Whether logits is a numpy array.
        """
        if is_numpy:
            logits[:] = float("-inf")
            logits[token_id] = 0.0
        else:
            logits.copy_(self._onehot_template, non_blocking=True)
            logits[token_id] = 0.0

    def _force_onehot_row(
        self,
        logits: Any,
        row_idx: int,
        token_id: int,
        is_numpy: bool,
    ) -> None:
        """Force a batch row to one-hot: all -inf except token_id = 0.0.

        Args:
            logits: 2-D logit array or tensor.
            row_idx: Batch row index.
            token_id: The selected token index.
            is_numpy: Whether logits is a numpy array.
        """
        if is_numpy:
            logits[row_idx, :] = float("-inf")
            logits[row_idx, token_id] = 0.0
        else:
            logits[row_idx].copy_(self._onehot_template, non_blocking=True)
            logits[row_idx, token_id] = 0.0

    @property
    def entropy_source(self) -> Any:
        """The active entropy source (may be a FallbackEntropySource wrapper)."""
        return self._pipeline.entropy_source

    @property
    def default_config(self) -> QRSamplerConfig:
        """The default configuration loaded from environment."""
        return self._default_config

    @property
    def sampling_logger(self) -> SamplingLogger:
        """The diagnostic logger for this processor."""
        return self._pipeline.sampling_logger

    def close(self) -> None:
        """Release all resources held by the adapter.

        Closes every pre-initialised pipeline. Safe to call multiple times.
        """
        seen: set[int] = set()
        for pipeline in self._pipelines.values():
            if id(pipeline) in seen:
                continue
            seen.add(id(pipeline))
            pipeline.close()
