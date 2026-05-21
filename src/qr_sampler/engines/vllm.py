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
    qr_sampler = "qr_sampler.processor:QRSamplerLogitsProcessor"

The processor applies globally to all requests in a vLLM instance. Deploy
separate instances for different sampling strategies.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.amplification.registry import AmplifierRegistry
from qr_sampler.config import QRSamplerConfig, resolve_config, validate_extra_args
from qr_sampler.core.pipeline import SamplingPipeline, build_pipeline, config_hash
from qr_sampler.engines.base import EngineAdapter
from qr_sampler.engines.registry import EngineAdapterRegistry
from qr_sampler.exceptions import ConfigValidationError
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

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
    """

    __slots__ = (
        "amplifier",
        "config",
        "config_hash_str",
        "dominant_source_name",
        "pipeline",
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


@EngineAdapterRegistry.register("vllm")
class VLLMAdapter(EngineAdapter):
    """vLLM V1 LogitsProcessor that replaces token sampling with
    external-entropy-driven selection.

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

        # --- Pre-compute default state ---
        self._default_config_hash = config_hash(self._default_config)

        # --- Pre-allocate tensors ---
        self._onehot_template = self._create_onehot_template()
        self._cpu_buffer = self._create_cpu_buffer()

        # --- Per-request state ---
        # Maps request index (batch position) to its state.
        self._request_states: dict[int, _RequestState] = {}

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

    # Process-wide marker so ``[QR-SAMPLER DIAG] validate_params FIRST
    # CALL`` only fires once per Python process instead of once per
    # request. The marker is a class attribute (not instance) because
    # ``validate_params`` is a classmethod.
    _qr_diag_validate_params_seen: bool = False

    @classmethod
    def validate_params(cls, params: Any) -> None:
        """Validate ``qr_*`` keys in ``params.extra_args``.

        Wrapped in try/except so that any exception bubbling up to vLLM's
        request-creation path lands a tagged traceback in the container's
        stderr. The wrapper is load-bearing for ongoing investigation of
        the EngineCore 500 cascade documented in
        ``qr-llm-chat/docs/PHASE_K_STATUS.md`` — without it, the
        traceback is silently swallowed somewhere between vLLM and
        Modal's log aggregator.
        """
        import sys as _sys
        try:
            extra_args = getattr(params, "extra_args", None) or {}
            if not cls._qr_diag_validate_params_seen:
                cls._qr_diag_validate_params_seen = True
                print(
                    f"[QR-SAMPLER DIAG] validate_params FIRST CALL: "
                    f"extra_args_keys="
                    f"{sorted(extra_args.keys()) if isinstance(extra_args, dict) else type(extra_args).__name__}",
                    file=_sys.stderr,
                    flush=True,
                )
            if extra_args:
                validate_extra_args(extra_args)
        except BaseException:
            import traceback as _tb
            print(
                "[QR-SAMPLER DIAG] validate_params RAISED:\n"
                + "".join(_tb.format_exc()),
                file=_sys.stderr,
                flush=True,
            )
            raise

    def update_state(self, batch_update: Any | None) -> None:
        """Process batch composition changes.

        Must be called every engine step before ``apply()``. Processes
        changes in the required order: removed -> moved -> added.

        Args:
            batch_update: A ``BatchUpdate`` with ``removed``, ``moved``,
                and ``added`` sequences, or ``None`` if no changes.
        """
        try:
            self._update_state_impl(batch_update)
        except BaseException:
            import sys as _sys
            import traceback as _tb
            print(
                "[QR-SAMPLER DIAG] update_state RAISED:\n"
                + "".join(_tb.format_exc()),
                file=_sys.stderr,
                flush=True,
            )
            raise

    def _update_state_impl(self, batch_update: Any | None) -> None:
        import sys as _sys
        if not getattr(self, "_qr_diag_update_state_seen", False):
            self._qr_diag_update_state_seen = True
            print(
                f"[QR-SAMPLER DIAG] update_state FIRST CALL: batch_update_type={type(batch_update).__name__}",
                file=_sys.stderr,
                flush=True,
            )
        if batch_update is None:
            return

        # 1. Process removals.
        for removed in getattr(batch_update, "removed", []):
            req_idx = removed if isinstance(removed, int) else getattr(removed, "req_index", None)
            if req_idx is not None:
                state = self._request_states.pop(req_idx, None)
                if state is not None:
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

            self._request_states[req_idx] = _RequestState(
                pipeline=target_pipeline,
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
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
        import sys as _sys

        # DIAG: print on first call so we KNOW apply is invoked.
        if not getattr(self, "_qr_diag_apply_seen", False):
            self._qr_diag_apply_seen = True
            try:
                _shape = getattr(logits, "shape", None)
                _type = type(logits).__name__
                _dtype = getattr(logits, "dtype", None)
            except Exception:
                _shape = "<err>"; _type = "<err>"; _dtype = "<err>"
            print(
                f"[QR-SAMPLER DIAG] apply FIRST CALL: type={_type} shape={_shape} dtype={_dtype}",
                file=_sys.stderr,
                flush=True,
            )

        # BYPASS escape hatch: set QR_SAMPLER_BYPASS=1 in the env to make
        # apply() a no-op (returns logits unchanged) — useful for isolating
        # whether our processor or vLLM itself is the cause of an
        # EngineCore crash. Single-line revert: set the env to 0 or remove
        # it from the Modal Secret.
        if os.environ.get("QR_SAMPLER_BYPASS") == "1":
            if not getattr(self, "_qr_diag_bypass_logged", False):
                self._qr_diag_bypass_logged = True
                print(
                    "[QR-SAMPLER DIAG] QR_SAMPLER_BYPASS=1 -> apply() is a no-op",
                    file=_sys.stderr,
                    flush=True,
                )
            return logits

        try:
            return self._apply_impl(logits)
        except BaseException:
            import traceback as _tb
            print(
                "[QR-SAMPLER DIAG] apply RAISED:\n"
                + "".join(_tb.format_exc()),
                file=_sys.stderr,
                flush=True,
            )
            raise

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
            if state is not None:
                pipeline = state.pipeline
                req_config: QRSamplerConfig | None = state.config
                amplifier: SignalAmplifier | None = state.amplifier
                strategy: TemperatureStrategy | None = state.strategy
                hash_str: str | None = state.config_hash_str
                state.tokens_generated += 1
            else:
                pipeline = self._pipeline
                req_config = None
                amplifier = None
                strategy = None
                hash_str = None

            # --- Extract row as numpy ---
            if is_1d:
                row = logits if is_numpy else self._to_numpy(logits)
            else:
                row = logits[i] if is_numpy else self._to_numpy(logits[i])

            # --- Delegate to pipeline (routed by entropy source) ---
            result = pipeline.sample_token(
                row,
                config=req_config,
                amplifier=amplifier,
                strategy=strategy,
                config_hash_str=hash_str,
            )

            # --- Force one-hot logits using engine tensor ---
            if is_1d:
                self._force_onehot(logits, result.token_id, is_numpy)
            else:
                self._force_onehot_row(logits, i, result.token_id, is_numpy)

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
