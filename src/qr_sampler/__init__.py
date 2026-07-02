"""qr-sampler: Plug any randomness source into LLM token sampling.

An engine-agnostic framework that replaces standard token sampling with
external-entropy-driven selection. Supports quantum random number generators,
processor timing jitter, and any user-supplied entropy source via gRPC.
Ships with a vLLM V1 adapter out of the box; other engines supported via
the EngineAdapter plugin system.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("qr-sampler")
except PackageNotFoundError:
    __version__ = "0.0.0"

from qr_sampler.config import (
    BUILTIN_PRESETS,
    QRSamplerConfig,
    expand_extra_args,
    resolve_config,
    resolve_preset,
    validate_extra_args,
)
from qr_sampler.core import SamplingPipeline, SamplingResult, build_pipeline
from qr_sampler.engines.base import EngineAdapter
from qr_sampler.exceptions import (
    ConfigValidationError,
    EntropyUnavailableError,
    QRSamplerError,
    SignalAmplificationError,
    TokenSelectionError,
)

__all__ = [
    "BUILTIN_PRESETS",
    "ConfigValidationError",
    "EngineAdapter",
    "EntropyUnavailableError",
    "QRSamplerConfig",
    "QRSamplerError",
    "SamplingPipeline",
    "SamplingResult",
    "SignalAmplificationError",
    "TokenSelectionError",
    "__version__",
    "build_pipeline",
    "expand_extra_args",
    "resolve_config",
    "resolve_preset",
    "validate_extra_args",
]
