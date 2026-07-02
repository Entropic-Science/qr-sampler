"""Configuration subsystem for qr-sampler.

Three modules with a strict downward dependency order:

- :mod:`~qr_sampler.config.model` — the ``QRSamplerConfig`` settings model
  and the metadata-derived field sets.
- :mod:`~qr_sampler.config.presets` — named preset bundles
  (``BUILTIN_PRESETS``) and their expansion into ``qr_*`` overrides.
- :mod:`~qr_sampler.config.resolve` — per-request resolution + validation
  (imports both of the above; nothing here imports back up).

All public names are re-exported at the package level; downstream
consumers outside this repo must import them via ``qr_sampler.contract``.
"""

from qr_sampler.config.model import ALL_FIELDS, PER_REQUEST_FIELDS, QRSamplerConfig
from qr_sampler.config.presets import (
    BUILTIN_PRESETS,
    PRESET_QTHOUGHT,
    PRESET_QTHOUGHT_THINK,
    PRESET_QTHOUGHT_VOICE,
    expand_extra_args,
    resolve_preset,
)
from qr_sampler.config.resolve import resolve_config, validate_extra_args

__all__ = [
    "ALL_FIELDS",
    "BUILTIN_PRESETS",
    "PER_REQUEST_FIELDS",
    "PRESET_QTHOUGHT",
    "PRESET_QTHOUGHT_THINK",
    "PRESET_QTHOUGHT_VOICE",
    "QRSamplerConfig",
    "expand_extra_args",
    "resolve_config",
    "resolve_preset",
    "validate_extra_args",
]
