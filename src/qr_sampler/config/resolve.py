"""Per-request configuration resolution and extra-args validation.

This module is the single validation point for per-request overrides:
``resolve_config`` expands presets, validates every ``qr_*`` key, and
merges the surviving overrides into a fresh, fully-revalidated
:class:`~qr_sampler.config.model.QRSamplerConfig`. Engine adapters also
call :func:`validate_extra_args` directly at request-creation time so bad
keys are rejected before the request enters the batch.

Both ``model`` and ``presets`` are imported top-level — this module sits
above them, so no import cycle exists.
"""

from __future__ import annotations

from typing import Any

from qr_sampler.config.model import ALL_FIELDS, PER_REQUEST_FIELDS, QRSamplerConfig
from qr_sampler.config.presets import expand_extra_args, resolve_preset
from qr_sampler.exceptions import ConfigValidationError


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
    is validated by :func:`~qr_sampler.config.presets.resolve_preset` —
    the single home of the preset-name check — so unknown names fail at
    the same point as unknown qr_* keys.

    Args:
        extra_args: Dictionary of extra arguments, potentially with qr_ prefix.

    Raises:
        ConfigValidationError: If any qr_* key is unknown or non-overridable,
            or if ``qr_preset`` names an unknown preset.
    """
    for key in extra_args:
        if not key.startswith("qr_"):
            continue
        if key == "qr_preset":
            # Delegates the name check to resolve_preset (single owner).
            resolve_preset(extra_args[key], {})
            continue
        field_name = _strip_prefix(key)
        if field_name not in ALL_FIELDS:
            raise ConfigValidationError(
                f"Unknown config field: '{key}' (no field '{field_name}' exists)"
            )
        if field_name not in PER_REQUEST_FIELDS:
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
    Only fields in PER_REQUEST_FIELDS are overridable. Keys without the
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
