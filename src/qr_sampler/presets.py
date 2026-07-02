"""Named sampling presets.

A preset is a named bundle of per-request overrides. Two built-ins ship
with qr-sampler:

- ``creative_sampling``: HVH-Drift dynamic temperature + min-p (the
  V6_HVD_R01_01 winner from createmp-evalsuite). Experimental.
- ``normal_t1``: vanilla fixed temperature 1.0 baseline.

Resolution flow (engine-agnostic; see CLAUDE.md invariant 15):

1. A caller supplies ``qr_preset=<name>`` in ``extra_args`` (per-request),
   or sets ``QR_PRESET=<name>`` in the environment (process default,
   ingested via ``QRSamplerConfig.preset``).
2. ``resolve_config`` calls :func:`expand_extra_args` before
   ``validate_extra_args`` runs. The preset's field overrides are merged
   underneath any caller-supplied ``qr_*`` keys (caller wins per FR-10),
   and ``qr_preset`` is stripped from the output.
3. The resulting ``qr_*`` dict flows through the usual
   ``validate_extra_args`` -> ``_PER_REQUEST_FIELDS`` merge path.

``BUILTIN_PRESETS`` is the runtime source of truth (CLAUDE.md invariant
16: profile YAML in ``profiles/presets/`` is documentation only; a
sync test guards against drift).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from qr_sampler.exceptions import ConfigValidationError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig

#: Preset-name constants for the three qthought lanes, re-exported via
#: ``qr_sampler.contract`` as the canonical spelling qthought binds against
#: (rather than hand-copied string literals that could drift from the keys
#: below).
PRESET_QTHOUGHT: Final[str] = "qthought"
PRESET_QTHOUGHT_THINK: Final[str] = "qthought_think"
PRESET_QTHOUGHT_VOICE: Final[str] = "qthought_voice"

# Single source of truth for preset -> field-override mapping.
# Keys are field names (no ``qr_`` prefix); values are the override values.
# ``resolve_preset`` adds the ``qr_`` prefix when projecting into extra_args.
BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    # V6_HVD_R01_01 winner from createmp-evalsuite (results/v6/round_final).
    # Experimental — see profiles/presets/creative_sampling.yaml for origin.
    "creative_sampling": {
        "temperature_strategy": "hvh_drift",
        "hvh_t_base": 1.35,
        "hvh_alpha_h": 0.3,
        "hvh_alpha_vh": -0.2,
        "hvh_gamma_dh": 1.0,
        "hvh_delta_dvh": 0.5,
        "hvh_lambda_ema": 0.02,
        "hvh_min_p_base": 0.025,
        "hvh_kappa_h": 0.03,
        "hvh_nu_dh": 0.02,
        "top_k": 0,
        "top_p": 1.0,
    },
    # Vanilla T=1 baseline (quantum entropy still drives selection).
    "normal_t1": {
        "temperature_strategy": "fixed",
        "fixed_temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
    },
    # Qthought decode lane (qr_sampler.qthought.QthoughtRoller). The grammar
    # makes one full-size entropy fetch per case-frame decision, each reduced to
    # a uniform via the amplifier — same shape as one token-sampling step's
    # entropy half. Pins the quantum source and the optional thought-level
    # amplifier (zscore_thought) so a per-thought aggregate bias rides alongside
    # the unchanged per-decision draws; the lineage is explicit in config_hash.
    PRESET_QTHOUGHT: {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_thought",
        "sample_count": 10000,
    },
    # Qthought REFLECT lane — the private inner-voice / propose-speech completion.
    # Derived from creative_sampling's HVH-Drift family (the unspecified hvh_*
    # terms inherit the V6_HVD_R01_01 field defaults), with a hotter base for
    # divergent reflection and a smaller fetch. Plain zscore_mean: the
    # thought-level aggregate lives on the qthought decode lane, not the model lane.
    PRESET_QTHOUGHT_THINK: {
        "temperature_strategy": "hvh_drift",
        "hvh_t_base": 1.45,
        "top_k": 0,
        "top_p": 1.0,
        "sample_count": 6000,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
    },
    # Qthought SPEAK lane — the user-visible voice. EDT temperature with nucleus
    # + top-k truncation for fluent, focused speech (cooler and tighter than the
    # REFLECT lane). Full-size fetch on the quantum source + zscore_mean.
    PRESET_QTHOUGHT_VOICE: {
        "temperature_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "sample_count": 10000,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
    },
}


def resolve_preset(preset_name: str, extra_args: dict[str, Any]) -> dict[str, Any]:
    """Expand a preset name into a merged ``qr_*`` extra_args dict.

    Caller's ``qr_*`` keys in ``extra_args`` win over preset defaults
    (FR-10). The ``qr_preset`` key is always stripped from the output.

    Args:
        preset_name: Identifier present in :data:`BUILTIN_PRESETS`.
        extra_args: Per-request extras whose ``qr_*`` keys override the
            preset's defaults. Keys without the ``qr_`` prefix are
            passed through unchanged (they belong to other processors).

    Returns:
        A new dict with preset-derived ``qr_*`` keys merged underneath
        ``extra_args``.

    Raises:
        ConfigValidationError: If ``preset_name`` is not a known preset.
    """
    if preset_name not in BUILTIN_PRESETS:
        raise ConfigValidationError(
            f"Unknown preset {preset_name!r}; known: {sorted(BUILTIN_PRESETS)}"
        )

    overrides = BUILTIN_PRESETS[preset_name]
    merged: dict[str, Any] = {f"qr_{field}": value for field, value in overrides.items()}
    merged.update(extra_args)
    merged.pop("qr_preset", None)
    return merged


def expand_extra_args(
    extra_args: dict[str, Any] | None,
    default_config: QRSamplerConfig,
) -> dict[str, Any]:
    """Expand any preset reference in ``extra_args`` into concrete overrides.

    Resolution priority:

    1. ``qr_preset`` in ``extra_args`` (per-request preset).
    2. ``default_config.preset`` (env-var ``QR_PRESET``).
    3. Pass-through (no preset to expand).

    Args:
        extra_args: Per-request extras as supplied to ``resolve_config``.
        default_config: The base configuration, consulted for
            ``preset`` when no per-request preset is supplied.

    Returns:
        A new dict suitable for the standard ``qr_*`` merge path. The
        ``qr_preset`` key is never present in the returned dict.
    """
    if extra_args is None:
        extra_args = {}

    if "qr_preset" in extra_args:
        preset_name = extra_args["qr_preset"]
        remaining = {key: value for key, value in extra_args.items() if key != "qr_preset"}
        return resolve_preset(preset_name, remaining)

    if default_config.preset is not None:
        return resolve_preset(default_config.preset, extra_args)

    return extra_args
