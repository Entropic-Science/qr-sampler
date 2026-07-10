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
   ``validate_extra_args`` -> ``PER_REQUEST_FIELDS`` merge path.

``BUILTIN_PRESETS`` is the runtime source of truth (CLAUDE.md invariant
16: profile YAML in ``profiles/presets/`` is documentation only; a
sync test guards against drift).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from qr_sampler.exceptions import ConfigValidationError

if TYPE_CHECKING:
    from qr_sampler.config.model import QRSamplerConfig

#: Preset-name constants for the three qthought lanes, re-exported via
#: ``qr_sampler.contract`` as the canonical spelling qthought binds against
#: (rather than hand-copied string literals that could drift from the keys
#: below).
PRESET_QTHOUGHT: Final[str] = "qthought"
PRESET_QTHOUGHT_THINK: Final[str] = "qthought_think"
PRESET_QTHOUGHT_VOICE: Final[str] = "qthought_voice"
PRESET_QTHOUGHT_PURITY: Final[str] = "qthought_purity"

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
    # Lighter chat lane for owui and any future external caller. Fresh quantum
    # entropy drives selection (entropy_source_type=quantum_grpc via the local
    # zscore_mean amplifier — a plain per-token byte fetch + z-score, NOT the
    # server-integrated qr_purity draw), under a plain fixed T=1 with no
    # truncation. Deliberately carries NO coherence_gate: the cross-device
    # coherence statistic is a qthought scientific-lineage concern, not a
    # general chatbot one, so this lane never asks the server for the coherence
    # triple and never boosts temperature. This preset is NOT scientific
    # lineage and is referenced by the plain qr_preset string, so it does not
    # cross contract.py.
    "chat_light": {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
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
    # Qthought decode lane (the QthoughtRoller). Every grammar decision is now
    # ONE server-integrated draw (qr_purity GetDraw) — a 1 MiB baseline-
    # referenced block the server integrates against the device fingerprint —
    # not a local byte fetch + amplify. The baseline correction happens at the
    # source, so the "acorn" static-bias pinning cannot occur.
    # ``sample_count`` / ``zscore_calibration_samples`` remain ONLY for the
    # labelled degrade fallback: a dead PurityService draw falls back to a
    # calibrated local zscore so a decision is never muted (never silent).
    PRESET_QTHOUGHT: {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    },
    # Qthought REFLECT lane — the private inner-voice / propose-speech completion.
    # Every token rides a server-integrated 1 MiB draw (qr_purity GetDraw) under
    # the coherence-gated EDT temperature. This lane must emit a PARSEABLE
    # ``propose_speech`` tool call, and that constrains BOTH knobs:
    #   * top_k=50 / top_p=0.9 truncation removes the low-probability tail, and
    #   * a COOL/adaptive inner strategy (EDT ~0.8) keeps the structure intact.
    # A hot inner strategy (hvh_drift base 1.45) corrupts the tool call even WITH
    # truncation — under quantum-uniform selection the heat re-introduces garbage
    # tokens → strength parses 0.0 → the mind never speaks. So REFLECT is the
    # coherent lane; SPEAK is the wild one (see below). ``sample_count`` /
    # ``zscore_calibration_samples`` are degrade-fallback only.
    PRESET_QTHOUGHT_THINK: {
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 6000,
        "zscore_calibration_samples": 200,
    },
    # Qthought SPEAK lane — the user-visible voice, deliberately UNtruncated
    # (top_k=0 / top_p=1.0): the spoken turn is free text with no structure to
    # protect, so it samples the FULL distribution over server-integrated 1 MiB
    # draws under the coherence-gated EDT temperature — all the quantum
    # randomness and weirdness the entropy affords. Truncation stays on the
    # REFLECT lane only.
    # ``sample_count`` / ``zscore_calibration_samples`` are degrade-fallback only.
    PRESET_QTHOUGHT_VOICE: {
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 0,
        "top_p": 1.0,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 1048576,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    },
    # Qthought purity reference lane — the neutral fixed-inner-temperature
    # profile under the same coherence gate and server-integrated 1 MiB draws.
    # The three lanes above are all server-draw now; this remains the plain
    # fixed-temp reference. A draw failure degrades to fallback bytes + a local
    # zscore_mean amplifier (labelled, never silent).
    PRESET_QTHOUGHT_PURITY: {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "fixed",
        "fixed_temperature": 1.0,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "draw_block_bytes": 1048576,  # 1 MiB (was 0 = server default 2 MiB)
        "top_k": 0,
        "top_p": 1.0,
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

    # Truthiness (not ``is not None``): ``QR_PRESET=""`` in the environment
    # ingests as ``""``, which must mean "no preset" rather than a hard
    # ``Unknown preset ''`` error on every request (review fix 2026-07).
    if default_config.preset:
        return resolve_preset(default_config.preset, extra_args)

    return extra_args
