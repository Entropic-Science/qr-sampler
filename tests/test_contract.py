"""Drift guard for ``qr_sampler.contract`` — the cross-repo seam.

``qr-llm-qthought`` imports exclusively through :mod:`qr_sampler.contract`
(see that module's docstring). This test pins the three things a qthought-side
change would silently break on:

1. The exact set of names re-exported (``__all__``) — an accidental removal
   or rename here is exactly the "partial rename" failure mode a cross-repo
   seam exists to catch at the qr-sampler side, before it ever reaches
   qthought's CI.
2. The exact field dicts of the three qthought lane presets — these encode a
   scientific lineage (see ``presets.py``) that must not drift silently.
3. The public signature shape of :class:`~qr_sampler.qthought.QthoughtRoller`
   and :class:`~qr_sampler.qthought.ChoiceProvenance` — a signature change
   here is a breaking change for qthought even though nothing in qr-sampler's
   own suite would catch it.

Note: the :meth:`QthoughtRoller.draw_u` / :meth:`QthoughtRoller.draw_index`
signature snapshots are added in a later step alongside those methods
themselves — do not pre-add them here.
"""

from __future__ import annotations

import dataclasses
import inspect

from qr_sampler import contract
from qr_sampler.presets import BUILTIN_PRESETS

# ── 1. __all__ is frozen ──────────────────────────────────────────────────

_EXPECTED_ALL = [
    "CONTRACT_VERSION",
    "QthoughtRoller",
    "ChoiceProvenance",
    "BindSpec",
    "IntRange",
    "QRSamplerConfig",
    "resolve_config",
    "resolve_preset",
    "BUILTIN_PRESETS",
    "PRESET_QTHOUGHT",
    "PRESET_QTHOUGHT_THINK",
    "PRESET_QTHOUGHT_VOICE",
    "EntropySource",
    "MockUniformSource",
    "FallbackEntropySource",
    "EntropyUnavailableError",
    "ConfigValidationError",
]


def test_contract_all_is_frozen() -> None:
    """``contract.__all__`` matches the pinned list exactly (order included)."""
    assert contract.__all__ == _EXPECTED_ALL


def test_contract_version_is_one() -> None:
    """The seam's version starts at 1; bump both this pin and qthought's on a break."""
    assert contract.CONTRACT_VERSION == 1


def test_every_exported_name_is_importable_from_contract() -> None:
    """Every name in ``__all__`` actually resolves on the module (no typos)."""
    for name in contract.__all__:
        assert hasattr(contract, name), f"contract.__all__ lists {name!r} but it is not defined"


# ── 2. Preset dicts are pinned ────────────────────────────────────────────


def test_preset_qthought_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT] == {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_thought",
        "sample_count": 10000,
    }


def test_preset_qthought_think_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT_THINK] == {
        "temperature_strategy": "hvh_drift",
        "hvh_t_base": 1.45,
        "top_k": 0,
        "top_p": 1.0,
        "sample_count": 6000,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
    }


def test_preset_qthought_voice_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT_VOICE] == {
        "temperature_strategy": "edt",
        "edt_base_temp": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "sample_count": 10000,
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "zscore_mean",
    }


def test_preset_name_constants_match_dict_keys() -> None:
    """The three constants are exactly the keys they name (no drift between them)."""
    assert contract.PRESET_QTHOUGHT == "qthought"
    assert contract.PRESET_QTHOUGHT_THINK == "qthought_think"
    assert contract.PRESET_QTHOUGHT_VOICE == "qthought_voice"


# ── 3. QthoughtRoller / ChoiceProvenance signature snapshots ─────────────

_ROLLER_METHOD_NAMES = (
    "choose",
    "choose_weighted",
    "coin",
    "bind_int",
    "drain",
    "status",
    "begin_thought",
)


def test_qthought_roller_method_signatures_pinned() -> None:
    """Public decision-method signatures are exactly what qthought expects."""
    roller_cls = contract.QthoughtRoller
    signatures = {
        name: str(inspect.signature(getattr(roller_cls, name))) for name in _ROLLER_METHOD_NAMES
    }
    assert signatures == {
        "choose": "(self, k: 'int') -> 'int'",
        "choose_weighted": "(self, weights: 'Sequence[float]') -> 'int'",
        "coin": "(self, p: 'float') -> 'bool'",
        "bind_int": "(self, spec: 'BindSpec') -> 'int'",
        "drain": "(self) -> 'tuple[ChoiceProvenance, ...]'",
        "status": "(self) -> 'dict[str, Any]'",
        "begin_thought": "(self) -> 'None'",
    }


def test_choice_provenance_field_names_pinned() -> None:
    """The provenance dataclass exposes exactly this field set, in this order."""
    field_names = tuple(f.name for f in dataclasses.fields(contract.ChoiceProvenance))
    assert field_names == (
        "kind",
        "value",
        "u",
        "z_score",
        "bias",
        "source",
        "is_fallback",
        "generation_timestamp",
        "latency_ms",
        "thought_aggregate",
    )
