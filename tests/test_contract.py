"""Drift guard for ``qr_sampler.contract`` — the cross-repo seam.

``qr-llm-qthought`` imports exclusively through :mod:`qr_sampler.contract`
(see that module's docstring). This test pins the three things a qthought-side
change would silently break on:

1. The exact set of names re-exported (``__all__``) — an accidental removal
   or rename here is exactly the "partial rename" failure mode a cross-repo
   seam exists to catch at the qr-sampler side, before it ever reaches
   qthought's CI.
2. The exact field dicts of the four qthought lane presets — these encode a
   scientific lineage (see ``presets.py``) that must not drift silently.
3. The public signature shape of :class:`~qr_sampler.qthought.QthoughtRoller`
   and :class:`~qr_sampler.qthought.ChoiceProvenance` — a signature change
   here is a breaking change for qthought even though nothing in qr-sampler's
   own suite would catch it.
"""

from __future__ import annotations

import dataclasses
import inspect

from qr_sampler import contract
from qr_sampler.config import BUILTIN_PRESETS

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
    "PER_REQUEST_FIELDS",
    "PRESET_QTHOUGHT",
    "PRESET_QTHOUGHT_THINK",
    "PRESET_QTHOUGHT_VOICE",
    "PRESET_QTHOUGHT_PURITY",
    "EntropySource",
    "MockUniformSource",
    "FallbackEntropySource",
    "build_entropy_source",
    "DrawMeta",
    "EntropyUnavailableError",
    "ConfigValidationError",
]


def test_contract_all_is_frozen() -> None:
    """``contract.__all__`` matches the pinned list exactly (order included)."""
    assert contract.__all__ == _EXPECTED_ALL


def test_contract_version_is_two() -> None:
    """v2 (2026-07): qthought lanes moved to server-integrated draws. Bump both
    this pin and qthought's import-time assert together on a break."""
    assert contract.CONTRACT_VERSION == 3


def test_per_request_fields_is_the_derived_frozenset() -> None:
    """``PER_REQUEST_FIELDS`` (additive export for qthought's sampler registry)
    is the metadata-derived frozenset and covers the per-lane override keys
    qthought curates against it."""
    assert isinstance(contract.PER_REQUEST_FIELDS, frozenset)
    assert {"sample_count", "signal_amplifier_type", "entropy_source_type"} <= (
        contract.PER_REQUEST_FIELDS
    )


def test_server_draw_surface_is_exported() -> None:
    """``build_entropy_source`` / ``DrawMeta`` (additive export for qthought's
    server-integrated dispose draw, F7.2) cross the seam: the factory is
    callable and the meta type is the frozen dataclass ``get_draw`` returns."""
    import dataclasses as _dc

    assert callable(contract.build_entropy_source)
    assert _dc.is_dataclass(contract.DrawMeta)
    assert "z" in {f.name for f in _dc.fields(contract.DrawMeta)}


def test_every_exported_name_is_importable_from_contract() -> None:
    """Every name in ``__all__`` actually resolves on the module (no typos)."""
    for name in contract.__all__:
        assert hasattr(contract, name), f"contract.__all__ lists {name!r} but it is not defined"


# ── 2. Preset dicts are pinned ────────────────────────────────────────────


def test_preset_qthought_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT] == {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "draw_block_bytes": 102400,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    }


def test_preset_qthought_think_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT_THINK] == {
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
        "draw_block_bytes": 102400,
        "sample_count": 6000,
        "zscore_calibration_samples": 200,
    }


def test_preset_qthought_voice_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT_VOICE] == {
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
        "draw_block_bytes": 102400,
        "sample_count": 10000,
        "zscore_calibration_samples": 200,
    }


def test_preset_qthought_purity_dict_pinned() -> None:
    assert BUILTIN_PRESETS[contract.PRESET_QTHOUGHT_PURITY] == {
        "entropy_source_type": "quantum_grpc",
        "signal_amplifier_type": "server",
        "temperature_strategy": "coherence_gate",
        "coherence_inner_strategy": "fixed",
        "fixed_temperature": 1.0,
        "coherence_threshold": 3.5,
        "coherence_t_boost_max": 0.5,
        "coherence_ema_alpha": 0.3,
        "draw_block_bytes": 102400,
        "top_k": 0,
        "top_p": 1.0,
    }


def test_preset_name_constants_match_dict_keys() -> None:
    """The four constants are exactly the keys they name (no drift between them)."""
    assert contract.PRESET_QTHOUGHT == "qthought"
    assert contract.PRESET_QTHOUGHT_THINK == "qthought_think"
    assert contract.PRESET_QTHOUGHT_VOICE == "qthought_voice"
    assert contract.PRESET_QTHOUGHT_PURITY == "qthought_purity"


# ── 3. QthoughtRoller / ChoiceProvenance signature snapshots ─────────────

_ROLLER_METHOD_NAMES = (
    "choose",
    "choose_weighted",
    "coin",
    "bind_int",
    "drain",
    "status",
    "begin_thought",
    "draw_u",
    "draw_index",
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
        "draw_u": "(self) -> 'ChoiceProvenance'",
        "draw_index": "(self, k: 'int') -> 'ChoiceProvenance'",
    }


def test_qthought_roller_ctor_has_entropy_source_kwarg() -> None:
    """The explicit-injection ctor seam ``entropy_source=`` is present and keyword-only."""
    signature = inspect.signature(contract.QthoughtRoller.__init__)
    assert str(signature) == (
        "(self, config: 'QRSamplerConfig | None' = None, *, "
        "entropy_source: 'EntropySource | None' = None) -> 'None'"
    )


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
