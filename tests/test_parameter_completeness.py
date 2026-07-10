"""Per-request parameter completeness (spec §7.5 / AC-5).

The "self-containment" split turns qthought and owui into thin clients of a
single shared vLLM: every knob a lane needs must therefore be expressible
per-request via ``extra_args`` (``qr_*`` keys) and reach
``config.resolve.resolve_config`` — the shared server pins only the one
documented infrastructure exception (``fallback_mode``, §6.1).

These tests derive the per-request set from ``config/model.py`` field metadata
(``PER_REQUEST_FIELDS``, itself derived from ``json_schema_extra={"per_request":
True}``) so they stay honest against the code rather than a hand-copied list.
"""

from __future__ import annotations

import pytest

from qr_sampler.config import QRSamplerConfig, resolve_config
from qr_sampler.config.model import PER_REQUEST_FIELDS
from qr_sampler.config.resolve import validate_extra_args
from qr_sampler.exceptions import ConfigValidationError

#: The knobs each current lane must be able to set per-request (spec §7.5).
#: Stripped of the ``qr_`` prefix so they can be checked against the derived
#: ``PER_REQUEST_FIELDS`` set. ``preset`` is the special selector key handled
#: by ``resolve_preset`` (validated separately below).
_REQUIRED_LANE_FIELDS: frozenset[str] = frozenset(
    {
        "entropy_source_type",
        "draw_source_id",
        "draw_block_bytes",
        "temperature_strategy",
        "coherence_threshold",
        "coherence_t_boost_max",
        "coherence_ema_alpha",
        "coherence_inner_strategy",
    }
)


class TestPerRequestCompleteness:
    """Every lane knob is per-request overridable and reaches resolve_config."""

    def test_required_lane_fields_are_all_per_request(self) -> None:
        """Derived guard: each required knob carries the per_request marker.

        Keeps the test honest against the code — if a field silently loses
        its ``json_schema_extra={"per_request": True}`` marker, this fails
        rather than the hand-listed set drifting from reality.
        """
        missing = _REQUIRED_LANE_FIELDS - PER_REQUEST_FIELDS
        assert not missing, f"lane knobs not per-request overridable: {sorted(missing)}"

    def test_qr_preset_selector_is_accepted(self) -> None:
        """``qr_preset`` is the supported per-request preset selector."""
        # A known preset name validates; the selector key itself is accepted.
        validate_extra_args({"qr_preset": "chat_light"})

    def test_every_required_knob_reaches_resolve_config(self) -> None:
        """A full per-request payload merges through to the resolved config."""
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        extra_args = {
            "qr_preset": "chat_light",
            "qr_entropy_source_type": "quantum_grpc",
            "qr_draw_source_id": "dragonfly-0",
            "qr_draw_block_bytes": 1048576,
            "qr_temperature_strategy": "coherence_gate",
            "qr_coherence_threshold": 4.0,
            "qr_coherence_t_boost_max": 0.6,
            "qr_coherence_ema_alpha": 0.25,
            "qr_coherence_inner_strategy": "edt",
        }
        resolved = resolve_config(defaults, extra_args)
        # Caller overrides win over the preset (FR-10) and reach the instance.
        assert resolved.entropy_source_type == "quantum_grpc"
        assert resolved.draw_source_id == "dragonfly-0"
        assert resolved.draw_block_bytes == 1048576
        assert resolved.temperature_strategy == "coherence_gate"
        assert resolved.coherence_threshold == 4.0
        assert resolved.coherence_t_boost_max == 0.6
        assert resolved.coherence_ema_alpha == 0.25
        assert resolved.coherence_inner_strategy == "edt"

    @pytest.mark.parametrize("knob", sorted(_REQUIRED_LANE_FIELDS))
    def test_each_required_knob_passes_validation(self, knob: str) -> None:
        """Each knob individually survives ``validate_extra_args`` (not
        rejected as an unknown or infrastructure-only field)."""
        validate_extra_args({f"qr_{knob}": QRSamplerConfig.model_fields[knob].default})


class TestFallbackModeRejectedPerRequest:
    """``QR_FALLBACK_MODE`` is server-startup only — the documented §6.1 gap."""

    def test_fallback_mode_is_not_per_request(self) -> None:
        assert "fallback_mode" not in PER_REQUEST_FIELDS

    def test_qr_fallback_mode_rejected_in_extra_args(self) -> None:
        """A caller must not be able to disable honest fallback or force mock."""
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            validate_extra_args({"qr_fallback_mode": "mock_uniform"})

    def test_qr_fallback_mode_rejected_by_resolve_config(self) -> None:
        defaults = QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]
        with pytest.raises(ConfigValidationError, match="infrastructure field"):
            resolve_config(defaults, {"qr_fallback_mode": "mock_uniform"})
