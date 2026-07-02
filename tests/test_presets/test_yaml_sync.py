"""Sync test between BUILTIN_PRESETS (runtime) and YAML profiles (docs).

CLAUDE.md invariant 16: profile loading never affects runtime sampling.
``BUILTIN_PRESETS`` in ``qr_sampler.config.presets`` is the runtime source of
truth; the YAML files in ``qr_sampler.profiles.presets`` are
documentation. This test is the only guard against drift between them.
"""

from __future__ import annotations

import pytest

from qr_sampler.config import BUILTIN_PRESETS
from qr_sampler.profiles.loader import ProfileLoader


@pytest.fixture()
def loader() -> ProfileLoader:
    """Loader using the real built-in profiles."""
    return ProfileLoader()


def test_builtin_presets_match_yaml(loader: ProfileLoader) -> None:
    """For every BUILTIN_PRESET, the YAML override block must match exactly.

    Checks key-for-key and value-for-value equality so any drift between
    the runtime dict and the documentation YAML is caught immediately.
    """
    for preset_name, overrides in BUILTIN_PRESETS.items():
        profile = loader.load_preset(preset_name)
        assert profile.overrides == overrides, (
            f"Preset {preset_name!r}: YAML overrides "
            f"{profile.overrides!r} do not match BUILTIN_PRESETS "
            f"{overrides!r}"
        )


def test_every_yaml_preset_has_builtin_entry(loader: ProfileLoader) -> None:
    """Every shipped YAML preset must have a corresponding runtime entry."""
    yaml_ids = {p.id for p in loader.list_presets()}
    builtin_ids = set(BUILTIN_PRESETS.keys())
    assert yaml_ids == builtin_ids, (
        f"YAML preset ids {yaml_ids} do not match BUILTIN_PRESETS ids {builtin_ids}"
    )
