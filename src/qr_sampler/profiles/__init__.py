"""Declarative YAML profile system for component metadata.

Profiles describe engines, entropy sources, amplifiers, and samplers
as read-only YAML data. They inform CLI validation and documentation
but never affect runtime sampling behavior (invariant #16).
"""

from qr_sampler.profiles.compatibility import (
    CompatibilityChecker,
    CompatibilityReport,
    CompatibilityStatus,
)
from qr_sampler.profiles.loader import ProfileLoader
from qr_sampler.profiles.schema import (
    AmplifierProfile,
    DockerConfig,
    EngineProfile,
    EntropySourceProfile,
    PlatformConstraint,
    PresetProfile,
    SamplerProfile,
)

__all__ = [
    "AmplifierProfile",
    "CompatibilityChecker",
    "CompatibilityReport",
    "CompatibilityStatus",
    "DockerConfig",
    "EngineProfile",
    "EntropySourceProfile",
    "PlatformConstraint",
    "PresetProfile",
    "ProfileLoader",
    "SamplerProfile",
]
