"""Pydantic models for profile YAML validation.

Profiles are declarative metadata describing engines, entropy sources,
amplifiers, and samplers. They are read-only data used by the CLI for
validation and documentation -- they never affect runtime sampling behavior.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlatformConstraint(BaseModel):
    """Platform requirements for an engine or source."""

    model_config = ConfigDict(frozen=True)

    os: list[str] = Field(default_factory=list)
    accelerator: list[str] = Field(default_factory=list)


class DockerConfig(BaseModel):
    """Docker build configuration for an engine profile."""

    model_config = ConfigDict(frozen=True)

    base_image: str = ""
    dockerfile: str = ""


class EngineProfile(BaseModel):
    """Declarative profile for an inference engine.

    Describes platform constraints, known-working models, Docker
    configuration, and default settings.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    adapter: str
    entry_point_group: str = ""
    platform: PlatformConstraint = Field(default_factory=PlatformConstraint)
    dependencies: list[str] = Field(default_factory=list)
    docker: DockerConfig = Field(default_factory=DockerConfig)
    defaults: dict[str, Any] = Field(default_factory=dict)
    known_working_models: list[str] = Field(default_factory=list)
    known_incompatible_models: list[str] = Field(default_factory=list)


class EntropySourceProfile(BaseModel):
    """Declarative profile for an entropy source.

    Lists compatible and incompatible amplifiers, transport mechanism,
    and default configuration values.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    source_class: str
    transport: str = ""
    dependencies: list[str] = Field(default_factory=list)
    compatible_amplifiers: list[str] = Field(default_factory=list)
    known_incompatible_amplifiers: list[str] = Field(default_factory=list)
    defaults: dict[str, Any] = Field(default_factory=dict)


class AmplifierProfile(BaseModel):
    """Declarative profile for a signal amplifier.

    Describes input assumptions and entropy source compatibility.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    amplifier_class: str
    input_assumptions: str = ""
    compatible_entropy_sources: list[str] = Field(default_factory=list)
    known_incompatible_entropy_sources: list[str] = Field(default_factory=list)


class SamplerProfile(BaseModel):
    """Declarative profile for an adaptive temperature sampler.

    Describes performance characteristics and build-time constraints.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str = ""
    sampler_class: str
    performance_impact: str = "none"
    build_time_locked: bool = False
    requires_vocab_size: bool = False


class PresetProfile(BaseModel):
    """Declarative profile for a named sampling preset.

    A preset is a documented bundle of per-request overrides surfaced
    through the CLI (``qr-sampler list presets`` / ``info preset <id>``)
    and the Open WebUI user toggle. Runtime expansion lives in
    ``qr_sampler.config.presets.BUILTIN_PRESETS``; this YAML-backed profile is
    documentation that a sync test holds in lock-step with the dict.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str
    experimental: bool
    origin: str | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)
