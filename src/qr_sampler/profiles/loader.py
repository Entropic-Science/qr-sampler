"""Profile loader with built-in and user-override discovery.

Loads YAML profile data from:
1. Built-in profiles shipped in ``src/qr_sampler/profiles/``
2. User override directory (``$QR_PROFILES_DIR`` or ``~/.qr-sampler/profiles/``)

User profiles override built-in profiles with the same ``id``.
Profiles are cached after first load.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

from qr_sampler.profiles.schema import (
    AmplifierProfile,
    EngineProfile,
    EntropySourceProfile,
    PresetProfile,
    SamplerProfile,
)

logger = logging.getLogger("qr_sampler")

_T = TypeVar("_T", bound=BaseModel)

# Mapping from profile category to (subdirectory, model class).
_CATEGORIES: dict[str, tuple[str, type[BaseModel]]] = {
    "engines": ("engines", EngineProfile),
    "entropy": ("entropy", EntropySourceProfile),
    "amplifiers": ("amplifiers", AmplifierProfile),
    "samplers": ("samplers", SamplerProfile),
    "presets": ("presets", PresetProfile),
}


def _default_user_dir() -> Path | None:
    """Resolve the user profile override directory.

    Checks ``$QR_PROFILES_DIR`` first, then falls back to
    ``~/.qr-sampler/profiles/``. Returns ``None`` if neither exists.
    """
    env_dir = os.environ.get("QR_PROFILES_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
        return None

    home_dir = Path.home() / ".qr-sampler" / "profiles"
    if home_dir.is_dir():
        return home_dir
    return None


def _load_yaml_file(path: Path) -> dict[str, object]:
    """Load a single YAML file and return its contents as a dict.

    Raises:
        ValueError: If the file contains invalid YAML or is not a mapping.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return data


class ProfileLoader:
    """Load and cache YAML profile data.

    Profiles are loaded from:
    1. Built-in profiles shipped in ``src/qr_sampler/profiles/``
    2. User override directory (``$QR_PROFILES_DIR`` or ``~/.qr-sampler/profiles/``)

    User profiles override built-in profiles with the same ``id``.
    """

    def __init__(
        self,
        builtin_dir: Path | None = None,
        user_dir: Path | None = None,
    ) -> None:
        self._builtin_dir = builtin_dir or self._resolve_builtin_dir()
        self._user_dir = user_dir if user_dir is not None else _default_user_dir()
        # Caches keyed by (category, profile_id).
        self._cache: dict[tuple[str, str], BaseModel] = {}

    @staticmethod
    def _resolve_builtin_dir() -> Path:
        """Locate the built-in profiles directory via importlib.resources."""
        ref = importlib.resources.files("qr_sampler") / "profiles"
        # Traversable -> Path for on-disk packages.
        return Path(str(ref))

    def _discover_profiles(self, category: str) -> dict[str, Path]:
        """Discover YAML files for a category, with user overrides.

        Returns:
            Mapping of profile ``id`` (stem of filename) to resolved file path.
        """
        subdir, _ = _CATEGORIES[category]
        profiles: dict[str, Path] = {}

        # Built-in profiles.
        builtin_subdir = self._builtin_dir / subdir
        if builtin_subdir.is_dir():
            for p in sorted(builtin_subdir.glob("*.yaml")):
                profiles[p.stem] = p

        # User overrides (replace built-in by stem).
        if self._user_dir is not None:
            user_subdir = self._user_dir / subdir
            if user_subdir.is_dir():
                for p in sorted(user_subdir.glob("*.yaml")):
                    profiles[p.stem] = p

        return profiles

    def _load_profile(self, category: str, profile_id: str) -> BaseModel:
        """Load and validate a single profile by category and id."""
        cache_key = (category, profile_id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        _, model_cls = _CATEGORIES[category]
        profiles = self._discover_profiles(category)

        if profile_id not in profiles:
            available = ", ".join(sorted(profiles.keys())) or "(none)"
            raise KeyError(f"Unknown {category} profile: {profile_id!r}. Available: {available}")

        path = profiles[profile_id]
        data = _load_yaml_file(path)
        instance = model_cls.model_validate(data)
        self._cache[cache_key] = instance
        return instance

    def _list_profiles(self, category: str) -> list[BaseModel]:
        """Load and return all profiles for a category."""
        _, model_cls = _CATEGORIES[category]
        profiles = self._discover_profiles(category)
        result: list[BaseModel] = []
        for profile_id, path in sorted(profiles.items()):
            cache_key = (category, profile_id)
            if cache_key not in self._cache:
                data = _load_yaml_file(path)
                instance = model_cls.model_validate(data)
                self._cache[cache_key] = instance
            result.append(self._cache[cache_key])
        return result

    # --- Typed public API ---

    def load_engine(self, engine_id: str) -> EngineProfile:
        """Load a single engine profile by id.

        Args:
            engine_id: Profile identifier (e.g. ``'vllm'``).

        Returns:
            Validated ``EngineProfile``.

        Raises:
            KeyError: If no profile matches ``engine_id``.
        """
        return self._load_profile("engines", engine_id)  # type: ignore[return-value]

    def load_entropy_source(self, source_id: str) -> EntropySourceProfile:
        """Load a single entropy source profile by id.

        Args:
            source_id: Profile identifier (e.g. ``'quantum_grpc'``).

        Returns:
            Validated ``EntropySourceProfile``.

        Raises:
            KeyError: If no profile matches ``source_id``.
        """
        return self._load_profile("entropy", source_id)  # type: ignore[return-value]

    def load_amplifier(self, amplifier_id: str) -> AmplifierProfile:
        """Load a single amplifier profile by id.

        Args:
            amplifier_id: Profile identifier (e.g. ``'zscore_mean'``).

        Returns:
            Validated ``AmplifierProfile``.

        Raises:
            KeyError: If no profile matches ``amplifier_id``.
        """
        return self._load_profile("amplifiers", amplifier_id)  # type: ignore[return-value]

    def load_sampler(self, sampler_id: str) -> SamplerProfile:
        """Load a single sampler profile by id.

        Args:
            sampler_id: Profile identifier (e.g. ``'edt'``).

        Returns:
            Validated ``SamplerProfile``.

        Raises:
            KeyError: If no profile matches ``sampler_id``.
        """
        return self._load_profile("samplers", sampler_id)  # type: ignore[return-value]

    def list_engines(self) -> list[EngineProfile]:
        """Load and return all engine profiles."""
        return self._list_profiles("engines")  # type: ignore[return-value]

    def list_entropy_sources(self) -> list[EntropySourceProfile]:
        """Load and return all entropy source profiles."""
        return self._list_profiles("entropy")  # type: ignore[return-value]

    def list_amplifiers(self) -> list[AmplifierProfile]:
        """Load and return all amplifier profiles."""
        return self._list_profiles("amplifiers")  # type: ignore[return-value]

    def list_samplers(self) -> list[SamplerProfile]:
        """Load and return all sampler profiles."""
        return self._list_profiles("samplers")  # type: ignore[return-value]

    def load_preset(self, preset_id: str) -> PresetProfile:
        """Load a single preset profile by id.

        Args:
            preset_id: Profile identifier (e.g. ``'creative_sampling'``).

        Returns:
            Validated ``PresetProfile``.

        Raises:
            KeyError: If no profile matches ``preset_id``.
        """
        return self._load_profile("presets", preset_id)  # type: ignore[return-value]

    def list_presets(self) -> list[PresetProfile]:
        """Load and return all preset profiles."""
        return self._list_profiles("presets")  # type: ignore[return-value]
