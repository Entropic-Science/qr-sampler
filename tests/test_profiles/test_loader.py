"""Tests for ProfileLoader: built-in loading, user overrides, caching."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qr_sampler.profiles.loader import ProfileLoader
from qr_sampler.profiles.schema import (
    AmplifierProfile,
    EngineProfile,
    EntropySourceProfile,
    PresetProfile,
    SamplerProfile,
)


@pytest.fixture()
def loader() -> ProfileLoader:
    """Loader using the real built-in profiles."""
    return ProfileLoader()


class TestLoadBuiltinProfiles:
    """Test loading built-in YAML profiles."""

    def test_load_engine_vllm(self, loader: ProfileLoader) -> None:
        engine = loader.load_engine("vllm")
        assert isinstance(engine, EngineProfile)
        assert engine.id == "vllm"
        assert engine.name == "vLLM (NVIDIA GPU)"
        assert "vllm" in engine.dependencies
        assert len(engine.known_working_models) > 0

    def test_load_engine_vllm_metal(self, loader: ProfileLoader) -> None:
        engine = loader.load_engine("vllm_metal")
        assert isinstance(engine, EngineProfile)
        assert engine.id == "vllm_metal"
        assert engine.platform.os == ["darwin"]

    def test_load_entropy_system(self, loader: ProfileLoader) -> None:
        source = loader.load_entropy_source("system")
        assert isinstance(source, EntropySourceProfile)
        assert source.id == "system"
        assert "zscore_mean" in source.compatible_amplifiers

    def test_load_entropy_quantum_grpc(self, loader: ProfileLoader) -> None:
        source = loader.load_entropy_source("quantum_grpc")
        assert isinstance(source, EntropySourceProfile)
        assert source.transport == "grpc"
        assert "grpcio" in source.dependencies

    def test_load_entropy_timing_noise(self, loader: ProfileLoader) -> None:
        source = loader.load_entropy_source("timing_noise")
        assert isinstance(source, EntropySourceProfile)
        assert source.id == "timing_noise"

    def test_load_entropy_mock_uniform(self, loader: ProfileLoader) -> None:
        source = loader.load_entropy_source("mock_uniform")
        assert isinstance(source, EntropySourceProfile)
        assert source.id == "mock_uniform"

    def test_load_entropy_openentropy(self, loader: ProfileLoader) -> None:
        source = loader.load_entropy_source("openentropy")
        assert isinstance(source, EntropySourceProfile)
        assert "openentropy" in source.dependencies

    def test_load_amplifier_zscore(self, loader: ProfileLoader) -> None:
        amp = loader.load_amplifier("zscore_mean")
        assert isinstance(amp, AmplifierProfile)
        assert amp.id == "zscore_mean"
        assert "system" in amp.compatible_entropy_sources

    def test_load_amplifier_ecdf(self, loader: ProfileLoader) -> None:
        amp = loader.load_amplifier("ecdf")
        assert isinstance(amp, AmplifierProfile)
        assert amp.id == "ecdf"

    def test_load_sampler_fixed(self, loader: ProfileLoader) -> None:
        sampler = loader.load_sampler("fixed")
        assert isinstance(sampler, SamplerProfile)
        assert sampler.id == "fixed"
        assert sampler.requires_vocab_size is False

    def test_load_sampler_edt(self, loader: ProfileLoader) -> None:
        sampler = loader.load_sampler("edt")
        assert isinstance(sampler, SamplerProfile)
        assert sampler.id == "edt"
        assert sampler.requires_vocab_size is True

    def test_load_preset_creative_sampling(self, loader: ProfileLoader) -> None:
        preset = loader.load_preset("creative_sampling")
        assert isinstance(preset, PresetProfile)
        assert preset.id == "creative_sampling"
        assert preset.experimental is True
        assert preset.origin is not None
        assert "V6_HVD_R01_01" in preset.origin
        assert preset.overrides["temperature_strategy"] == "hvh_drift"
        assert preset.overrides["hvh_t_base"] == 1.35

    def test_load_preset_normal_t1(self, loader: ProfileLoader) -> None:
        preset = loader.load_preset("normal_t1")
        assert isinstance(preset, PresetProfile)
        assert preset.id == "normal_t1"
        assert preset.experimental is False
        assert preset.origin is None
        assert preset.overrides["temperature_strategy"] == "fixed"
        assert preset.overrides["fixed_temperature"] == 1.0


class TestListProfiles:
    """Test listing all profiles for each category."""

    def test_list_engines(self, loader: ProfileLoader) -> None:
        engines = loader.list_engines()
        assert len(engines) >= 2
        ids = [e.id for e in engines]
        assert "vllm" in ids
        assert "vllm_metal" in ids

    def test_list_entropy_sources(self, loader: ProfileLoader) -> None:
        sources = loader.list_entropy_sources()
        assert len(sources) >= 5
        ids = [s.id for s in sources]
        assert "system" in ids
        assert "quantum_grpc" in ids

    def test_list_amplifiers(self, loader: ProfileLoader) -> None:
        amps = loader.list_amplifiers()
        assert len(amps) >= 2
        ids = [a.id for a in amps]
        assert "zscore_mean" in ids
        assert "ecdf" in ids

    def test_list_samplers(self, loader: ProfileLoader) -> None:
        samplers = loader.list_samplers()
        assert len(samplers) >= 2
        ids = [s.id for s in samplers]
        assert "fixed" in ids
        assert "edt" in ids

    def test_list_presets_returns_both(self, loader: ProfileLoader) -> None:
        presets = loader.list_presets()
        ids = [p.id for p in presets]
        assert "creative_sampling" in ids
        assert "normal_t1" in ids


class TestMissingProfile:
    """Test error handling for nonexistent profiles."""

    def test_missing_engine(self, loader: ProfileLoader) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            loader.load_engine("nonexistent")

    def test_missing_entropy_source(self, loader: ProfileLoader) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            loader.load_entropy_source("nonexistent")

    def test_missing_amplifier(self, loader: ProfileLoader) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            loader.load_amplifier("nonexistent")

    def test_missing_sampler(self, loader: ProfileLoader) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            loader.load_sampler("nonexistent")

    def test_load_preset_missing_raises(self, loader: ProfileLoader) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            loader.load_preset("nonexistent")


class TestUserOverrides:
    """Test that user-supplied profiles override built-in ones."""

    def test_user_profile_overrides_builtin(self, tmp_path: Path) -> None:
        # Create user override directory with custom engine profile.
        engines_dir = tmp_path / "engines"
        engines_dir.mkdir()
        custom_data = {
            "id": "vllm",
            "name": "Custom vLLM Override",
            "adapter": "custom.module:CustomAdapter",
            "known_working_models": ["custom/model-1"],
        }
        yaml_path = engines_dir / "vllm.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(custom_data, f)

        loader = ProfileLoader(user_dir=tmp_path)
        engine = loader.load_engine("vllm")
        assert engine.name == "Custom vLLM Override"
        assert engine.adapter == "custom.module:CustomAdapter"
        assert engine.known_working_models == ["custom/model-1"]

    def test_user_profile_adds_new_engine(self, tmp_path: Path) -> None:
        engines_dir = tmp_path / "engines"
        engines_dir.mkdir()
        custom_data = {
            "id": "custom_engine",
            "name": "Custom Engine",
            "adapter": "custom.module:Adapter",
        }
        yaml_path = engines_dir / "custom_engine.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(custom_data, f)

        loader = ProfileLoader(user_dir=tmp_path)
        engine = loader.load_engine("custom_engine")
        assert engine.id == "custom_engine"
        assert engine.name == "Custom Engine"

    def test_nonexistent_user_dir_ignored(self) -> None:
        loader = ProfileLoader(user_dir=Path("/nonexistent/path"))
        # Should still load built-in profiles fine.
        engines = loader.list_engines()
        assert len(engines) >= 2


class TestCaching:
    """Test that profiles are cached after first load."""

    def test_repeated_load_returns_same_object(self, loader: ProfileLoader) -> None:
        engine1 = loader.load_engine("vllm")
        engine2 = loader.load_engine("vllm")
        assert engine1 is engine2

    def test_list_populates_cache(self, loader: ProfileLoader) -> None:
        _ = loader.list_engines()
        # Subsequent load should hit cache.
        engine = loader.load_engine("vllm")
        assert engine.id == "vllm"
