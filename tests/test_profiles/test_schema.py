"""Tests for profile schema Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qr_sampler.profiles.schema import (
    AmplifierProfile,
    DockerConfig,
    EngineProfile,
    EntropySourceProfile,
    PlatformConstraint,
    PresetProfile,
    SamplerProfile,
)


class TestPlatformConstraint:
    """Tests for PlatformConstraint model."""

    def test_defaults(self) -> None:
        pc = PlatformConstraint()
        assert pc.os == []
        assert pc.accelerator == []

    def test_with_values(self) -> None:
        pc = PlatformConstraint(os=["linux"], accelerator=["nvidia"])
        assert pc.os == ["linux"]
        assert pc.accelerator == ["nvidia"]

    def test_frozen(self) -> None:
        pc = PlatformConstraint(os=["linux"])
        with pytest.raises(ValidationError):
            pc.os = ["darwin"]  # type: ignore[misc]


class TestDockerConfig:
    """Tests for DockerConfig model."""

    def test_defaults(self) -> None:
        dc = DockerConfig()
        assert dc.base_image == ""
        assert dc.dockerfile == ""

    def test_with_values(self) -> None:
        dc = DockerConfig(base_image="vllm/vllm-openai:latest", dockerfile="Dockerfile")
        assert dc.base_image == "vllm/vllm-openai:latest"
        assert dc.dockerfile == "Dockerfile"

    def test_frozen(self) -> None:
        dc = DockerConfig()
        with pytest.raises(ValidationError):
            dc.base_image = "new"  # type: ignore[misc]


class TestEngineProfile:
    """Tests for EngineProfile model."""

    def test_minimal_valid(self) -> None:
        ep = EngineProfile(id="test", name="Test Engine", adapter="mod:Cls")
        assert ep.id == "test"
        assert ep.name == "Test Engine"
        assert ep.adapter == "mod:Cls"
        assert ep.description == ""
        assert ep.platform.os == []
        assert ep.docker.base_image == ""
        assert ep.known_working_models == []
        assert ep.known_incompatible_models == []
        assert ep.defaults == {}

    def test_full_valid(self) -> None:
        ep = EngineProfile(
            id="vllm",
            name="vLLM",
            description="Standard vLLM",
            adapter="qr_sampler.engines.vllm:VLLMAdapter",
            entry_point_group="vllm.logits_processors",
            platform=PlatformConstraint(os=["linux"], accelerator=["nvidia"]),
            dependencies=["vllm"],
            docker=DockerConfig(base_image="vllm/vllm-openai:latest"),
            defaults={"dtype": "half"},
            known_working_models=["Qwen/Qwen2.5-1.5B-Instruct"],
            known_incompatible_models=[],
        )
        assert ep.platform.os == ["linux"]
        assert ep.dependencies == ["vllm"]
        assert ep.known_working_models == ["Qwen/Qwen2.5-1.5B-Instruct"]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            EngineProfile(id="x", name="X")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        ep = EngineProfile(id="test", name="Test", adapter="m:C")
        with pytest.raises(ValidationError):
            ep.id = "other"  # type: ignore[misc]


class TestEntropySourceProfile:
    """Tests for EntropySourceProfile model."""

    def test_minimal_valid(self) -> None:
        esp = EntropySourceProfile(id="sys", name="System", source_class="mod:Cls")
        assert esp.id == "sys"
        assert esp.transport == ""
        assert esp.compatible_amplifiers == []
        assert esp.known_incompatible_amplifiers == []

    def test_full_valid(self) -> None:
        esp = EntropySourceProfile(
            id="quantum_grpc",
            name="Quantum gRPC",
            description="Remote entropy via gRPC",
            source_class="qr_sampler.entropy.qgrpc:QuantumGrpcSource",
            transport="grpc",
            dependencies=["grpcio", "protobuf"],
            compatible_amplifiers=["zscore_mean", "ecdf"],
            known_incompatible_amplifiers=[],
            defaults={"sample_count": 20480},
        )
        assert esp.compatible_amplifiers == ["zscore_mean", "ecdf"]
        assert esp.dependencies == ["grpcio", "protobuf"]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            EntropySourceProfile(id="x", name="X")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        esp = EntropySourceProfile(id="sys", name="System", source_class="m:C")
        with pytest.raises(ValidationError):
            esp.transport = "grpc"  # type: ignore[misc]


class TestAmplifierProfile:
    """Tests for AmplifierProfile model."""

    def test_minimal_valid(self) -> None:
        ap = AmplifierProfile(id="zs", name="Z-Score", amplifier_class="mod:Cls")
        assert ap.id == "zs"
        assert ap.input_assumptions == ""
        assert ap.compatible_entropy_sources == []
        assert ap.known_incompatible_entropy_sources == []

    def test_full_valid(self) -> None:
        ap = AmplifierProfile(
            id="zscore_mean",
            name="Z-Score Mean",
            description="Z-score of byte-mean",
            amplifier_class="qr_sampler.amplification.zscore:ZScoreMeanAmplifier",
            input_assumptions="Uniform byte distribution",
            compatible_entropy_sources=["system", "quantum_grpc"],
            known_incompatible_entropy_sources=[],
        )
        assert ap.compatible_entropy_sources == ["system", "quantum_grpc"]

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            AmplifierProfile(id="x", name="X")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        ap = AmplifierProfile(id="zs", name="Z-Score", amplifier_class="m:C")
        with pytest.raises(ValidationError):
            ap.id = "other"  # type: ignore[misc]


class TestSamplerProfile:
    """Tests for SamplerProfile model."""

    def test_minimal_valid(self) -> None:
        sp = SamplerProfile(id="fix", name="Fixed", sampler_class="mod:Cls")
        assert sp.performance_impact == "none"
        assert sp.build_time_locked is False
        assert sp.requires_vocab_size is False

    def test_full_valid(self) -> None:
        sp = SamplerProfile(
            id="edt",
            name="EDT",
            description="Entropy-dependent temperature",
            sampler_class="qr_sampler.temperature.edt:EDTTemperatureStrategy",
            performance_impact="none",
            build_time_locked=False,
            requires_vocab_size=True,
        )
        assert sp.requires_vocab_size is True

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            SamplerProfile(id="x", name="X")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        sp = SamplerProfile(id="fix", name="Fixed", sampler_class="m:C")
        with pytest.raises(ValidationError):
            sp.id = "other"  # type: ignore[misc]


class TestPresetProfile:
    """Tests for PresetProfile model."""

    def test_preset_profile_validates_required_fields(self) -> None:
        pp = PresetProfile(
            id="creative_sampling",
            name="Creative Sampling",
            description="Experimental V6 winner preset.",
            experimental=True,
            origin="V6_HVD_R01_01",
            overrides={"temperature_strategy": "hvh_drift"},
        )
        assert pp.id == "creative_sampling"
        assert pp.name == "Creative Sampling"
        assert pp.experimental is True
        assert pp.origin == "V6_HVD_R01_01"
        assert pp.overrides == {"temperature_strategy": "hvh_drift"}

    def test_origin_defaults_to_none(self) -> None:
        pp = PresetProfile(
            id="normal_t1",
            name="Normal T=1",
            description="Vanilla baseline.",
            experimental=False,
            overrides={"temperature_strategy": "fixed"},
        )
        assert pp.origin is None

    def test_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            PresetProfile(id="x", name="X")  # type: ignore[call-arg]

    def test_preset_profile_is_frozen(self) -> None:
        pp = PresetProfile(
            id="creative_sampling",
            name="Creative Sampling",
            description="x",
            experimental=True,
            overrides={},
        )
        with pytest.raises(ValidationError):
            pp.id = "other"  # type: ignore[misc]

    def test_preset_profile_overrides_accepts_arbitrary_keys(self) -> None:
        pp = PresetProfile(
            id="creative_sampling",
            name="Creative Sampling",
            description="x",
            experimental=True,
            overrides={
                "temperature_strategy": "hvh_drift",
                "hvh_t_base": 1.35,
                "top_k": 0,
                "top_p": 1.0,
            },
        )
        assert pp.overrides["hvh_t_base"] == 1.35
        assert pp.overrides["top_k"] == 0
