"""Tests for ``qr-sampler build`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _mock_dep_check() -> object:
    """Mock dependency checks since vllm is not installed in test env."""
    with patch(
        "qr_sampler.profiles.compatibility.CompatibilityChecker.check_dependencies",
        return_value=[],
    ):
        yield


class TestBuildCommand:
    """Tests for the build subcommand."""

    def test_dry_run_prints_compose(self, runner: CliRunner) -> None:
        """--dry-run prints docker-compose.yml without writing files."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--dry-run",
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--entropy",
                "system",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0
        assert "docker-compose.yml" in result.output
        assert "services:" in result.output
        # The generated stack is engine + optional entropy server only.
        assert "open-webui" not in result.output

    def test_dry_run_prints_env(self, runner: CliRunner) -> None:
        """--dry-run includes .env content."""
        result = runner.invoke(
            cli,
            ["build", "--dry-run", "--engine", "vllm"],
        )
        assert result.exit_code == 0
        assert ".env" in result.output
        assert "MODEL=" in result.output

    def test_output_writes_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """Build writes docker-compose.yml, .env, and .env.example to output dir."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--output",
                str(tmp_path),
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "--entropy",
                "system",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0

        assert (tmp_path / "docker-compose.yml").exists()
        assert (tmp_path / ".env").exists()
        assert (tmp_path / ".env.example").exists()

        compose_text = (tmp_path / "docker-compose.yml").read_text()
        assert "services:" in compose_text
        assert "inference:" in compose_text
        assert "open-webui" not in compose_text

    def test_grpc_entropy_includes_server(self, runner: CliRunner, tmp_path: Path) -> None:
        """When entropy source uses gRPC transport, entropy-server is included."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--output",
                str(tmp_path),
                "--engine",
                "vllm",
                "--entropy",
                "quantum_grpc",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0

        compose_text = (tmp_path / "docker-compose.yml").read_text()
        assert "entropy-server:" in compose_text

    def test_local_entropy_no_server(self, runner: CliRunner, tmp_path: Path) -> None:
        """When entropy source is local, no entropy-server service."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--output",
                str(tmp_path),
                "--engine",
                "vllm",
                "--entropy",
                "system",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0

        compose_text = (tmp_path / "docker-compose.yml").read_text()
        assert "entropy-server:" not in compose_text

    def test_force_bypasses_warnings(self, runner: CliRunner, tmp_path: Path) -> None:
        """--force allows build to proceed despite untested model warning."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--output",
                str(tmp_path),
                "--force",
                "--engine",
                "vllm",
                "--model",
                "some/untested-model",
                "--entropy",
                "system",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "docker-compose.yml").exists()

    def test_untested_model_without_force_exits_one(self, runner: CliRunner) -> None:
        """Untested model without --force exits 1."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--engine",
                "vllm",
                "--model",
                "some/untested-model",
                "--entropy",
                "system",
                "--amplifier",
                "zscore_mean",
            ],
        )
        assert result.exit_code == 1

    def test_config_file_support(self, runner: CliRunner, tmp_path: Path) -> None:
        """Build command loads stack from a YAML config file."""
        config = tmp_path / "stack.yaml"
        config.write_text(
            "engine: vllm\n"
            "model: Qwen/Qwen2.5-1.5B-Instruct\n"
            "entropy_source: system\n"
            "amplifier: zscore_mean\n"
            "sampler: fixed\n"
        )
        output = tmp_path / "output"
        result = runner.invoke(
            cli,
            ["build", "--config", str(config), "--output", str(output)],
        )
        assert result.exit_code == 0
        assert (output / "docker-compose.yml").exists()

    def test_nvidia_gpu_deploy_section(self, runner: CliRunner) -> None:
        """vLLM engine produces NVIDIA GPU deploy section in compose."""
        result = runner.invoke(
            cli,
            [
                "build",
                "--dry-run",
                "--engine",
                "vllm",
                "--model",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ],
        )
        assert result.exit_code == 0
        assert "nvidia" in result.output
