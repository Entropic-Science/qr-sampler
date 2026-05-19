"""Tests for ``qr-sampler list presets`` subcommand."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestListPresets:
    """Tests for ``qr-sampler list presets``."""

    def test_list_shows_both_presets(self, runner: CliRunner) -> None:
        """Lists both built-in preset profiles, exit code 0."""
        result = runner.invoke(cli, ["list", "presets"])
        assert result.exit_code == 0, result.output
        assert "creative_sampling" in result.output
        assert "normal_t1" in result.output

    def test_experimental_flag_only_on_creative_sampling(
        self, runner: CliRunner
    ) -> None:
        """The ``experimental`` marker appears only on the creative_sampling row."""
        result = runner.invoke(cli, ["list", "presets"])
        assert result.exit_code == 0, result.output

        creative_idx = result.output.index("creative_sampling")
        normal_idx = result.output.index("normal_t1")

        if normal_idx > creative_idx:
            creative_section = result.output[creative_idx:normal_idx]
            normal_section = result.output[normal_idx:]
        else:
            normal_section = result.output[normal_idx:creative_idx]
            creative_section = result.output[creative_idx:]

        assert "experimental" in creative_section
        assert "experimental" not in normal_section
