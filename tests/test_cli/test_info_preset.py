"""Tests for ``qr-sampler info preset <id>``."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli

# V7_GDT_R05_02 "GDT bell-lift" hyperparameters that must appear in info
# output (replaced the V6_HVD_R01_01 HVH-Drift tuning on 2026-08-26).
V7_HYPERPARAM_VALUES = [
    "0.9",  # gdt_t_base
    "1.5",  # gdt_t_peak
    "0.397",  # gdt_mu
    "0.283",  # gdt_sigma
    "0.952",  # gdt_alpha
    "10.0",  # gdt_lambda_vh
    "0.009635",  # gdt_min_p_base
    "0.081",  # gdt_min_p_scale
]


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestInfoPreset:
    """Tests for ``qr-sampler info preset``."""

    def test_info_creative_sampling_shows_v7_values(self, runner: CliRunner) -> None:
        """Info output includes strategy, every V7 hyperparameter, and the experimental label."""
        result = runner.invoke(cli, ["info", "preset", "creative_sampling"])
        assert result.exit_code == 0, result.output
        assert "gdt" in result.output
        assert "experimental" in result.output.lower()
        for value in V7_HYPERPARAM_VALUES:
            assert value in result.output, (
                f"V7 hyperparameter value {value!r} not in info output: {result.output!r}"
            )

    def test_info_normal_t1_shows_baseline(self, runner: CliRunner) -> None:
        """Info for normal_t1 shows fixed strategy and T=1, without the experimental label."""
        result = runner.invoke(cli, ["info", "preset", "normal_t1"])
        assert result.exit_code == 0, result.output
        assert "fixed" in result.output
        assert "1.0" in result.output
        assert "experimental" not in result.output.lower()

    def test_info_unknown_preset_exits_nonzero(self, runner: CliRunner) -> None:
        """Unknown preset id produces a nonzero exit with a helpful message."""
        result = runner.invoke(cli, ["info", "preset", "nonexistent_preset"])
        assert result.exit_code != 0
        assert "nonexistent_preset" in result.output
