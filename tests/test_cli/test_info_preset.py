"""Tests for ``qr-sampler info preset <id>``."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from qr_sampler.cli.main import cli

# V6_HVD_R01_01 hyperparameters that must appear in info output.
V6_HYPERPARAM_VALUES = [
    "1.35",  # hvh_t_base
    "0.3",  # hvh_alpha_h
    "-0.2",  # hvh_alpha_vh
    "1.0",  # hvh_gamma_dh
    "0.5",  # hvh_delta_dvh
    "0.02",  # hvh_lambda_ema (also hvh_nu_dh)
    "0.025",  # hvh_min_p_base
    "0.03",  # hvh_kappa_h
]


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


class TestInfoPreset:
    """Tests for ``qr-sampler info preset``."""

    def test_info_creative_sampling_shows_v6_values(self, runner: CliRunner) -> None:
        """Info output includes strategy, every V6 hyperparameter, and the experimental label."""
        result = runner.invoke(cli, ["info", "preset", "creative_sampling"])
        assert result.exit_code == 0, result.output
        assert "hvh_drift" in result.output
        assert "experimental" in result.output.lower()
        for value in V6_HYPERPARAM_VALUES:
            assert value in result.output, (
                f"V6 hyperparameter value {value!r} not in info output: {result.output!r}"
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
