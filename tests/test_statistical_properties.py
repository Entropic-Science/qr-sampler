"""Statistical validation tests for the qr-sampler pipeline.

These tests verify the mathematical properties that make the pipeline
suitable for consciousness-influence research:

1. **U-value uniformity**: Under the null hypothesis (unbiased entropy),
   the amplified u-values should follow Uniform(0, 1).
2. **Bias detection**: A biased entropy source (mean != 127.5) should
   produce u-values that detectably deviate from uniform.
3. **EDT monotonicity**: Higher Shannon entropy should produce higher
   temperature values.
4. **CDF coverage**: Uniform u-values should produce tokens respecting
   the probability distribution.
5. **One-hot correctness**: Every batch row should have exactly one 0.0
   and the rest -inf after apply().

Requires scipy (dev dependency):
    pip install qr-sampler[dev]
"""

from __future__ import annotations

import numpy as np
import pytest

scipy_stats = pytest.importorskip("scipy.stats")

from qr_sampler.amplification.registry import AmplifierRegistry  # noqa: E402
from qr_sampler.config import QRSamplerConfig  # noqa: E402
from qr_sampler.entropy.mock import MockUniformSource  # noqa: E402
from qr_sampler.selection.selector import TokenSelector  # noqa: E402
from qr_sampler.temperature.registry import TemperatureStrategyRegistry  # noqa: E402


def _make_config(**overrides: object) -> QRSamplerConfig:
    """Create a config with test defaults and optional overrides."""
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestUValueUniformity:
    """Verify that u-values from the amplification pipeline follow U(0,1)
    under the null hypothesis (unbiased entropy source).

    Note: MockUniformSource generates bytes from N(mean, 40), clamped to
    [0, 255]. The population_std in the default config is 73.6 (for true
    uniform [0, 255]). To get correct z-score statistics, we use
    SystemEntropySource (os.urandom) which produces truly uniform bytes,
    or adjust the population_std to match MockUniformSource's distribution.
    """

    def test_ks_test_os_urandom(self) -> None:
        """KS-test: 1000 u-values from os.urandom should be uniform.

        Uses SystemEntropySource which produces truly uniform bytes.
        The population_std=73.61 is calibrated for uniform [0, 255],
        so the z-score pipeline should produce U(0, 1) u-values.
        """
        from qr_sampler.entropy.system import SystemEntropySource

        config = _make_config(sample_count=4096)
        source = SystemEntropySource()
        amplifier = AmplifierRegistry.build(config)

        u_values = []
        for _ in range(1000):
            raw = source.get_random_bytes(config.sample_count)
            result = amplifier.amplify(raw)
            u_values.append(result.u)

        u_arr = np.array(u_values)

        # KS-test against Uniform(0, 1).
        stat, p_value = scipy_stats.kstest(u_arr, "uniform")

        assert p_value > 0.01, (
            f"KS-test rejected uniformity: statistic={stat:.4f}, p={p_value:.4f}. "
            f"u range: [{u_arr.min():.4f}, {u_arr.max():.4f}], "
            f"mean={u_arr.mean():.4f}"
        )

    def test_mock_source_u_value_spread(self) -> None:
        """MockUniformSource produces u-values that span the (0, 1) range.

        Even though MockUniformSource's byte distribution doesn't perfectly
        match the default population_std (it uses N(127.5, 40.0) clamped to
        [0, 255] rather than true uniform), the amplification pipeline should
        still produce u-values that cover a broad range of the (0, 1) interval.
        """
        config = _make_config(sample_count=2048)
        source = MockUniformSource(mean=127.5, seed=42)
        amplifier = AmplifierRegistry.build(config)

        u_values = []
        for _ in range(500):
            raw = source.get_random_bytes(config.sample_count)
            result = amplifier.amplify(raw)
            u_values.append(result.u)

        u_arr = np.array(u_values)

        # u-values should span a reasonable range of (0, 1).
        assert u_arr.min() < 0.15, f"u min={u_arr.min():.4f} should be below 0.15"
        assert u_arr.max() > 0.85, f"u max={u_arr.max():.4f} should be above 0.85"

        # Standard deviation should indicate spread (not all clustered).
        assert u_arr.std() > 0.1, f"u std={u_arr.std():.4f} too small — values are clustered"

    def test_u_value_range(self) -> None:
        """All u-values must be in (epsilon, 1-epsilon)."""
        config = _make_config(sample_count=2048)
        source = MockUniformSource(mean=127.5, seed=123)
        amplifier = AmplifierRegistry.build(config)

        eps = config.uniform_clamp_epsilon
        for _ in range(500):
            raw = source.get_random_bytes(config.sample_count)
            result = amplifier.amplify(raw)
            assert eps <= result.u <= 1.0 - eps, (
                f"u={result.u} outside valid range ({eps}, {1 - eps})"
            )

    def test_u_mean_near_half_os_urandom(self) -> None:
        """Mean of u-values from os.urandom should be near 0.5."""
        from qr_sampler.entropy.system import SystemEntropySource

        config = _make_config(sample_count=4096)
        source = SystemEntropySource()
        amplifier = AmplifierRegistry.build(config)

        u_values = [
            amplifier.amplify(source.get_random_bytes(config.sample_count)).u for _ in range(1000)
        ]
        mean_u = np.mean(u_values)

        assert abs(mean_u - 0.5) < 0.05, (
            f"Mean u={mean_u:.4f} is too far from 0.5 for unbiased source"
        )


class TestBiasDetection:
    """Verify that biased entropy sources produce detectably non-uniform u-values."""

    def test_positive_bias_shifts_u_above_half(self) -> None:
        """Biased source (mean=130.0) should produce u-values with mean > 0.5.

        A mean of 130.0 (vs null 127.5) creates a positive z-score bias,
        which maps through the normal CDF to u-values shifted toward 1.
        """
        config = _make_config(sample_count=2048)
        source = MockUniformSource(mean=130.0, seed=42)
        amplifier = AmplifierRegistry.build(config)

        u_values = [
            amplifier.amplify(source.get_random_bytes(config.sample_count)).u for _ in range(500)
        ]
        mean_u = np.mean(u_values)

        assert mean_u > 0.5, (
            f"Biased source (mean=130) should shift u > 0.5, got mean_u={mean_u:.4f}"
        )

    def test_negative_bias_shifts_u_below_half(self) -> None:
        """Biased source (mean=125.0) should produce u-values with mean < 0.5."""
        config = _make_config(sample_count=2048)
        source = MockUniformSource(mean=125.0, seed=42)
        amplifier = AmplifierRegistry.build(config)

        u_values = [
            amplifier.amplify(source.get_random_bytes(config.sample_count)).u for _ in range(500)
        ]
        mean_u = np.mean(u_values)

        assert mean_u < 0.5, (
            f"Biased source (mean=125) should shift u < 0.5, got mean_u={mean_u:.4f}"
        )

    def test_ks_detects_strong_bias(self) -> None:
        """KS-test rejects uniformity for strongly biased source."""
        config = _make_config(sample_count=2048)
        source = MockUniformSource(mean=132.0, seed=42)
        amplifier = AmplifierRegistry.build(config)

        u_values = [
            amplifier.amplify(source.get_random_bytes(config.sample_count)).u for _ in range(500)
        ]

        stat, p_value = scipy_stats.kstest(u_values, "uniform")
        assert p_value < 0.05, (
            f"KS-test should reject uniformity for biased source: stat={stat:.4f}, p={p_value:.4f}"
        )


class TestEDTMonotonicity:
    """Verify that EDT temperature increases with Shannon entropy."""

    def test_higher_entropy_higher_temperature(self) -> None:
        """Logit distributions with increasing entropy should produce
        monotonically increasing temperatures from EDT.
        """
        vocab_size = 100
        config = _make_config(
            temperature_strategy="edt",
            edt_base_temp=1.0,
            edt_exponent=0.5,
            edt_min_temp=0.01,
            edt_max_temp=5.0,
        )
        strategy = TemperatureStrategyRegistry.build(config, vocab_size)

        # Create logit distributions with increasing entropy:
        # 1. Very peaked (low entropy).
        peaked = np.full(vocab_size, -10.0)
        peaked[0] = 10.0

        # 2. Moderately spread.
        moderate = np.linspace(5.0, -5.0, vocab_size)

        # 3. Nearly uniform (high entropy).
        uniform = np.zeros(vocab_size)

        results = [
            strategy.compute_temperature(logits, config) for logits in [peaked, moderate, uniform]
        ]

        entropies = [r.shannon_entropy for r in results]
        temperatures = [r.temperature for r in results]

        # Entropies should be strictly increasing.
        assert entropies[0] < entropies[1] < entropies[2], f"Entropies should increase: {entropies}"

        # Temperatures should be monotonically non-decreasing.
        assert temperatures[0] <= temperatures[1] <= temperatures[2], (
            f"EDT temperatures should increase with entropy: {temperatures}"
        )

    def test_edt_exponent_effects(self) -> None:
        """Different exponents change the concavity of the entropy → temp mapping."""
        vocab_size = 100
        moderate_logits = np.linspace(5.0, -5.0, vocab_size)

        # Exponent < 1 (concave): temperature rises quickly.
        config_concave = _make_config(
            temperature_strategy="edt",
            edt_base_temp=1.0,
            edt_exponent=0.3,
            edt_min_temp=0.01,
            edt_max_temp=5.0,
        )
        strategy_concave = TemperatureStrategyRegistry.build(config_concave, vocab_size)

        # Exponent > 1 (convex): temperature rises slowly.
        config_convex = _make_config(
            temperature_strategy="edt",
            edt_base_temp=1.0,
            edt_exponent=2.0,
            edt_min_temp=0.01,
            edt_max_temp=5.0,
        )
        strategy_convex = TemperatureStrategyRegistry.build(config_convex, vocab_size)

        t_concave = strategy_concave.compute_temperature(moderate_logits, config_concave)
        t_convex = strategy_convex.compute_temperature(moderate_logits, config_convex)

        # For intermediate entropy, concave exponent should give higher temp
        # than convex exponent (since H_norm is between 0 and 1).
        assert t_concave.temperature > t_convex.temperature, (
            f"Concave exponent should give higher temp for intermediate entropy: "
            f"concave={t_concave.temperature:.4f}, convex={t_convex.temperature:.4f}"
        )


class TestCDFCoverage:
    """Verify that uniform u-values produce probability-respecting token selection."""

    def test_most_probable_selected_most_often(self) -> None:
        """With uniform u-values, the most probable token should be
        selected most frequently.

        Uses high temperature and disabled top_p to ensure multiple
        tokens survive filtering with meaningful probability.
        """
        logits = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
        selector = TokenSelector()
        temperature = 1.5
        top_k = 10
        top_p = 1.0  # Disable top-p so all tokens survive.

        # Generate uniform u-values.
        rng = np.random.default_rng(42)
        selections = []
        for _ in range(2000):
            u = float(rng.uniform(0.0, 1.0))
            result = selector.select(logits, temperature, top_k, top_p, u)
            selections.append(result.token_id)

        # Count token frequencies.
        counts = np.bincount(selections, minlength=10)

        # Token 0 (highest logit) should be selected most often.
        assert counts[0] == max(counts), f"Token 0 should be most frequent: counts={counts}"

        # Frequency should decrease with rank (at least for the top 3 tokens).
        assert counts[0] > counts[1] > counts[2], (
            f"Selection frequency should decrease with rank: top 3 counts={counts[:3]}"
        )

    def test_u_zero_selects_most_probable(self) -> None:
        """u near 0 should select the most probable token."""
        logits = np.array([5.0, 3.0, 1.0, -1.0, -3.0])
        selector = TokenSelector()
        result = selector.select(logits, temperature=0.7, top_k=50, top_p=1.0, u=0.001)
        assert result.token_id == 0
        assert result.token_rank == 0

    def test_u_one_selects_low_probability_token(self) -> None:
        """u near 1 should select a lower-probability token (high rank).

        With high temperature and all tokens surviving, u=0.999 selects
        near the tail of the CDF — a token with high rank (low probability).
        """
        logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5])
        selector = TokenSelector()
        result = selector.select(logits, temperature=2.0, top_k=10, top_p=1.0, u=0.999)
        # Should select a high-rank token (toward the tail).
        assert result.token_rank >= 5, (
            f"u=0.999 should select a high-rank token, got rank={result.token_rank}"
        )


class TestHVHDriftNonDegenerate:
    """Verify HVH-Drift produces a non-degenerate token distribution.

    Spec §6.5: with quantum-uniform u-values, the empirical token-rank
    distribution must be distinguishable from both extremes:

    * **Greedy** — argmax-only (delta spike at rank 0).
    * **Uniform** — equal probability across all candidate ranks.

    This guards against accidental regressions where the preset
    collapses to one of the trivial baselines.
    """

    def test_hvh_drift_kolmogorov_smirnov_non_degenerate(self) -> None:
        from qr_sampler.temperature.hvh_drift import HVHDriftStrategy

        vocab_size = 50
        n_tokens = 1000

        # Logits with moderate spread so the distribution is non-trivial
        # but still has structure (HVH-Drift cares about H/VH drift).
        logits = np.linspace(2.0, -2.0, vocab_size, dtype=np.float64)

        config = _make_config(
            temperature_strategy="hvh_drift",
            sample_count=2048,
            top_k=0,
            top_p=1.0,
            min_p_base=0.0,
        )
        strategy = HVHDriftStrategy(vocab_size=vocab_size)
        amplifier = AmplifierRegistry.build(config)
        selector = TokenSelector()
        source = MockUniformSource(mean=127.5, seed=2026)

        ranks = np.empty(n_tokens, dtype=np.int64)
        for i in range(n_tokens):
            temp_result = strategy.compute_temperature(logits, config)
            min_p = float(temp_result.diagnostics.get("min_p", 0.0))
            raw = source.get_random_bytes(config.sample_count)
            u = amplifier.amplify(raw).u
            sel = selector.select(
                logits,
                temp_result.temperature,
                config.top_k,
                config.top_p,
                u,
                min_p=min_p,
            )
            ranks[i] = sel.token_rank

        # --- Distinguishability from GREEDY (delta at rank 0) ---
        # A greedy sampler would put every token at rank 0. HVH-Drift
        # must spread some mass to higher ranks.
        non_argmax_fraction = float(np.mean(ranks != 0))
        assert non_argmax_fraction > 0.2, (
            f"HVH-Drift selections collapsed toward greedy: "
            f"only {non_argmax_fraction:.3f} of tokens left rank 0"
        )
        assert ranks.max() > 0, "HVH-Drift never selected anything beyond argmax"

        # --- Distinguishability from UNIFORM over ranks ---
        # Normalize discrete ranks to (0, 1) and run a one-sample KS test
        # against the continuous uniform CDF. A truly uniform sampler
        # would give p > 0.05; HVH-Drift must skew toward low ranks
        # (higher-probability tokens), so this should reject decisively.
        normalized = (ranks.astype(np.float64) + 0.5) / vocab_size
        stat, p_value = scipy_stats.kstest(normalized, "uniform")
        assert p_value < 0.01, (
            f"HVH-Drift rank distribution is indistinguishable from uniform: "
            f"KS statistic={stat:.4f}, p={p_value:.4f}"
        )


class TestOneHotCorrectness:
    """Verify that the processor produces valid one-hot output."""

    def test_multi_row_batch(self) -> None:
        """Every row in a multi-row batch should have exactly one 0.0."""
        # Import here to use the helper.
        import os

        env_vars = {
            "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
            "QR_FALLBACK_MODE": "error",
            "QR_LOG_LEVEL": "none",
        }
        old = {k: os.environ.get(k) for k in env_vars}
        for k, v in env_vars.items():
            os.environ[k] = v

        try:
            from qr_sampler.processor import QRSamplerLogitsProcessor

            class _VConfig:
                vocab_size = 20

            proc = QRSamplerLogitsProcessor(vllm_config=_VConfig())

            rng = np.random.default_rng(42)
            batch_size = 10
            logits = rng.standard_normal((batch_size, 20)).astype(np.float32)
            result = proc.apply(logits)

            for i in range(batch_size):
                row = result[i]
                zeros = np.sum(row == 0.0)
                neginfs = np.sum(np.isneginf(row))
                assert zeros == 1, f"Row {i}: expected 1 zero, got {zeros}"
                assert neginfs == 19, f"Row {i}: expected 19 -inf, got {neginfs}"
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_selected_token_within_vocab(self) -> None:
        """Selected token IDs are always valid vocabulary indices."""
        import os

        env_vars = {
            "QR_ENTROPY_SOURCE_TYPE": "mock_uniform",
            "QR_FALLBACK_MODE": "error",
            "QR_LOG_LEVEL": "none",
            "QR_DIAGNOSTIC_MODE": "true",
        }
        old = {k: os.environ.get(k) for k in env_vars}
        for k, v in env_vars.items():
            os.environ[k] = v

        try:
            from qr_sampler.processor import QRSamplerLogitsProcessor

            class _VConfig:
                vocab_size = 50

            proc = QRSamplerLogitsProcessor(vllm_config=_VConfig())

            rng = np.random.default_rng(99)
            for _ in range(20):
                logits = rng.standard_normal((1, 50)).astype(np.float32)
                proc.apply(logits)

            for record in proc.sampling_logger.get_diagnostic_data():
                assert 0 <= record.token_id < 50
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
