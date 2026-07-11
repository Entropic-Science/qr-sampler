"""Tests for the RingBufferARStrategy."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig
from qr_sampler.temperature.registry import TemperatureStrategyRegistry
from qr_sampler.temperature.ring_buffer_ar import RingBufferARStrategy


def _make_config(**overrides: Any) -> QRSamplerConfig:
    return QRSamplerConfig(_env_file=None, **overrides)  # type: ignore[call-arg]


class TestRBAFallbackMode:
    """Exact-id penalty when no embedding table is attached."""

    def test_no_penalty_with_empty_buffer(self) -> None:
        strategy = RingBufferARStrategy(vocab_size=6)
        result = strategy.compute_temperature(np.zeros(6), _make_config())
        assert "transformed_logits" not in result.diagnostics
        assert result.diagnostics["n_buffered"] == 0

    def test_buffered_ids_penalized_by_lam_times_margin(self) -> None:
        config = _make_config(rba_lam=2.0, rba_threshold=0.65)
        strategy = RingBufferARStrategy(vocab_size=6)
        strategy.observe_selected_token(1)
        strategy.observe_selected_token(3)
        logits = np.zeros(6)
        result = strategy.compute_temperature(logits, config)
        transformed = result.diagnostics["transformed_logits"]
        expected_penalty = 2.0 * (1.0 - 0.65)
        assert transformed[1] == pytest.approx(-expected_penalty)
        assert transformed[3] == pytest.approx(-expected_penalty)
        assert transformed[0] == 0.0
        assert result.diagnostics["n_penalized"] == 2
        assert result.diagnostics["embeddings_attached"] is False

    def test_buffer_trimmed_to_configured_window(self) -> None:
        config = _make_config(rba_buffer_n=2)
        strategy = RingBufferARStrategy(vocab_size=6)
        for token in (0, 1, 2, 3):
            strategy.observe_selected_token(token)
        result = strategy.compute_temperature(np.zeros(6), config)
        transformed = result.diagnostics["transformed_logits"]
        # Only the last 2 ids (2, 3) survive the window.
        assert result.diagnostics["n_buffered"] == 2
        assert transformed[0] == 0.0
        assert transformed[1] == 0.0
        assert transformed[2] < 0.0
        assert transformed[3] < 0.0

    def test_static_t_and_min_p_published(self) -> None:
        config = _make_config(rba_t=1.2, rba_min_p=0.02)
        strategy = RingBufferARStrategy(vocab_size=6)
        result = strategy.compute_temperature(np.zeros(6), config)
        assert result.temperature == pytest.approx(1.2)
        assert result.diagnostics["min_p"] == pytest.approx(0.02)

    def test_t_clamped_to_guardrail_box(self) -> None:
        config = _make_config(rba_t=5.0)
        strategy = RingBufferARStrategy(vocab_size=6)
        result = strategy.compute_temperature(np.zeros(6), config)
        assert result.temperature == 2.2


class TestRBAEmbeddingMode:
    """Cosine-to-centroid penalty with an attached table."""

    def test_similar_tokens_penalized(self) -> None:
        # Tokens 0 and 1 share a direction; token 2 is orthogonal.
        table = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ]
        )
        strategy = RingBufferARStrategy(vocab_size=3)
        strategy.attach_embeddings(table)
        strategy.observe_selected_token(0)
        config = _make_config(rba_lam=1.0, rba_threshold=0.5)
        result = strategy.compute_temperature(np.zeros(3), config)
        transformed = result.diagnostics["transformed_logits"]
        # cos(e0, centroid=e0) = 1 -> penalty 0.5; near-duplicate token 1
        # is also penalized; orthogonal token 2 is not.
        assert transformed[0] == pytest.approx(-(1.0 - 0.5))
        assert transformed[1] < 0.0
        assert transformed[2] == 0.0
        assert result.diagnostics["embeddings_attached"] is True

    def test_attach_rejects_wrong_shape(self) -> None:
        strategy = RingBufferARStrategy(vocab_size=3)
        with pytest.raises(ValueError, match="shape"):
            strategy.attach_embeddings(np.zeros((4, 2)))
        with pytest.raises(ValueError, match="shape"):
            strategy.attach_embeddings(np.zeros(3))


class TestRBAStaticClone:
    def test_lam_zero_disables_gate_entirely(self) -> None:
        """rba_lam=0: fixed T/min_p, no transformed logits, ever."""
        config = _make_config(rba_lam=0.0, rba_t=1.2, rba_min_p=0.005)
        strategy = RingBufferARStrategy(vocab_size=6)
        for token in (1, 2, 3):
            strategy.observe_selected_token(token)
        result = strategy.compute_temperature(np.zeros(6), config)
        assert result.temperature == pytest.approx(1.2)
        assert result.diagnostics["min_p"] == pytest.approx(0.005)
        assert "transformed_logits" not in result.diagnostics
        assert result.diagnostics["n_penalized"] == 0


class TestRBAState:
    def test_instances_have_independent_history(self) -> None:
        config = _make_config(rba_lam=1.0)
        a = RingBufferARStrategy(vocab_size=6)
        b = RingBufferARStrategy(vocab_size=6)
        a.observe_selected_token(1)
        result_b = b.compute_temperature(np.zeros(6), config)
        assert result_b.diagnostics["n_buffered"] == 0

    def test_registered_under_ring_buffer_ar(self) -> None:
        assert TemperatureStrategyRegistry.get("ring_buffer_ar") is RingBufferARStrategy

    def test_built_instance_computes(self) -> None:
        config = _make_config(temperature_strategy="ring_buffer_ar")
        strategy = TemperatureStrategyRegistry.build(config, vocab_size=50)
        result = strategy.compute_temperature(np.zeros(50), config)
        assert result.temperature == pytest.approx(1.0)
