"""Tests for the duck-typed strategy seams in SamplingPipeline.

Two seams (2026-07, v7 research program):

- ``diagnostics["transformed_logits"]``: a strategy publishing fully
  transformed logits has the selector operate on those instead of the raw
  logits (mixture-of-temperatures / ring-buffer-AR families).
- ``observe_selected_token``: per-request stateful strategies receive the
  selected token id after every selection (one-token structural lag).

Absent both, pipeline behavior is byte-identical to the pre-seam code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qr_sampler.config import QRSamplerConfig
from qr_sampler.core.pipeline import build_pipeline
from qr_sampler.temperature.base import TemperatureResult, TemperatureStrategy


def _make_config(**overrides: Any) -> QRSamplerConfig:
    defaults: dict[str, Any] = {
        "entropy_source_type": "mock_uniform",
        "fallback_mode": "error",
        "log_level": "none",
    }
    defaults.update(overrides)
    return QRSamplerConfig(_env_file=None, **defaults)  # type: ignore[call-arg]


class _ForcingStrategy(TemperatureStrategy):
    """Publishes transformed logits that force selection of one token."""

    def __init__(self, forced_token: int) -> None:
        self._forced_token = forced_token

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        transformed = np.full(logits.size, -np.inf)
        transformed[self._forced_token] = 0.0
        return TemperatureResult(
            temperature=1.0,
            shannon_entropy=0.0,
            diagnostics={"transformed_logits": transformed},
        )


class _ObservingStrategy(TemperatureStrategy):
    """Records every token id fed back through the selection hook."""

    def __init__(self) -> None:
        self.observed: list[int] = []

    def observe_selected_token(self, token_id: int) -> None:
        self.observed.append(token_id)

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        return TemperatureResult(temperature=1.0, shannon_entropy=0.0, diagnostics={})


class TestTransformedLogitsSeam:
    def test_selector_operates_on_transformed_logits(self) -> None:
        """Raw logits favor token 0 overwhelmingly; the transform forces 3."""
        config = _make_config()
        pipeline = build_pipeline(config, vocab_size=5)
        logits = np.array([50.0, 0.0, 0.0, 0.0, 0.0])
        result = pipeline.sample_token(logits, strategy=_ForcingStrategy(forced_token=3))
        assert result.token_id == 3
        pipeline.close()

    def test_absent_key_uses_raw_logits(self) -> None:
        config = _make_config()
        pipeline = build_pipeline(config, vocab_size=5)
        logits = np.array([50.0, -50.0, -50.0, -50.0, -50.0])
        result = pipeline.sample_token(logits)
        assert result.token_id == 0
        pipeline.close()

    def test_mix_temperatures_end_to_end(self) -> None:
        """The mixture family samples through the seam without error."""
        config = _make_config(temperature_strategy="mix_temperatures")
        pipeline = build_pipeline(config, vocab_size=8)
        logits = np.array([2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0])
        result = pipeline.sample_token(logits)
        assert 0 <= result.token_id < 8
        assert result.record.temperature_used == 1.0
        pipeline.close()


class TestObserveSelectedTokenSeam:
    def test_hook_receives_each_selected_token(self) -> None:
        config = _make_config()
        pipeline = build_pipeline(config, vocab_size=5)
        strategy = _ObservingStrategy()
        logits = np.array([50.0, -50.0, -50.0, -50.0, -50.0])
        first = pipeline.sample_token(logits, strategy=strategy)
        second = pipeline.sample_token(logits, strategy=strategy)
        assert strategy.observed == [first.token_id, second.token_id]
        pipeline.close()

    def test_ring_buffer_ar_accumulates_history_end_to_end(self) -> None:
        config = _make_config(temperature_strategy="ring_buffer_ar", rba_lam=2.0)
        pipeline = build_pipeline(config, vocab_size=6)
        strategy = pipeline.strategy
        logits = np.array([50.0, -50.0, -50.0, -50.0, -50.0, -50.0])
        first = pipeline.sample_token(logits)
        # Second call: the previously selected token is now buffered and
        # its logit is penalized (visible in the strategy diagnostics).
        second = pipeline.sample_token(logits)
        assert first.token_id == 0
        assert second.token_id == 0  # penalty (0.7 logits) cannot flip a 100-logit gap
        result = strategy.compute_temperature(logits, config)
        assert result.diagnostics["n_buffered"] >= 1
        pipeline.close()
