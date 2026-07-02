"""Shared test fixtures for qr-sampler tests."""

from __future__ import annotations

import numpy as np
import pytest

from qr_sampler.config import QRSamplerConfig


@pytest.fixture()
def default_config() -> QRSamplerConfig:
    """Return a QRSamplerConfig with all default values.

    Uses _env_file=None to prevent .env file interference in tests.
    """
    return QRSamplerConfig(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture()
def sample_logits() -> np.ndarray:
    """Return a sample logits array for testing.

    Shape: (vocab_size=10,) with a clear probability structure:
    token 0 has the highest logit, token 9 the lowest.
    """
    return np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])


@pytest.fixture()
def batch_logits() -> np.ndarray:
    """Return a batch of logits arrays for testing.

    Shape: (batch_size=3, vocab_size=10).
    """
    return np.array(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 10.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ]
    )
