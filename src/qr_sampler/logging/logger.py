"""Diagnostic logger for per-token sampling events.

Uses the standard ``logging`` module with the ``"qr_sampler"`` logger.
No ``print()`` statements. Supports three verbosity levels and an
in-memory diagnostic mode for post-hoc analysis.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.logging.types import TokenSamplingRecord

logger = logging.getLogger("qr_sampler")

# Cap on in-memory diagnostic records (iter-55). ~300 bytes/record →
# ~15 MB worst case. Oldest records are dropped once the cap is hit.
_MAX_DIAGNOSTIC_RECORDS = 50_000


class SamplingLogger:
    """Per-token diagnostic logger.

    Log levels:
        ``"none"``: No logging output. Records are still stored if
        ``diagnostic_mode=True``.

        ``"summary"``: One-line per token with key metrics (u_value,
        token_id, token_rank, temperature, entropy source).

        ``"full"``: Full JSON dump of all record fields.

    Diagnostic mode stores all records in memory for post-hoc statistical
    analysis via ``get_diagnostic_data()`` and ``get_summary_stats()``.
    """

    def __init__(self, config: QRSamplerConfig) -> None:
        """Initialize the logger from configuration.

        Args:
            config: Configuration providing ``log_level`` and ``diagnostic_mode``.
        """
        self._log_level = config.log_level
        self._diagnostic_mode = config.diagnostic_mode
        # iter-55: ring buffer instead of an unbounded list. A
        # per-request ``qr_diagnostic_mode=True`` on a long generation
        # previously accumulated records for the pipeline's lifetime
        # with no cap and no drain. Keeps the most recent records.
        self._records: deque[TokenSamplingRecord] = deque(maxlen=_MAX_DIAGNOSTIC_RECORDS)

    def log_token(self, record: TokenSamplingRecord) -> None:
        """Log a single token sampling event.

        Args:
            record: Immutable record of the sampling pipeline execution.
        """
        # Store in memory if diagnostic mode is enabled.
        if self._diagnostic_mode:
            self._records.append(record)

        # Emit log output based on level.
        if self._log_level == "none":
            return

        if self._log_level == "summary":
            # isEnabledFor guard (iter-55): in the production EngineCore
            # process this logger's effective level is WARNING, so the
            # per-token INFO call would otherwise still evaluate all its
            # arguments before being discarded.
            if not logger.isEnabledFor(logging.INFO):
                return
            logger.info(
                "token=%d rank=%d prob=%.4f u=%.6f temp=%.3f entropy=%.3f "
                "source=%s%s fetch=%.2fms total=%.2fms",
                record.token_id,
                record.token_rank,
                record.token_prob,
                record.u_value,
                record.temperature_used,
                record.shannon_entropy,
                record.entropy_source_used,
                " [FALLBACK]" if record.entropy_is_fallback else "",
                record.entropy_fetch_ms,
                record.total_sampling_ms,
            )
        elif self._log_level == "full":
            logger.info("sampling_record: %s", json.dumps(asdict(record), default=str))

    def get_diagnostic_data(self) -> list[TokenSamplingRecord]:
        """Return all stored records (requires ``diagnostic_mode=True``).

        Returns:
            List of all TokenSamplingRecord instances logged so far.
            Empty if diagnostic_mode is False.
        """
        return list(self._records)

    def get_summary_stats(self) -> dict[str, Any]:
        """Compute summary statistics over all stored records.

        Returns:
            Dictionary with aggregate stats, or empty dict if no records.
        """
        if not self._records:
            return {}

        u_values = [r.u_value for r in self._records]
        ranks = [r.token_rank for r in self._records]
        probs = [r.token_prob for r in self._records]
        fetch_times = [r.entropy_fetch_ms for r in self._records]
        total_times = [r.total_sampling_ms for r in self._records]
        fallback_count = sum(1 for r in self._records if r.entropy_is_fallback)

        n = len(self._records)
        return {
            "total_tokens": n,
            "mean_u": sum(u_values) / n,
            "min_u": min(u_values),
            "max_u": max(u_values),
            "mean_rank": sum(ranks) / n,
            "mean_prob": sum(probs) / n,
            "mean_fetch_ms": sum(fetch_times) / n,
            "mean_total_ms": sum(total_times) / n,
            "max_total_ms": max(total_times),
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / n,
        }
