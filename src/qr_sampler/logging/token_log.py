"""Per-request token log: one JSON line per generated token, written to disk.

A request opts in with ``qr_token_log: true`` and names its file with
``qr_request_id`` (both in ``extra_args``). The server opts in with the
``QR_TOKEN_LOG_DIR`` environment variable; without it nothing is ever written.
The file is ``<dir>/<request_id>.jsonl`` with ``/`` in the id replaced by
``__``.

The log is deliberately small. Every row carries what is needed to check that
the token is the one its uniform draw selects:

``token_index``, ``token_id``, ``timestamp_ns``, ``u_value``, ``cdf_lo``,
``cdf_hi``, ``token_rank``, ``token_prob``, ``temperature_used``,
``min_p_used``, ``num_candidates``, ``entropy_is_fallback``

The first token's row (``token_index`` 0) also carries ``entropy_source_used``
and ``top_entries``: the candidates after temperature and truncation as
``[token_id, probability]`` pairs in descending probability, cut at the
smallest set holding :data:`CANDIDATE_MASS` of the distribution. Timings,
moving averages, gate and server-draw fields stay out; they are on
:class:`~qr_sampler.logging.types.TokenSamplingRecord` for anyone who needs
them in process.

Two properties the readers rely on:

* Rows are indexed by token POSITION. vLLM samples a row and then discards it
  during partial prefill, so a position can be sampled twice; the later
  sample is the one that was emitted and it replaces the earlier row.
* A request id that is reused (a retried request) starts its file again at
  token 0.

A write failure disables the writer for that request and is logged once. It
never raises: the caller is the engine worker, and an exception there would
take the shared engine down.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qr_sampler.logging.types import TokenSamplingRecord

logger = logging.getLogger("qr_sampler")

LOG_DIR_ENV = "QR_TOKEN_LOG_DIR"

#: Share of the probability mass the first token's candidate list must hold.
CANDIDATE_MASS = 0.9995
#: Hard ceiling on the candidate list, whatever the mass says.
MAX_CANDIDATES = 4096
#: Significant digits kept for every float in a row.
FLOAT_DIGITS = 10


def token_log_dir() -> Path | None:
    """The directory from ``QR_TOKEN_LOG_DIR``, or ``None`` when logging is off."""
    raw = os.environ.get(LOG_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def log_path(directory: Path, request_id: str) -> Path:
    """``<directory>/<request_id with '/' as '__'>.jsonl``."""
    return directory / (request_id.replace("/", "__") + ".jsonl")


def _num(value: float | None) -> float | None:
    return None if value is None else float(f"{float(value):.{FLOAT_DIGITS}g}")


def trim_candidates(
    entries: Sequence[tuple[int, float]],
    *,
    mass: float = CANDIDATE_MASS,
    limit: int = MAX_CANDIDATES,
) -> list[list[float | int]]:
    """The leading candidates that together hold ``mass`` of the distribution.

    ``entries`` are ``(token_id, probability)`` in descending probability.
    """
    kept: list[list[float | int]] = []
    total = 0.0
    for token_id, prob in entries:
        if len(kept) >= limit:
            break
        kept.append([int(token_id), _num(prob) or 0.0])
        total += float(prob)
        if total >= mass:
            break
    return kept


def sober_row(token_index: int, record: TokenSamplingRecord) -> dict[str, Any]:
    """The JSON object written for one token."""
    row: dict[str, Any] = {
        "token_index": token_index,
        "token_id": record.token_id,
        "timestamp_ns": record.timestamp_ns,
        "u_value": _num(record.u_value),
        "cdf_lo": _num(record.cdf_lo),
        "cdf_hi": _num(record.cdf_hi),
        "token_rank": record.token_rank,
        "token_prob": _num(record.token_prob),
        "temperature_used": _num(record.temperature_used),
        "min_p_used": _num(record.min_p_used),
        "num_candidates": record.num_candidates,
        "entropy_is_fallback": record.entropy_is_fallback,
    }
    if token_index == 0:
        row["entropy_source_used"] = record.entropy_source_used
        if record.top_entries is not None:
            row["top_entries"] = trim_candidates(record.top_entries)
    return row


class TokenLogWriter:
    """Writes one request's rows; later samples of a position replace earlier ones."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._offsets: list[int] = []
        self._file: Any | None = None
        self._failed = False
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Truncating here is what restarts a reused request id at token 0.
            self._file = open(path, "wb")  # noqa: SIM115 - closed in close()
        except OSError:
            self._fail("could not open")

    @property
    def rows_written(self) -> int:
        return len(self._offsets)

    def _fail(self, what: str) -> None:
        if not self._failed:
            logger.exception(
                "token log %s: %s; this request is not logged",
                self.path,
                what,
                extra={"event": "token_log.failed", "path": str(self.path)},
            )
        self._failed = True
        self.close()

    def write(self, token_index: int, record: TokenSamplingRecord) -> None:
        """Write the row for ``token_index``, replacing it and any later row."""
        with self._lock:
            handle = self._file
            if handle is None or token_index < 0:
                return
            try:
                if token_index < len(self._offsets):
                    handle.seek(self._offsets[token_index])
                    handle.truncate()
                    del self._offsets[token_index:]
                elif token_index > len(self._offsets):
                    # A position was skipped; the file would have a gap the
                    # readers refuse, so say so instead of writing a wrong index.
                    raise OSError(f"token {token_index} arrived after {len(self._offsets)} rows")
                self._offsets.append(handle.tell())
                line = json.dumps(sober_row(token_index, record), separators=(",", ":"))
                handle.write(line.encode("utf-8") + b"\n")
                handle.flush()
            except (OSError, ValueError, TypeError):
                self._fail(f"could not write token {token_index}")

    def close(self) -> None:
        handle, self._file = self._file, None
        if handle is not None:
            try:
                handle.close()
            except OSError:
                logger.warning("token log %s: close failed", self.path)


__all__ = [
    "CANDIDATE_MASS",
    "LOG_DIR_ENV",
    "MAX_CANDIDATES",
    "TokenLogWriter",
    "log_path",
    "sober_row",
    "token_log_dir",
    "trim_candidates",
]
