"""Cross-process entropy-status file — the EngineCore→APIServer IPC channel.

iter-53 (2026-06-09). vLLM runs the qr-sampler in two separate processes:
the **EngineCore** (logits processors, where ``FallbackEntropySource``
actually observes per-token fallbacks) and the **APIServer** (where an
out-of-process health reader can answer probes). Module globals do not
cross that boundary, so this module is the file-based bridge:
``FallbackEntropySource`` writes a small JSON snapshot of its state here
on every state transition (and throttled count refreshes during a
degraded window); any health reader in another process reads it. Both
processes share a filesystem, so a tmpdir file is the cheapest reliable
channel. Only the write side ships in-tree today; the read side is kept
so a deliberate reader can be reintroduced.

Atomicity: writes go to a same-directory temp file followed by
``os.replace`` (atomic on POSIX and Windows), so a reader can never
observe a half-written JSON document.

Best-effort throughout: a failed write or read degrades to "no status
available" — entropy fetching and request serving must never block or
raise on telemetry plumbing.

The file path is controlled by the ``QR_ENTROPY_STATUS_FILE`` env var:

* unset      → ``<tempdir>/qr_entropy_status.json`` (default ON; both
  vLLM processes inherit the same env so they agree on the path)
* empty (``""``) → disabled; writes become no-ops and reads return None
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from typing import Any

logger = logging.getLogger("qr_sampler")

_STATUS_FILE_ENV_VAR = "QR_ENTROPY_STATUS_FILE"
_DEFAULT_BASENAME = "qr_entropy_status.json"

# iter-55: second channel on the same bridge — per-stage sampling
# performance aggregates written by the VLLMAdapter (EngineCore) and
# surfaced through /health/entropy's "perf" block (APIServer). Separate
# file so the high-churn perf writes never race the load-bearing
# degraded/recovered transitions in the entropy-status file.
_PERF_FILE_ENV_VAR = "QR_SAMPLER_PERF_FILE"
_PERF_DEFAULT_BASENAME = "qr_sampler_perf_status.json"


def _resolve_path(env_var: str, default_basename: str) -> str | None:
    raw = os.environ.get(env_var)
    if raw is None:
        return os.path.join(tempfile.gettempdir(), default_basename)
    raw = raw.strip()
    return raw or None


def status_file_path() -> str | None:
    """Resolve the status-file path from the environment.

    Returns ``None`` when the channel is explicitly disabled via
    ``QR_ENTROPY_STATUS_FILE=""``. Resolved per-call (not cached at
    import) so tests can flip the env var without reloading the module.
    """
    return _resolve_path(_STATUS_FILE_ENV_VAR, _DEFAULT_BASENAME)


def perf_file_path() -> str | None:
    """Resolve the perf-status path (``QR_SAMPLER_PERF_FILE``), or ``None``."""
    return _resolve_path(_PERF_FILE_ENV_VAR, _PERF_DEFAULT_BASENAME)


def _write_json(path: str | None, payload: dict[str, Any]) -> bool:
    if path is None:
        return False
    record = dict(payload)
    record["updated_at"] = time.time()
    tmp_path = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, separators=(",", ":"))
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        # Broad on purpose (iter-55 review): a non-OSError from json.dump
        # (e.g. an unserializable payload value) must also clean up the
        # tmp file and degrade silently. Log at debug: a read-only tmpdir
        # would otherwise emit one warning per token during an outage.
        logger.debug("status write failed (%s): %s", path, exc)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        return False


def _read_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_entropy_status(payload: dict[str, Any]) -> bool:
    """Atomically persist *payload* (plus an ``updated_at`` stamp).

    Returns ``True`` on success, ``False`` when disabled or on any I/O
    failure. Never raises — this sits on the per-token sampling hot
    path's failure branch and must not add failure modes of its own.
    """
    return _write_json(status_file_path(), payload)


def read_entropy_status() -> dict[str, Any] | None:
    """Read the last-written status snapshot, or ``None``.

    ``None`` covers every miss case identically: channel disabled, file
    not yet written (EngineCore still initialising), or unparseable
    content. Callers must treat ``None`` as "no sampler state available",
    NOT as "degraded".
    """
    return _read_json(status_file_path())


def write_perf_status(payload: dict[str, Any]) -> bool:
    """Atomically persist the sampling-perf aggregate. Never raises."""
    return _write_json(perf_file_path(), payload)


def read_perf_status() -> dict[str, Any] | None:
    """Read the last sampling-perf aggregate, or ``None`` (miss/disabled)."""
    return _read_json(perf_file_path())
