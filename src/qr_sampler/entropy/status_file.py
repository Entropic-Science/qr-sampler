"""Cross-process entropy-status file — the EngineCore→APIServer IPC channel.

iter-53 (2026-06-09). vLLM runs the qr-sampler in two separate processes:
the **EngineCore** (logits processors, where ``FallbackEntropySource``
actually observes per-token fallbacks) and the **APIServer** (FastAPI,
where the ``/health/entropy`` middleware answers health probes). Module
globals do not cross that boundary — ``set_fallback_source()`` called in
EngineCore is invisible to the middleware (see the
``health_entropy_middleware`` module docstring for the iter-49 history of
this limitation).

This module is the file-based bridge sketched in that docstring:
``FallbackEntropySource`` writes a small JSON snapshot of its state here
on every state transition (and throttled count refreshes during a
degraded window); the middleware reads it on each ``/health/entropy``
hit. Both processes share the container filesystem, so a tmpdir file is
the cheapest reliable channel.

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


def status_file_path() -> str | None:
    """Resolve the status-file path from the environment.

    Returns ``None`` when the channel is explicitly disabled via
    ``QR_ENTROPY_STATUS_FILE=""``. Resolved per-call (not cached at
    import) so tests can flip the env var without reloading the module.
    """
    raw = os.environ.get(_STATUS_FILE_ENV_VAR)
    if raw is None:
        return os.path.join(tempfile.gettempdir(), _DEFAULT_BASENAME)
    raw = raw.strip()
    return raw or None


def write_entropy_status(payload: dict[str, Any]) -> bool:
    """Atomically persist *payload* (plus an ``updated_at`` stamp).

    Returns ``True`` on success, ``False`` when disabled or on any I/O
    failure. Never raises — this sits on the per-token sampling hot
    path's failure branch and must not add failure modes of its own.
    """
    path = status_file_path()
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
    except OSError as exc:
        # Log at debug: a read-only tmpdir would otherwise emit one
        # warning per token during a degraded window.
        logger.debug("entropy status write failed (%s): %s", path, exc)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        return False


def read_entropy_status() -> dict[str, Any] | None:
    """Read the last-written status snapshot, or ``None``.

    ``None`` covers every miss case identically: channel disabled, file
    not yet written (EngineCore still initialising), or unparseable
    content. Callers must treat ``None`` as "no sampler state available",
    NOT as "degraded".
    """
    path = status_file_path()
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None
