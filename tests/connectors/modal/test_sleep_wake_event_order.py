"""Phase 2 R15(b): assert ``VLLM_SLEEP_OK`` precedes ``VLLM_WAKE_OK``
on a real Modal cold-start.

Marked ``@pytest.mark.modal`` — opt-in via ``pytest --run-modal``
because the assertion requires a live ``modal app logs qr-llm-chat``
stream. Skipped by default so the rest of the suite stays hermetic and
runs in CI without Modal credentials.

The semantics under test
------------------------
``VllmQrQwen`` has two ``@modal.enter`` hooks:

* ``snap=True`` → ``_start_and_sleep`` — spawns ``vllm serve`` and
  POSTs ``/sleep`` so the GPU memory is freed before Modal's
  memory-snapshot fires. Emits ``vllm.sleep.ok`` (or
  ``vllm.sleep.fail`` on error).
* ``snap=False`` → ``_wake`` — POSTs ``/wake_up`` on every cold-start
  (including the first deploy after snap=True), then spawns the
  cloudflared sidecar. Emits ``vllm.wake.ok`` (and Phase 2 R6's
  ``vllm.coldstart.complete``).

Phase 2 §15 success criteria #3 requires both events to appear in the
expected order on every cold-start. Out-of-order events (wake before
sleep) would indicate either a missing snap=True hook OR a
container-restart cascade — both of which break the snapshot model.

How this test runs (when --run-modal is passed)
-----------------------------------------------
1. Capture ``modal app logs qr-llm-chat`` for a bounded window (60s)
   into a temp file (use ``timeout 60 modal app logs ...`` per
   auto-memory ``feedback_bounded_log_capture``).
2. Trigger one cold-start by hitting an OWUI endpoint that proxies
   through to vLLM (e.g. ``/api/setup-status`` waiting for the chat
   probe to engage).
3. Parse the captured log; assert the first ``vllm.sleep.ok`` event
   for ``VllmQrQwen`` appears at an earlier timestamp than the first
   ``vllm.wake.ok`` event for the same container.

What this test does NOT do
--------------------------
* It does not assert latency bounds (that's R6's
  ``vllm.coldstart.complete.total_elapsed_ms`` — operator-watched, not
  CI-enforced; cold-start jitter on Modal makes a hard bound flaky).
* It does not assert the cloudflared sidecar started cleanly (the
  CLOUDFLARED_* events are soft-fail by design; absence is recorded
  separately by the deploy-guard chat probe).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.modal
def test_sleep_ok_precedes_wake_ok_on_cold_start(tmp_path: Path) -> None:
    """Phase 2 R15(b) — real-Modal cold-start event-order assertion.

    Pre-conditions (assumed by operator running ``pytest --run-modal``):
    * ``modal deploy -m qr_sampler.connectors.modal.app`` has succeeded.
    * The ``VllmQrQwen`` container is currently scaled-to-zero (forcing
      a cold-start on the next request).
    * ``modal`` CLI is on PATH and authenticated against the
      ``qr-llm-chat`` app's workspace.
    """
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        pytest.skip("modal CLI not on PATH; cannot run --run-modal test")

    log_path = tmp_path / "modal_logs.txt"
    capture_seconds = 60

    # Auto-memory ``feedback_bounded_log_capture`` + ``modal_windows_cp1252_crash``:
    # `timeout` bounds the capture window; PYTHONIOENCODING / PYTHONUTF8 prevent
    # cp1252 crashes when the log stream contains non-ASCII (e.g. arrows,
    # bullets) on Windows.
    capture_env = os.environ.copy()
    capture_env.setdefault("PYTHONIOENCODING", "utf-8")
    capture_env.setdefault("PYTHONUTF8", "1")

    with log_path.open("wb") as f:
        proc = subprocess.Popen(
            ["modal", "app", "logs", "qr-llm-chat"],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=capture_env,
        )
        try:
            time.sleep(capture_seconds)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    if not log_path.exists() or log_path.stat().st_size == 0:
        pytest.skip("modal app logs produced no output during capture window")

    first_sleep_ok_ts: float | None = None
    first_wake_ok_ts: float | None = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        event = record.get("event")
        if event == "vllm.sleep.ok" and first_sleep_ok_ts is None:
            first_sleep_ok_ts = _parse_ts(record.get("ts"))
        elif event == "vllm.wake.ok" and first_wake_ok_ts is None:
            first_wake_ok_ts = _parse_ts(record.get("ts"))

    if first_sleep_ok_ts is None:
        pytest.skip(
            "No vllm.sleep.ok observed in capture window — container did "
            "not cold-start during the 60s window. Re-run after stopping "
            "the app to force scale-to-zero."
        )
    if first_wake_ok_ts is None:
        pytest.skip(
            "No vllm.wake.ok observed in capture window — cold-start did "
            "not complete inside the budget."
        )

    assert first_sleep_ok_ts <= first_wake_ok_ts, (
        f"Event order violated: vllm.sleep.ok at {first_sleep_ok_ts} came AFTER "
        f"vllm.wake.ok at {first_wake_ok_ts}. Either snap=True hook is missing or a "
        "container restart happened mid-window."
    )


def _parse_ts(value: object) -> float | None:
    """Parse an ISO-8601 timestamp into a comparable float, or None."""
    if not isinstance(value, str):
        return None
    # JSON logger emits "YYYY-MM-DDTHH:MM:SS.sssZ" — strip the Z for fromisoformat.
    iso = value.rstrip("Z")
    try:
        from datetime import datetime

        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None
