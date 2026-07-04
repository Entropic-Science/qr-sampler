"""Cross-process telemetry IPC for qr-sampler.

Home of the status-file plumbing (stdlib-only JSON files under the
system temp dir) that lets out-of-process health readers observe the
sampling process's entropy/perf state. This is telemetry infrastructure,
not an entropy source — hence its own package.
"""

from qr_sampler.telemetry.status_file import (
    perf_file_path,
    read_entropy_status,
    read_perf_status,
    status_file_path,
    write_entropy_status,
    write_gate_status,
    write_perf_status,
)

__all__ = [
    "perf_file_path",
    "read_entropy_status",
    "read_perf_status",
    "status_file_path",
    "write_entropy_status",
    "write_gate_status",
    "write_perf_status",
]
