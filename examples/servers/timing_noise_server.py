#!/usr/bin/env python3
"""gRPC entropy server using CPU timing jitter.

An educational example showing how to build a non-QRNG entropy source
as a gRPC server. Derives randomness from variations in CPU instruction
timing — useful for demos and testing, but NOT suitable for
consciousness-research experiments (timing noise is deterministic
under classical physics).

The algorithm:
  1. For each byte requested, perform 8 timing measurements.
  2. Each measurement times a tight loop of SHA-256 hash operations.
  3. Extract the least-significant bit of the nanosecond delta.
  4. Combine 8 bits into one byte.

This is significantly slower than os.urandom() but demonstrates how
to plug ANY physical process into qr-sampler's gRPC interface.

Usage:
    python timing_noise_server.py
    python timing_noise_server.py --port 50052
    python timing_noise_server.py --hash-iterations 128  # More jitter
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import signal
import sys
import time
from concurrent import futures

import grpc

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from qr_sampler.proto.entropy_service_pb2 import EntropyResponse  # noqa: E402
from qr_sampler.proto.entropy_service_pb2_grpc import (  # noqa: E402
    EntropyServiceServicer,
    add_EntropyServiceServicer_to_server,
)

logger = logging.getLogger("timing_entropy_server")

_DEVICE_ID = "timing-noise-server-v1"
_BITS_PER_BYTE = 8


def _generate_timing_bytes(n: int, hash_iterations: int) -> bytes:
    """Generate n bytes from CPU timing jitter.

    For each byte, this performs 8 timing measurements of tight hash
    loops and extracts the LSB of each nanosecond delta. The result
    depends on CPU scheduling, cache state, and thermal throttling.

    Args:
        n: Number of random bytes to generate.
        hash_iterations: Iterations per timing measurement. More iterations
            means more timing variance but also more latency.

    Returns:
        Exactly n bytes of timing-derived entropy.
    """
    result = bytearray(n)
    for i in range(n):
        byte_val = 0
        for bit in range(_BITS_PER_BYTE):
            t0 = time.perf_counter_ns()
            # Tight computation loop. Timing varies with CPU state,
            # cache pressure, instruction pipeline, etc.
            h = hashlib.sha256(b"timing_noise")
            for _ in range(hash_iterations):
                h = hashlib.sha256(h.digest())
            t1 = time.perf_counter_ns()
            delta = t1 - t0
            byte_val |= (delta & 1) << bit
        result[i] = byte_val
    return bytes(result)


class TimingNoiseEntropyServicer(EntropyServiceServicer):
    """Serves random bytes derived from CPU timing jitter.

    Warning:
        This is a DEMO / educational source. Timing noise is
        deterministic in principle (no quantum collapse involved).
        Use UrandomEntropyServicer or a real QRNG for research.

    Args:
        hash_iterations: Number of SHA-256 iterations per timing measurement.
    """

    def __init__(self, hash_iterations: int = 64) -> None:
        self._hash_iterations = hash_iterations

    def GetEntropy(self, request, context):  # noqa: N802
        """Handle a single entropy request (unary RPC)."""
        n = request.bytes_needed
        if n <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"bytes_needed must be > 0, got {n}")
            return EntropyResponse()

        t0 = time.perf_counter_ns()
        data = _generate_timing_bytes(n, self._hash_iterations)
        generation_time = time.time_ns()
        elapsed_us = (time.perf_counter_ns() - t0) / 1000.0

        logger.debug(
            "Unary: seq=%d, bytes=%d, elapsed=%.1f us",
            request.sequence_id,
            n,
            elapsed_us,
        )
        return EntropyResponse(
            data=data,
            sequence_id=request.sequence_id,
            generation_timestamp_ns=generation_time,
            device_id=_DEVICE_ID,
        )

    def StreamEntropy(self, request_iterator, context):  # noqa: N802
        """Handle a bidirectional entropy stream."""
        for request in request_iterator:
            n = request.bytes_needed
            if n <= 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"bytes_needed must be > 0, got {n}")
                return

            data = _generate_timing_bytes(n, self._hash_iterations)
            generation_time = time.time_ns()

            logger.debug("Stream: seq=%d, bytes=%d", request.sequence_id, n)
            yield EntropyResponse(
                data=data,
                sequence_id=request.sequence_id,
                generation_timestamp_ns=generation_time,
                device_id=_DEVICE_ID,
            )


def serve(address: str, max_workers: int, hash_iterations: int) -> None:
    """Start the gRPC timing-noise entropy server.

    Args:
        address: Bind address (e.g. ``localhost:50051``).
        max_workers: Thread pool size.
        hash_iterations: SHA-256 iterations per timing measurement.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = TimingNoiseEntropyServicer(hash_iterations=hash_iterations)
    add_EntropyServiceServicer_to_server(servicer, server)

    server.add_insecure_port(address)
    server.start()

    logger.info("Timing-noise entropy server listening on %s", address)
    logger.info("Device ID: %s", _DEVICE_ID)
    logger.info("Hash iterations per measurement: %d", hash_iterations)
    logger.warning("This source is EXPERIMENTAL. Timing jitter is not quantum-random!")
    logger.info("Press Ctrl+C to stop")

    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        server.stop(grace=5)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


def main() -> None:
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="gRPC entropy server using CPU timing jitter (educational)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Note: This source is EXPERIMENTAL and educational. Timing noise is NOT
quantum-random. Use simple_urandom_server.py or a real QRNG for research.

Examples:
  %(prog)s                              # Default: 0.0.0.0:50051
  %(prog)s --port 50052                 # Custom port
  %(prog)s --hash-iterations 128        # More jitter (slower)
""",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen on (default: 50051).",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Full bind address (overrides --port).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size (default: 4).",
    )
    parser.add_argument(
        "--hash-iterations",
        type=int,
        default=64,
        help="SHA-256 iterations per timing measurement (default: 64).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    address = args.address or f"0.0.0.0:{args.port}"
    serve(address, args.max_workers, args.hash_iterations)


if __name__ == "__main__":
    main()
