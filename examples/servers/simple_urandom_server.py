#!/usr/bin/env python3
"""Minimal gRPC entropy server using os.urandom().

This is the fastest way to get a working entropy server for qr-sampler.
It serves random bytes from the operating system's cryptographic RNG
and supports all three transport modes (unary, server-streaming, and
bidirectional streaming).

Usage:
    # Start on the default port (50051):
    python simple_urandom_server.py

    # Start on a custom port:
    python simple_urandom_server.py --port 50052

    # Listen on a Unix domain socket (Linux/macOS):
    python simple_urandom_server.py --address unix:///var/run/qrng.sock

    # Enable reflection for debugging with grpcurl:
    python simple_urandom_server.py --reflection

Then configure qr-sampler:
    export QR_GRPC_SERVER_ADDRESS=localhost:50051
    export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
    export QR_GRPC_MODE=unary       # or bidi_streaming for lowest latency
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from concurrent import futures

import grpc

# ---------------------------------------------------------------------------
# The proto stubs live inside the qr_sampler package. We add the src/
# directory to the path so this script works both when qr-sampler is
# installed and when running directly from the repository.
# ---------------------------------------------------------------------------
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

logger = logging.getLogger("qr_entropy_server")

# A device identifier for this server — included in every response so
# clients can distinguish entropy sources in diagnostic logs.
_DEVICE_ID = "urandom-server-v1"


class UrandomEntropyServicer(EntropyServiceServicer):
    """Serves random bytes from os.urandom().

    Implements both unary and bidirectional streaming RPCs. Every call
    generates entropy on the spot (just-in-time) — no pre-buffering.
    """

    def GetEntropy(self, request, context):  # noqa: N802
        """Handle a single entropy request (unary RPC).

        Args:
            request: EntropyRequest with bytes_needed and sequence_id.
            context: gRPC service context.

        Returns:
            EntropyResponse with fresh random bytes.
        """
        n = request.bytes_needed
        if n <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"bytes_needed must be > 0, got {n}")
            return EntropyResponse()

        # Just-in-time: entropy is generated NOW, not before.
        data = os.urandom(n)
        timestamp = time.time_ns()

        logger.debug(
            "Unary: seq=%d, bytes=%d, latency_ns=%d",
            request.sequence_id,
            n,
            time.time_ns() - timestamp,
        )
        return EntropyResponse(
            data=data,
            sequence_id=request.sequence_id,
            generation_timestamp_ns=timestamp,
            device_id=_DEVICE_ID,
        )

    def StreamEntropy(self, request_iterator, context):  # noqa: N802
        """Handle a bidirectional entropy stream.

        For each incoming request, generates fresh entropy and sends it
        back immediately. The stream stays open for the entire inference
        session, amortizing connection setup costs.

        Args:
            request_iterator: Iterator of EntropyRequest messages.
            context: gRPC service context.

        Yields:
            EntropyResponse for each incoming request.
        """
        for request in request_iterator:
            n = request.bytes_needed
            if n <= 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(f"bytes_needed must be > 0, got {n}")
                return

            data = os.urandom(n)
            timestamp = time.time_ns()

            logger.debug(
                "Stream: seq=%d, bytes=%d",
                request.sequence_id,
                n,
            )
            yield EntropyResponse(
                data=data,
                sequence_id=request.sequence_id,
                generation_timestamp_ns=timestamp,
                device_id=_DEVICE_ID,
            )


def serve(address: str, max_workers: int, reflection: bool) -> None:
    """Start the gRPC server and block until terminated.

    Args:
        address: Bind address (e.g. ``localhost:50051`` or ``unix:///path``).
        max_workers: Thread pool size for handling concurrent requests.
        reflection: Whether to enable gRPC server reflection.
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_EntropyServiceServicer_to_server(UrandomEntropyServicer(), server)

    # Enable server reflection if requested (useful for grpcurl debugging).
    if reflection:
        try:
            from grpc_reflection.v1alpha import reflection as grpc_reflection

            service_names = ("qr_entropy.EntropyService",)
            grpc_reflection.enable_server_reflection(service_names, server)
            logger.info("gRPC reflection enabled")
        except ImportError:
            logger.warning(
                "grpc-reflection not installed. Install with: pip install grpcio-reflection"
            )

    server.add_insecure_port(address)
    server.start()
    logger.info("Entropy server listening on %s", address)
    logger.info("Device ID: %s", _DEVICE_ID)
    logger.info("Press Ctrl+C to stop")

    # Graceful shutdown on SIGINT (SIGTERM not available on Windows).
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
        description="Minimal gRPC entropy server using os.urandom()",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                              # Default: 0.0.0.0:50051
  %(prog)s --port 50052                 # Custom port
  %(prog)s --address unix:///tmp/qrng.sock  # Unix socket
  %(prog)s --reflection                 # Enable grpcurl debugging
""",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to listen on (default: 50051). Ignored if --address is set.",
    )
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help="Full bind address (e.g. 'localhost:50051' or 'unix:///path').",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size (default: 4).",
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Enable gRPC server reflection for debugging.",
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
    serve(address, args.max_workers, args.reflection)


if __name__ == "__main__":
    main()
