#!/usr/bin/env python3
"""Annotated template for building your own QRNG entropy server.

This file is a starting point for connecting ANY quantum random number
generator (or other hardware entropy source) to qr-sampler. Copy this
file, implement the three TODO sections, and you have a production-ready
entropy server.

The template includes:
  - All three gRPC transport modes (unary, server-streaming, bidi)
  - Health checking and diagnostics
  - Graceful shutdown
  - Connection logging
  - Error handling patterns

=====================================================================
  HOW TO USE THIS TEMPLATE
=====================================================================

  1. Copy this file to your own project:
     cp qrng_template_server.py my_qrng_server.py

  2. Search for "TODO:" — there are 3 sections to implement:
     - TODO 1: Initialize your hardware connection
     - TODO 2: Generate entropy from your hardware
     - TODO 3: Clean up your hardware connection

  3. Install dependencies:
     pip install grpcio qr-sampler

  4. Run your server:
     python my_qrng_server.py --port 50051

  5. Configure qr-sampler to use it:
     export QR_GRPC_SERVER_ADDRESS=localhost:50051
     export QR_ENTROPY_SOURCE_TYPE=quantum_grpc

=====================================================================

Usage:
    python qrng_template_server.py --port 50051
    python qrng_template_server.py --address unix:///var/run/qrng.sock
    python qrng_template_server.py --verbose --port 50052
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

logger = logging.getLogger("qrng_server")


# ===================================================================
# TODO 1: HARDWARE INITIALIZATION
# ===================================================================
#
# Replace this class with your QRNG hardware interface.
# Examples of real hardware you might connect:
#
#   - ID Quantique Quantis (USB/PCIe QRNG)
#   - Quside FMC400 (photonic QRNG)
#   - ComScire PQ32MU (shot-noise QRNG)
#   - Custom photon detector setup
#   - Any device that produces random bits from quantum processes
#
# Your hardware class should provide:
#   - __init__(): Open the connection / device handle
#   - generate(n_bytes): Return exactly n_bytes of fresh entropy
#   - close(): Release the device handle
#   - is_healthy(): Return True if the device is operational


class QRNGHardware:
    """Interface to your QRNG hardware.

    TODO: Replace this with your actual hardware driver.

    This placeholder uses os.urandom() so the template runs out of the
    box. Replace the body of each method with your hardware calls.
    """

    def __init__(self, device_path: str = "/dev/qrng0") -> None:
        """Open a connection to the QRNG hardware.

        TODO: Replace with your hardware initialization code.
        Examples:
            self._device = open(device_path, "rb")
            self._usb_handle = usb.core.find(idVendor=0x1234)
            self._serial = serial.Serial(device_path, baudrate=115200)

        Args:
            device_path: Path to the hardware device or connection string.
        """
        self._device_path = device_path
        self._device_id = "template-placeholder-v1"
        logger.info("Hardware init: %s (using os.urandom placeholder)", device_path)

        # TODO: YOUR HARDWARE INIT CODE HERE
        # Example for a USB QRNG:
        #   import usb.core
        #   self._device = usb.core.find(idVendor=0x1234, idProduct=0x5678)
        #   if self._device is None:
        #       raise RuntimeError("QRNG device not found")
        #   self._device.set_configuration()

    @property
    def device_id(self) -> str:
        """Human-readable device identifier for diagnostics."""
        return self._device_id

    def generate(self, n_bytes: int) -> bytes:
        """Generate exactly n_bytes of quantum-random entropy.

        CRITICAL: This method must generate entropy JUST-IN-TIME.
        Do NOT read from a pre-buffered pool. The physical quantum
        measurement must happen during this call.

        TODO: Replace with your hardware read code.
        Examples:
            return self._device.read(n_bytes)
            return self._usb_handle.read(0x81, n_bytes)
            return self._serial.read(n_bytes)

        Args:
            n_bytes: Number of random bytes to generate.

        Returns:
            Exactly n_bytes of fresh quantum-random entropy.

        Raises:
            RuntimeError: If the hardware cannot generate bytes.
        """
        # TODO: YOUR HARDWARE ENTROPY GENERATION CODE HERE
        #
        # IMPORTANT: The just-in-time constraint means:
        #   - No pre-generated buffer pools
        #   - No caching of previously generated bytes
        #   - The quantum measurement happens NOW, during this call
        #
        # Example for reading from a device file:
        #   data = self._device.read(n_bytes)
        #   if len(data) != n_bytes:
        #       raise RuntimeError(f"Short read: got {len(data)}, wanted {n_bytes}")
        #   return data

        # Placeholder: os.urandom() so the template works without hardware
        return os.urandom(n_bytes)

    def is_healthy(self) -> bool:
        """Check if the QRNG hardware is operational.

        TODO: Replace with your hardware health check.
        Examples:
            - Try a small read and verify it succeeds
            - Check device status registers
            - Verify laser diode temperature is in range

        Returns:
            True if the device is operational and ready.
        """
        # TODO: YOUR HARDWARE HEALTH CHECK CODE HERE
        # Example:
        #   try:
        #       test_bytes = self._device.read(1)
        #       return len(test_bytes) == 1
        #   except Exception:
        #       return False

        return True  # Placeholder

    def close(self) -> None:
        """Release the QRNG hardware connection.

        TODO: Replace with your hardware cleanup code.
        Examples:
            self._device.close()
            self._usb_handle.reset()
            self._serial.close()
        """
        # TODO: YOUR HARDWARE CLEANUP CODE HERE
        logger.info("Hardware connection closed")


# ===================================================================
# gRPC SERVICE IMPLEMENTATION (usually no changes needed)
# ===================================================================


class QRNGEntropyServicer(EntropyServiceServicer):
    """gRPC service that delegates to QRNG hardware.

    This servicer handles all three transport modes. You typically
    don't need to modify this class — just implement the QRNGHardware
    class above.

    Args:
        hardware: An initialized QRNGHardware instance.
    """

    def __init__(self, hardware: QRNGHardware) -> None:
        self._hw = hardware
        self._total_bytes_served = 0
        self._total_requests = 0

    def GetEntropy(self, request, context):  # noqa: N802
        """Handle a single entropy request (unary RPC).

        This is the simplest transport mode. Each call creates a new
        HTTP/2 stream, which adds ~1-2ms of overhead. Use this for
        low-frequency sampling or when simplicity matters more than
        latency.

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

        try:
            # === JUST-IN-TIME GENERATION ===
            # The quantum measurement happens in this call, not before.
            data = self._hw.generate(n)
            timestamp = time.time_ns()
        except Exception as exc:
            logger.error("Hardware entropy generation failed: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Entropy generation failed: {exc}")
            return EntropyResponse()

        self._total_bytes_served += n
        self._total_requests += 1

        logger.debug(
            "Unary: seq=%d, bytes=%d, total_served=%d",
            request.sequence_id,
            n,
            self._total_bytes_served,
        )
        return EntropyResponse(
            data=data,
            sequence_id=request.sequence_id,
            generation_timestamp_ns=timestamp,
            device_id=self._hw.device_id,
        )

    def StreamEntropy(self, request_iterator, context):  # noqa: N802
        """Handle a bidirectional entropy stream.

        This is the LOWEST LATENCY mode. The gRPC stream stays open
        for the entire inference session. Each request generates fresh
        entropy on demand.

        For server-streaming mode, qr-sampler sends one request and
        reads one response per call. For true bidi mode, it keeps the
        stream open and sends/reads repeatedly.

        Both modes use this same RPC — the difference is in how the
        client manages the stream lifecycle.

        Args:
            request_iterator: Iterator of EntropyRequest messages.
            context: gRPC service context.

        Yields:
            EntropyResponse for each incoming request.
        """
        peer = context.peer()
        logger.info("Stream opened from %s", peer)

        request_count = 0
        try:
            for request in request_iterator:
                n = request.bytes_needed
                if n <= 0:
                    context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                    context.set_details(f"bytes_needed must be > 0, got {n}")
                    return

                try:
                    data = self._hw.generate(n)
                    timestamp = time.time_ns()
                except Exception as exc:
                    logger.error("Hardware entropy generation failed: %s", exc)
                    context.set_code(grpc.StatusCode.INTERNAL)
                    context.set_details(f"Entropy generation failed: {exc}")
                    return

                self._total_bytes_served += n
                self._total_requests += 1
                request_count += 1

                yield EntropyResponse(
                    data=data,
                    sequence_id=request.sequence_id,
                    generation_timestamp_ns=timestamp,
                    device_id=self._hw.device_id,
                )
        finally:
            logger.info("Stream closed from %s after %d requests", peer, request_count)

    @property
    def stats(self) -> dict:
        """Return server statistics for monitoring."""
        return {
            "total_bytes_served": self._total_bytes_served,
            "total_requests": self._total_requests,
            "hardware_healthy": self._hw.is_healthy(),
            "device_id": self._hw.device_id,
        }


# ===================================================================
# SERVER STARTUP (usually no changes needed)
# ===================================================================


def serve(
    address: str,
    max_workers: int,
    device_path: str,
    reflection: bool,
) -> None:
    """Start the gRPC QRNG entropy server.

    Args:
        address: Bind address (e.g. ``localhost:50051`` or ``unix:///path``).
        max_workers: Thread pool size for handling concurrent requests.
        device_path: Path to the QRNG hardware device.
        reflection: Whether to enable gRPC server reflection.
    """
    # Initialize hardware.
    hardware = QRNGHardware(device_path=device_path)
    if not hardware.is_healthy():
        logger.error("QRNG hardware health check failed!")
        sys.exit(1)

    # Build and start gRPC server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    servicer = QRNGEntropyServicer(hardware)
    add_EntropyServiceServicer_to_server(servicer, server)

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

    logger.info("QRNG entropy server listening on %s", address)
    logger.info("Device: %s (%s)", device_path, hardware.device_id)
    logger.info("Transport modes: unary + bidi streaming")
    logger.info("Press Ctrl+C to stop")

    # Graceful shutdown.
    def _shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        hardware.close()
        server.stop(grace=5)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


def main() -> None:
    """Parse arguments and start the server."""
    parser = argparse.ArgumentParser(
        description="QRNG entropy server template — connect your own hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Quick start:
  1. Edit this file — search for "TODO:" (3 sections)
  2. Implement your hardware's init/generate/close methods
  3. Run: python %(prog)s --port 50051
  4. Configure qr-sampler:
       export QR_GRPC_SERVER_ADDRESS=localhost:50051
       export QR_ENTROPY_SOURCE_TYPE=quantum_grpc

Transport modes:
  The server supports all three modes simultaneously:
  - Unary (QR_GRPC_MODE=unary): one request per HTTP/2 stream
  - Server streaming (QR_GRPC_MODE=server_streaming): short-lived streams
  - Bidi streaming (QR_GRPC_MODE=bidi_streaming): persistent stream, lowest latency

Examples:
  %(prog)s                                      # Default port 50051
  %(prog)s --port 50052 --device /dev/qrng0     # Custom device
  %(prog)s --address unix:///var/run/qrng.sock   # Unix socket (lowest latency)
  %(prog)s --verbose --reflection                # Debug mode
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
        "--device",
        type=str,
        default="/dev/qrng0",
        help="Path to QRNG hardware device (default: /dev/qrng0).",
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
        help="Enable gRPC server reflection.",
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
    serve(address, args.max_workers, args.device, args.reflection)


if __name__ == "__main__":
    main()
