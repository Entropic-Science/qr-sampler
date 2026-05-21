"""Cloudflared `access tcp` sidecar manager for QRNG reachability.

The QRNG gRPC service (`qbert-grpc.cipherstone.co`) is published behind a
Cloudflare Zero Trust Access policy and a `cloudflared` named tunnel.
A Modal worker cannot reach the origin directly — it needs to dial through
Cloudflare's edge using an Access service token. The pattern, lifted from
the reference Modal client shipped with the QRNG team, is to run
`cloudflared access tcp` as a per-container sidecar process: it opens a
local TCP listener on the loopback interface and forwards every byte
through Cloudflare's edge to the tunnel's origin.

Why a sidecar rather than embedding a TLS / mTLS gRPC client in the
sampler? Three reasons:

1. **Auth isolation.** Service-token auth (CF Access) lives entirely inside
   `cloudflared`. The gRPC client just dials loopback, so the sampler's
   `QuantumGrpcSource` stays protocol-agnostic and credential-free.
2. **Connection reuse.** `cloudflared` keeps one HTTPS connection to the
   Cloudflare edge open across many gRPC RPCs — there is no per-call
   tunnel setup overhead. Critical on the hot path (one entropy fetch
   per generated token).
3. **No TUN / NET_ADMIN.** `cloudflared access tcp` is a userspace TCP
   forwarder — no kernel capabilities, no virtual interface, nothing that
   a Modal container can't run.

Snapshot safety
---------------
The subprocess MUST NOT be alive at snapshot time. Modal freezes file
descriptors into the snapshot image, and a live TCP listener would
reopen as undefined behaviour on restore. Therefore:

* :meth:`CloudflaredSidecar.start` is intended to be called from
  ``@modal.enter(snap=False)`` (post-restore), NEVER from
  ``@modal.enter(snap=True)``.
* :meth:`CloudflaredSidecar.stop` is idempotent and exit-safe; call from
  ``@modal.exit()``.

Failure modes the operator should know about
--------------------------------------------
* Missing CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET / QRNG_TUNNEL_HOSTNAME
  → :class:`CloudflaredConfigError` on :meth:`start`. The container will
  fail to enter, with a clear stderr message naming the missing var.
* `cloudflared` binary not installed in the image → :class:`FileNotFoundError`
  propagates from :func:`subprocess.Popen`. Fix: ensure Dockerfile installs
  the package (see `Dockerfile.vllm`).
* Tunnel did not become ready within the timeout → :class:`CloudflaredStartupError`.
  This usually means CF Access denied the service token (revoked / wrong
  team / wrong app), or the tunnel hostname is wrong. The exception body
  includes the last ~20 lines of cloudflared stderr so the operator can
  diagnose without re-deploying.
"""

from __future__ import annotations

import collections
import logging
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("qr_sampler.cloudflared")

# Default loopback bind. The gRPC client must use the same address (set via
# QR_GRPC_SERVER_ADDRESS).
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 50051

# Conservative default: 15s is well above the artifact's measured ~1.5s
# cloudflared startup but stays under Modal's @modal.enter timeout budget.
DEFAULT_STARTUP_TIMEOUT_S = 15.0

# Number of stderr lines we retain in-process and surface on failure. Bounded
# so a chatty cloudflared build cannot blow our memory.
_STDERR_TAIL_LINES = 20


class CloudflaredConfigError(RuntimeError):
    """Required Cloudflare Access env var is missing or empty."""


class CloudflaredStartupError(RuntimeError):
    """cloudflared exited or did not become ready within the timeout."""


@dataclass(frozen=True)
class CloudflaredConfig:
    """Validated env-derived configuration for the sidecar.

    Use :meth:`from_env` rather than constructing directly so missing vars
    surface as a single :class:`CloudflaredConfigError` with the operator-
    facing message template.
    """

    hostname: str
    service_token_id: str
    service_token_secret: str
    bind_host: str = DEFAULT_BIND_HOST
    bind_port: int = DEFAULT_BIND_PORT
    startup_timeout_s: float = DEFAULT_STARTUP_TIMEOUT_S

    @classmethod
    def from_env(cls) -> CloudflaredConfig:
        """Build a config from QRNG_TUNNEL_HOSTNAME / CF_ACCESS_CLIENT_*.

        Reads:
            QRNG_TUNNEL_HOSTNAME      — tunnel hostname (e.g. qbert-grpc.cipherstone.co)
            CF_ACCESS_CLIENT_ID       — Access service-token client id
            CF_ACCESS_CLIENT_SECRET   — Access service-token client secret
            QRNG_TUNNEL_BIND_HOST     — optional, defaults to 127.0.0.1
            QRNG_TUNNEL_BIND_PORT     — optional, defaults to 50051
            QRNG_TUNNEL_STARTUP_TIMEOUT_S — optional, defaults to 15.0

        Raises:
            CloudflaredConfigError: when any required var is unset or empty.
        """
        missing: list[str] = []
        hostname = os.environ.get("QRNG_TUNNEL_HOSTNAME", "").strip()
        if not hostname:
            missing.append("QRNG_TUNNEL_HOSTNAME")
        token_id = os.environ.get("CF_ACCESS_CLIENT_ID", "").strip()
        if not token_id:
            missing.append("CF_ACCESS_CLIENT_ID")
        token_secret = os.environ.get("CF_ACCESS_CLIENT_SECRET", "").strip()
        if not token_secret:
            missing.append("CF_ACCESS_CLIENT_SECRET")
        if missing:
            raise CloudflaredConfigError(
                "cloudflared sidecar cannot start; the following env vars "
                f"are unset or empty: {', '.join(missing)}. Populate them "
                "in the qr-sampler-prod Modal Secret (see "
                "src/qr_sampler/connectors/modal/modal_secrets.md, section "
                "'QRNG via Cloudflare Access')."
            )

        bind_host = os.environ.get("QRNG_TUNNEL_BIND_HOST", DEFAULT_BIND_HOST).strip()
        bind_port_raw = os.environ.get("QRNG_TUNNEL_BIND_PORT", "").strip()
        bind_port = int(bind_port_raw) if bind_port_raw else DEFAULT_BIND_PORT
        timeout_raw = os.environ.get("QRNG_TUNNEL_STARTUP_TIMEOUT_S", "").strip()
        timeout_s = float(timeout_raw) if timeout_raw else DEFAULT_STARTUP_TIMEOUT_S

        return cls(
            hostname=hostname,
            service_token_id=token_id,
            service_token_secret=token_secret,
            bind_host=bind_host,
            bind_port=bind_port,
            startup_timeout_s=timeout_s,
        )


class CloudflaredSidecar:
    """Manages a single ``cloudflared access tcp`` subprocess.

    Lifecycle:
        sidecar = CloudflaredSidecar(CloudflaredConfig.from_env())
        sidecar.start()              # in @modal.enter(snap=False)
        # ... container serves requests; gRPC client dials 127.0.0.1:50051
        sidecar.stop()               # in @modal.exit()
    """

    def __init__(self, config: CloudflaredConfig) -> None:
        self._config = config
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_thread: threading.Thread | None = None

    @property
    def bind_address(self) -> str:
        """`host:port` the sidecar will listen on. Pass to gRPC client."""
        return f"{self._config.bind_host}:{self._config.bind_port}"

    def start(self) -> None:
        """Spawn cloudflared and block until its listener accepts connections.

        Raises:
            CloudflaredStartupError: when the listener does not come up
                within the configured timeout, or cloudflared exits early.
            FileNotFoundError: when the ``cloudflared`` binary is not on PATH
                (image misconfiguration).
        """
        if self._proc is not None:
            logger.warning(
                "cloudflared.start called twice; ignoring (pid=%s)",
                self._proc.pid,
            )
            return

        logger.info(
            "cloudflared.start: spawning sidecar (hostname=%s, bind=%s)",
            self._config.hostname,
            self.bind_address,
            extra={
                "event": "cloudflared.start",
                "hostname": self._config.hostname,
                "bind_address": self.bind_address,
            },
        )

        # We pass the service token via env vars (CF_ACCESS_CLIENT_ID /
        # CF_ACCESS_CLIENT_SECRET) so they never appear on the command line
        # or in `ps`. cloudflared reads these automatically when no
        # `--service-token-*` flags are present.
        env = os.environ.copy()
        env["TUNNEL_SERVICE_TOKEN_ID"] = self._config.service_token_id
        env["TUNNEL_SERVICE_TOKEN_SECRET"] = self._config.service_token_secret

        try:
            self._proc = subprocess.Popen(
                [
                    "cloudflared",
                    "access",
                    "tcp",
                    "--hostname",
                    self._config.hostname,
                    "--url",
                    self.bind_address,
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError:
            logger.error(
                "cloudflared.start: binary not on PATH. Install it in the "
                "container image (see Dockerfile.vllm).",
                extra={"event": "cloudflared.binary_missing"},
            )
            raise

        # Drain stderr in a background thread so it never blocks the
        # subprocess and so we can surface the tail on failure.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="cloudflared-stderr-drain",
            daemon=True,
        )
        self._stderr_thread.start()

        try:
            self._wait_for_listener()
        except CloudflaredStartupError:
            # Cleanup the subprocess before re-raising so the caller is not
            # responsible for the partially-started state.
            self._terminate_quietly()
            raise

        logger.info(
            "cloudflared.ready: listener up (pid=%s, bind=%s, hostname=%s)",
            self._proc.pid,
            self.bind_address,
            self._config.hostname,
            extra={
                "event": "cloudflared.ready",
                "pid": self._proc.pid,
                "bind_address": self.bind_address,
                "hostname": self._config.hostname,
            },
        )

    def stop(self) -> None:
        """Tear down the cloudflared subprocess. Idempotent."""
        proc = self._proc
        if proc is None:
            return
        self._proc = None

        if proc.poll() is not None:
            logger.info(
                "cloudflared.stop: already exited (returncode=%s)",
                proc.returncode,
                extra={
                    "event": "cloudflared.stop",
                    "returncode": proc.returncode,
                    "stage": "already_exited",
                },
            )
            return

        logger.info(
            "cloudflared.stop: terminating sidecar (pid=%s)",
            proc.pid,
            extra={"event": "cloudflared.stop", "pid": proc.pid, "stage": "terminate"},
        )
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.warning(
                "cloudflared.stop: SIGTERM did not finish in 5s; killing (pid=%s)",
                proc.pid,
                extra={"event": "cloudflared.stop", "pid": proc.pid, "stage": "kill"},
            )
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.error(
                    "cloudflared.stop: process refused to die after SIGKILL (pid=%s); leaking",
                    proc.pid,
                )

    def _wait_for_listener(self) -> None:
        """Block until cloudflared accepts TCP connections on its bind addr."""
        assert self._proc is not None
        deadline = time.monotonic() + self._config.startup_timeout_s
        last_exc: OSError | None = None
        while time.monotonic() < deadline:
            # If cloudflared exited early, no listener will ever come up.
            # Surface that immediately rather than waiting out the timeout.
            if self._proc.poll() is not None:
                raise CloudflaredStartupError(
                    self._exit_diagnostic(prefix="cloudflared exited before its listener came up")
                )
            try:
                with socket.create_connection(
                    (self._config.bind_host, self._config.bind_port),
                    timeout=0.2,
                ):
                    return
            except OSError as exc:
                last_exc = exc
                time.sleep(0.1)

        raise CloudflaredStartupError(
            self._exit_diagnostic(
                prefix=(
                    f"cloudflared did not accept connections on "
                    f"{self.bind_address} within "
                    f"{self._config.startup_timeout_s:.1f}s"
                ),
                tail_exc=last_exc,
            )
        )

    def _drain_stderr(self) -> None:
        """Background reader: tail cloudflared stderr into the deque."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                stripped = line.rstrip("\n")
                self._stderr_tail.append(stripped)
                # cloudflared logs in JSON when `--log-format json` is set,
                # which we deliberately do NOT do — its default text format
                # is more grep-friendly inside Modal logs.
                logger.debug("cloudflared.stderr: %s", stripped)
        except (ValueError, OSError):
            # Stream closed during shutdown — expected.
            return

    def _terminate_quietly(self) -> None:
        """Best-effort cleanup used from inside start() on failure paths."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _exit_diagnostic(self, *, prefix: str, tail_exc: OSError | None = None) -> str:
        """Build the operator-facing error message for startup failures."""
        proc = self._proc
        rc = proc.returncode if proc is not None else "<unknown>"
        tail = "\n".join(self._stderr_tail) or "<no stderr captured>"
        hint = (
            "Check: (1) CF_ACCESS_CLIENT_ID / CF_ACCESS_CLIENT_SECRET match a "
            "Service Token whose policy allows the tunnel app; (2) "
            f"QRNG_TUNNEL_HOSTNAME={self._config.hostname!r} is the hostname "
            "of the Cloudflare Access TCP application; (3) the cloudflared "
            "binary in the container image is recent enough to support "
            "`access tcp` (any 2024+ release works)."
        )
        last_sock = f" (last socket error: {tail_exc!r})" if tail_exc else ""
        return (
            f"{prefix}{last_sock}. cloudflared returncode={rc}. {hint}\n"
            f"--- cloudflared stderr (last {_STDERR_TAIL_LINES} lines) ---\n"
            f"{tail}\n--- end stderr ---"
        )
