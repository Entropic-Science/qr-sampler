#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install-dfqrng-dkms.sh — build/register the Dragonfly QRNG kernel driver
# ─────────────────────────────────────────────────────────────────────────────
# Run as root on the qr-server box:
#
#     sudo bash install-dfqrng-dkms.sh
#
# Why this exists (incident 2026-08-06): the box rebooted into a new kernel
# (7.0.0-28-generic) for which the out-of-tree dfqrng module had never been
# built. /dev/qrngDF* never appeared, qbert0g started device-less and kept
# serving, and every quantum draw across every app silently fell back to
# system PRNG for a week — surfaced only by OWUI's regenerate banner.
#
# This script makes that failure mode structurally impossible to repeat
# silently:
#   1. registers the driver with DKMS  → every future kernel upgrade rebuilds
#      it automatically (falls back to a one-off build for the running kernel
#      when DKMS cannot be installed);
#   2. loads it at boot via /etc/modules-load.d/dfqrng.conf;
#   3. installs a systemd drop-in so qbert0g REFUSES to start until both
#      /dev/qrngDF* chardevs exist, cycling loudly instead of limping
#      (mirrors the guard now in the repo's qbert0g.service);
#   4. restarts qbert0g and verifies the devices actually initialise.
#
# Idempotent: safe to re-run after editing the driver source or on a new
# kernel.
set -euo pipefail

DRIVER_SRC="${DRIVER_SRC:-/home/alchy/src/dragonfly/driver}"
DKMS_NAME=dfqrng
DKMS_VERSION="${DKMS_VERSION:-1.0}"
DKMS_SRC="/usr/src/${DKMS_NAME}-${DKMS_VERSION}"
KVER="$(uname -r)"

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)" >&2; exit 1; }
[ -d "$DRIVER_SRC" ] || { echo "driver source not found: $DRIVER_SRC" >&2; exit 1; }
[ -d "/lib/modules/${KVER}/build" ] || {
    echo "kernel headers missing for ${KVER}: apt-get install linux-headers-${KVER}" >&2
    exit 1
}

# ── 1. DKMS registration (preferred) or one-off build (fallback) ─────────────
if ! command -v dkms >/dev/null 2>&1; then
    echo "dkms not installed; attempting apt-get install..."
    apt-get install -y dkms || true
fi

if command -v dkms >/dev/null 2>&1; then
    echo "== registering ${DKMS_NAME}/${DKMS_VERSION} with DKMS =="
    # Fresh copy of the source (sans build artifacts) into /usr/src.
    rm -rf "$DKMS_SRC"
    mkdir -p "$DKMS_SRC"
    (cd "$DRIVER_SRC" && make clean >/dev/null 2>&1 || true)
    cp -a "$DRIVER_SRC/." "$DKMS_SRC/"
    cat > "$DKMS_SRC/dkms.conf" <<EOF
PACKAGE_NAME="${DKMS_NAME}"
PACKAGE_VERSION="${DKMS_VERSION}"
MAKE[0]="make -C \${kernel_source_dir} M=\${dkms_tree}/${DKMS_NAME}/${DKMS_VERSION}/build CONFIG_DFQRNG=m modules"
CLEAN="make -C \${kernel_source_dir} M=\${dkms_tree}/${DKMS_NAME}/${DKMS_VERSION}/build clean"
BUILT_MODULE_NAME[0]="${DKMS_NAME}"
DEST_MODULE_LOCATION[0]="/extra"
AUTOINSTALL="yes"
EOF
    # Re-registration must survive a prior add of the same version.
    dkms remove -m "$DKMS_NAME" -v "$DKMS_VERSION" --all >/dev/null 2>&1 || true
    dkms add    -m "$DKMS_NAME" -v "$DKMS_VERSION"
    dkms build  -m "$DKMS_NAME" -v "$DKMS_VERSION" -k "$KVER"
    dkms install -m "$DKMS_NAME" -v "$DKMS_VERSION" -k "$KVER" --force
else
    echo "== DKMS unavailable — one-off build for ${KVER} (WILL NOT survive the next kernel upgrade) =="
    (cd "$DRIVER_SRC" && make clean >/dev/null 2>&1 || true; make)
    install -D -m 644 "$DRIVER_SRC/${DKMS_NAME}.ko" \
        "/lib/modules/${KVER}/extra/${DKMS_NAME}.ko"
    depmod -a
fi

# ── 2. Load now + at every boot ──────────────────────────────────────────────
echo "$DKMS_NAME" > /etc/modules-load.d/dfqrng.conf
modprobe "$DKMS_NAME"

echo "== waiting for /dev/qrngDF* =="
for _ in $(seq 1 15); do
    [ -e /dev/qrngDF0 ] && [ -e /dev/qrngDF1 ] && break
    sleep 1
done
ls -l /dev/qrngDF0 /dev/qrngDF1

# ── 3. systemd device guard for qbert0g ──────────────────────────────────────
# Drop-in equivalent of the guard in this profile's qbert0g.service, so boxes
# still running the pre-guard unit get the protection without a unit swap.
mkdir -p /etc/systemd/system/qbert0g.service.d
cat > /etc/systemd/system/qbert0g.service.d/10-device-guard.conf <<'EOF'
# Installed by install-dfqrng-dkms.sh (2026-08-06 incident). qbert0g keeps
# serving even when all devices failed to open at startup; this guard makes a
# missing-driver boot a loudly cycling unit instead of a silent PRNG fallback.
[Unit]
StartLimitIntervalSec=0

[Service]
ExecStartPre=/bin/sh -c 'for i in $(seq 1 30); do [ -e /dev/qrngDF0 ] && [ -e /dev/qrngDF1 ] && exit 0; sleep 1; done; echo "qbert0g: /dev/qrngDF* missing after 30s — is the dfqrng module built+loaded for $(uname -r)? (dkms status dfqrng)" >&2; exit 1'
EOF
systemctl daemon-reload

# ── 4. Restart the daemon (device init happens only at startup) + verify ─────
echo "== restarting qbert0g =="
systemctl restart qbert0g
sleep 5
journalctl -u qbert0g --since "-30s" --no-pager | tail -n 20

if journalctl -u qbert0g --since "-30s" --no-pager | grep -q "Failed to initialize device"; then
    echo "!! device init still failing — see journal above" >&2
    exit 1
fi

echo "== OK: devices up. The shared vLLM's circuit breaker re-engages the QRNG"
echo "   on its next half-open (<=60s); /health/entropy rpc_ok should flip true."
echo "   Verify: curl -s http://127.0.0.1:8000/health/entropy"
