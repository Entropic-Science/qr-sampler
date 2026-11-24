"""Entropy conditioning: what the entropy server does to raw bytes before serving.

Two modes, selected per request with ``conditioning_mode``:

- ``raw`` (default) — the server's measured bytes, untouched.
- ``hash_truncate`` — the server cuts its raw read into
  ``conditioning_in_bytes`` blocks, hashes each block on its own, and
  serves only the first ``conditioning_keep_bytes`` of each digest. The
  served bytes are uniform whatever the device's own byte distribution
  is, and they are always much shorter than the raw read they came from.

The transform itself runs on the entropy server (Qbert0G,
``qbert0g.profiles.hash_truncate`` is the normative definition); this
module only carries the selection: validation, the gRPC metadata that
asks for it, and the canonical string the server echoes back.

Two rules are fixed and mirror the server's: at most half of every
digest is kept, and every block holds at least twice the bytes it
yields.

This module is deliberately dependency-free (stdlib only) so the config
model can validate with it and the gRPC source can send with it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Accepted ``conditioning_mode`` values.
CONDITIONING_MODES: frozenset[str] = frozenset({"raw", "hash_truncate"})

#: Accepted ``conditioning_hash`` values -> digest size in bytes.
CONDITIONING_HASHES: dict[str, int] = {
    "sha256": 32,
    "sha512": 64,
    "sha3_256": 32,
    "blake2b": 64,
}

#: gRPC request metadata keys (the entropy server's contract).
MODE_HEADER = "qr-conditioning"
HASH_HEADER = "qr-conditioning-hash"
IN_BYTES_HEADER = "qr-conditioning-in-bytes"
KEEP_BYTES_HEADER = "qr-conditioning-keep-bytes"
#: Initial-metadata key the server answers with: what it applied.
APPLIED_HEADER = "qr-conditioning-applied"


def conditioning_problem(mode: str, hash_name: str, in_bytes: int, keep_bytes: int) -> str | None:
    """Why this selection is invalid, or ``None`` when it is fine.

    ``raw`` ignores the parameters (they keep their defaults on every
    raw request, so they cannot be required to be absent).
    """
    if mode not in CONDITIONING_MODES:
        return f"conditioning_mode must be one of {sorted(CONDITIONING_MODES)}, got {mode!r}"
    if mode == "raw":
        return None
    if hash_name not in CONDITIONING_HASHES:
        return f"conditioning_hash must be one of {sorted(CONDITIONING_HASHES)}, got {hash_name!r}"
    digest_size = CONDITIONING_HASHES[hash_name]
    if not 1 <= keep_bytes <= digest_size // 2:
        return (
            f"conditioning_keep_bytes must be between 1 and {digest_size // 2} for "
            f"{hash_name}, got {keep_bytes}: at least half of every digest must be "
            "discarded, serving more may compromise the integrity of the signal"
        )
    if in_bytes < 2 * keep_bytes:
        return (
            f"conditioning_in_bytes must be at least 2 * conditioning_keep_bytes "
            f"({2 * keep_bytes}), got {in_bytes}"
        )
    return None


@dataclass(frozen=True, slots=True)
class Conditioning:
    """One request's conditioning selection."""

    mode: str = "raw"
    hash: str = ""
    in_bytes: int = 0
    keep_bytes: int = 0

    @property
    def is_raw(self) -> bool:
        return self.mode == "raw"

    def canonical(self) -> str:
        """The string the server echoes, e.g. ``hash_truncate;sha256;128;16``."""
        if self.is_raw:
            return "raw"
        return f"{self.mode};{self.hash};{self.in_bytes};{self.keep_bytes}"

    def metadata(self) -> tuple[tuple[str, str], ...]:
        """gRPC metadata asking the server for this conditioning.

        Empty for ``raw``: a raw request is byte-identical on the wire to
        one from a client that knows nothing about conditioning.
        """
        if self.is_raw:
            return ()
        return (
            (MODE_HEADER, self.mode),
            (HASH_HEADER, self.hash),
            (IN_BYTES_HEADER, str(self.in_bytes)),
            (KEEP_BYTES_HEADER, str(self.keep_bytes)),
        )

    def raw_bytes_needed(self, out_bytes: int) -> int:
        """Raw bytes the server reads to serve *out_bytes* (cost of a fetch)."""
        if self.is_raw:
            return out_bytes
        return -(-out_bytes // self.keep_bytes) * self.in_bytes


RAW = Conditioning()


def conditioning_of(mode: str, hash_name: str, in_bytes: int, keep_bytes: int) -> Conditioning:
    """The selection for a validated config's four fields.

    Parameters are dropped on ``raw`` so every raw selection is the one
    :data:`RAW` value regardless of the (unused) parameter defaults.
    """
    if mode == "raw":
        return RAW
    return Conditioning(mode=mode, hash=hash_name, in_bytes=in_bytes, keep_bytes=keep_bytes)
