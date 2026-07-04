"""Hand-written protobuf message stubs for the purity (server-draw) service.

These are lightweight message classes that mirror the ``purity_service.proto``
definition (package ``qr_purity``) using **standard protobuf wire encoding**
(primitives from :mod:`qr_sampler.proto.wire`). They produce bytes identical
to ``protoc``-generated code, making them compatible with any standard gRPC
server — in particular the Qbert0G ``PurityService``, whose proto file is
byte-identical to the copy committed in this package (enforced by a shared
sha256 pin in both repos' test suites).

Wire format reference (proto3):
- Varint fields: tag = (field_number << 3 | 0), then LEB128-encoded value
- Fixed64 fields (``double``): tag = (field_number << 3 | 1), then 8 bytes LE
- Length-delimited fields: tag = (field_number << 3 | 2), then varint length,
  then raw bytes
- Default-valued fields (0, 0.0, false, empty string) are omitted from the
  wire; repeated occurrences of a scalar field decode last-field-wins

If the proto definition changes, update these stubs or regenerate with
``grpc_tools.protoc`` — and keep ``purity_service.proto`` byte-identical to
Qbert0G's copy.
"""

from __future__ import annotations

from dataclasses import dataclass

from qr_sampler.proto.wire import (
    decode_fixed64,
    decode_varint,
    encode_fixed64,
    encode_tag,
    encode_varint,
)

# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


@dataclass
class DrawRequest:
    """Server-side integrated draw request message.

    Attributes:
        sequence_id: Commitment nonce; echoed verbatim (proto field 1, int64).
        source_id: Source id to integrate over; ``""`` = the API key's
            binding (proto field 2, string).
        block_bytes: Raw block size; 0 = server default
            (``integration.block_bytes``) (proto field 3, uint32).
    """

    sequence_id: int = 0
    source_id: str = ""
    block_bytes: int = 0

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to standard protobuf wire format.

        Proto3 omits default-valued fields (0, empty string) from the wire.
        """
        parts: list[bytes] = []
        if self.sequence_id != 0:
            # Field 1, wire type 0 (varint) — int64
            parts.append(encode_tag(1, 0))
            parts.append(encode_varint(self.sequence_id))
        if self.source_id:
            # Field 2, wire type 2 (length-delimited) — string
            source_bytes = self.source_id.encode("utf-8")
            parts.append(encode_tag(2, 2))
            parts.append(encode_varint(len(source_bytes)))
            parts.append(source_bytes)
        if self.block_bytes != 0:
            # Field 3, wire type 0 (varint) — uint32
            parts.append(encode_tag(3, 0))
            parts.append(encode_varint(self.block_bytes))
        return b"".join(parts)

    @classmethod
    def FromString(cls, data: bytes) -> DrawRequest:  # noqa: N802
        """Deserialize from standard protobuf wire format."""
        sequence_id = 0
        source_id = ""
        block_bytes = 0
        offset = 0
        while offset < len(data):
            tag, offset = decode_varint(data, offset)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:
                value, offset = decode_varint(data, offset)
                if field_number == 1:
                    sequence_id = value
                elif field_number == 3:
                    block_bytes = value
            elif wire_type == 2:
                length, offset = decode_varint(data, offset)
                raw = data[offset : offset + length]
                offset += length
                if field_number == 2:
                    source_id = raw.decode("utf-8")
            elif wire_type == 5:
                offset += 4  # Skip unknown 32-bit fields
            elif wire_type == 1:
                offset += 8  # Skip unknown 64-bit fields
            else:
                break  # Unknown wire type — stop parsing
        return cls(sequence_id=sequence_id, source_id=source_id, block_bytes=block_bytes)


@dataclass
class DrawResponse:
    """Server-side integrated draw response message.

    Attributes:
        u: Phi(z), clamped server-side to (1e-10, 1-1e-10) — so ``u == 0.0``
            on the wire is unambiguously "absent" (proto field 1, double).
        z: Baseline-referenced statistic (proto field 2, double).
        sequence_id: Echoed verbatim from the request (proto field 3, int64).
        generation_timestamp_ns: Last contributing raw measurement
            (proto field 4, uint64).
        source_id: The SERVING source id (proto field 5, string).
        coherence_z: Fisher z_c; meaningless unless ``coherence_valid``
            (proto field 6, double).
        coherence_valid: False => ignore ``coherence_z`` and ``coherence_r``
            (proto field 7, bool).
        purity_label: Canonical purity label string (proto field 8, string).
        integrated_bytes: Raw bytes integrated into this draw
            (proto field 9, uint32).
        integrator: Registry name, e.g. ``"bit_z"`` (proto field 10, string).
        coherence_r: Peak lag-scanned Pearson r (proto field 11, double).
    """

    u: float = 0.0
    z: float = 0.0
    sequence_id: int = 0
    generation_timestamp_ns: int = 0
    source_id: str = ""
    coherence_z: float = 0.0
    coherence_valid: bool = False
    purity_label: str = ""
    integrated_bytes: int = 0
    integrator: str = ""
    coherence_r: float = 0.0

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to standard protobuf wire format.

        Proto3 omits default-valued fields (0, 0.0, false, "") from the wire.
        """
        parts: list[bytes] = []
        if self.u != 0.0:
            # Field 1, wire type 1 (fixed64) — double
            parts.append(encode_tag(1, 1))
            parts.append(encode_fixed64(self.u))
        if self.z != 0.0:
            # Field 2, wire type 1 (fixed64) — double
            parts.append(encode_tag(2, 1))
            parts.append(encode_fixed64(self.z))
        if self.sequence_id != 0:
            # Field 3, wire type 0 (varint) — int64
            parts.append(encode_tag(3, 0))
            parts.append(encode_varint(self.sequence_id))
        if self.generation_timestamp_ns != 0:
            # Field 4, wire type 0 (varint) — uint64
            parts.append(encode_tag(4, 0))
            parts.append(encode_varint(self.generation_timestamp_ns))
        if self.source_id:
            # Field 5, wire type 2 (length-delimited) — string
            source_bytes = self.source_id.encode("utf-8")
            parts.append(encode_tag(5, 2))
            parts.append(encode_varint(len(source_bytes)))
            parts.append(source_bytes)
        if self.coherence_z != 0.0:
            # Field 6, wire type 1 (fixed64) — double
            parts.append(encode_tag(6, 1))
            parts.append(encode_fixed64(self.coherence_z))
        if self.coherence_valid:
            # Field 7, wire type 0 (varint) — bool
            parts.append(encode_tag(7, 0))
            parts.append(encode_varint(1))
        if self.purity_label:
            # Field 8, wire type 2 (length-delimited) — string
            label_bytes = self.purity_label.encode("utf-8")
            parts.append(encode_tag(8, 2))
            parts.append(encode_varint(len(label_bytes)))
            parts.append(label_bytes)
        if self.integrated_bytes != 0:
            # Field 9, wire type 0 (varint) — uint32
            parts.append(encode_tag(9, 0))
            parts.append(encode_varint(self.integrated_bytes))
        if self.integrator:
            # Field 10, wire type 2 (length-delimited) — string
            integrator_bytes = self.integrator.encode("utf-8")
            parts.append(encode_tag(10, 2))
            parts.append(encode_varint(len(integrator_bytes)))
            parts.append(integrator_bytes)
        if self.coherence_r != 0.0:
            # Field 11, wire type 1 (fixed64) — double
            parts.append(encode_tag(11, 1))
            parts.append(encode_fixed64(self.coherence_r))
        return b"".join(parts)

    @classmethod
    def FromString(cls, data: bytes) -> DrawResponse:  # noqa: N802
        """Deserialize from standard protobuf wire format.

        Proto3 semantics: the LAST occurrence of a repeated scalar field
        wins; unknown fields of any wire type are skipped cleanly.
        """
        msg = cls()
        offset = 0
        while offset < len(data):
            tag, offset = decode_varint(data, offset)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:
                value, offset = decode_varint(data, offset)
                if field_number == 3:
                    msg.sequence_id = value
                elif field_number == 4:
                    msg.generation_timestamp_ns = value
                elif field_number == 7:
                    msg.coherence_valid = bool(value)
                elif field_number == 9:
                    msg.integrated_bytes = value
            elif wire_type == 1:
                fvalue, offset = decode_fixed64(data, offset)
                if field_number == 1:
                    msg.u = fvalue
                elif field_number == 2:
                    msg.z = fvalue
                elif field_number == 6:
                    msg.coherence_z = fvalue
                elif field_number == 11:
                    msg.coherence_r = fvalue
                # else: skip unknown fixed64 fields (offset already advanced)
            elif wire_type == 2:
                length, offset = decode_varint(data, offset)
                raw = data[offset : offset + length]
                offset += length
                if field_number == 5:
                    msg.source_id = raw.decode("utf-8")
                elif field_number == 8:
                    msg.purity_label = raw.decode("utf-8")
                elif field_number == 10:
                    msg.integrator = raw.decode("utf-8")
            elif wire_type == 5:
                offset += 4  # Skip unknown 32-bit fields
            else:
                break  # Unknown wire type — stop parsing
        return msg
