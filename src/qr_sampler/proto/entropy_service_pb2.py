"""Hand-written protobuf message stubs for the entropy service.

These are lightweight message classes that mirror the ``entropy_service.proto``
definition using **standard protobuf wire encoding** (primitives from
:mod:`qr_sampler.proto.wire`). They produce bytes
identical to ``protoc``-generated code, making them compatible with any
standard gRPC server (including ``grpcurl``).

Wire format reference (proto3):
- Varint fields: tag = (field_number << 3 | 0), then LEB128-encoded value
- Length-delimited fields: tag = (field_number << 3 | 2), then varint length, then raw bytes
- Default-valued fields (0, empty bytes, empty string) are omitted from the wire

If the proto definition changes, update these stubs or regenerate with
``grpc_tools.protoc``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qr_sampler.proto.wire import decode_varint, encode_tag, encode_varint

# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


@dataclass
class EntropyRequest:
    """Entropy generation request message.

    Attributes:
        bytes_needed: Number of random bytes requested (proto field 1, int32).
        sequence_id: Correlates request to response in streaming mode (proto field 2, int64).
    """

    bytes_needed: int = 0
    sequence_id: int = 0

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to standard protobuf wire format.

        Proto3 omits default-valued fields (0) from the wire.
        """
        parts: list[bytes] = []
        if self.bytes_needed != 0:
            # Field 1, wire type 0 (varint) — int32
            parts.append(encode_tag(1, 0))
            parts.append(encode_varint(self.bytes_needed))
        if self.sequence_id != 0:
            # Field 2, wire type 0 (varint) — int64
            parts.append(encode_tag(2, 0))
            parts.append(encode_varint(self.sequence_id))
        return b"".join(parts)

    @classmethod
    def FromString(cls, data: bytes) -> EntropyRequest:  # noqa: N802
        """Deserialize from standard protobuf wire format."""
        bytes_needed = 0
        sequence_id = 0
        offset = 0
        while offset < len(data):
            tag, offset = decode_varint(data, offset)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:
                value, offset = decode_varint(data, offset)
                if field_number == 1:
                    bytes_needed = value
                elif field_number == 2:
                    sequence_id = value
            elif wire_type == 2:
                length, offset = decode_varint(data, offset)
                offset += length  # Skip unknown length-delimited fields
            elif wire_type == 5:
                offset += 4  # Skip unknown 32-bit fields
            elif wire_type == 1:
                offset += 8  # Skip unknown 64-bit fields
            else:
                break  # Unknown wire type — stop parsing
        return cls(bytes_needed=bytes_needed, sequence_id=sequence_id)


@dataclass
class EntropyResponse:
    """Entropy generation response message.

    Attributes:
        data: Random bytes (proto field 1, bytes).
        sequence_id: Matches the request's ``sequence_id`` (proto field 2, int64).
        generation_timestamp_ns: When physical entropy was generated in nanoseconds
            (proto field 3, int64).
        device_id: QRNG hardware identifier (proto field 4, string).
    """

    data: bytes = field(default_factory=bytes)
    sequence_id: int = 0
    generation_timestamp_ns: int = 0
    device_id: str = ""

    def SerializeToString(self) -> bytes:  # noqa: N802
        """Serialize to standard protobuf wire format.

        Proto3 omits default-valued fields from the wire.
        """
        parts: list[bytes] = []
        if self.data:
            # Field 1, wire type 2 (length-delimited) — bytes
            parts.append(encode_tag(1, 2))
            parts.append(encode_varint(len(self.data)))
            parts.append(self.data)
        if self.sequence_id != 0:
            # Field 2, wire type 0 (varint) — int64
            parts.append(encode_tag(2, 0))
            parts.append(encode_varint(self.sequence_id))
        if self.generation_timestamp_ns != 0:
            # Field 3, wire type 0 (varint) — int64
            parts.append(encode_tag(3, 0))
            parts.append(encode_varint(self.generation_timestamp_ns))
        if self.device_id:
            # Field 4, wire type 2 (length-delimited) — string
            device_bytes = self.device_id.encode("utf-8")
            parts.append(encode_tag(4, 2))
            parts.append(encode_varint(len(device_bytes)))
            parts.append(device_bytes)
        return b"".join(parts)

    @classmethod
    def FromString(cls, data: bytes) -> EntropyResponse:  # noqa: N802
        """Deserialize from standard protobuf wire format."""
        resp_data = b""
        sequence_id = 0
        generation_timestamp_ns = 0
        device_id = ""
        offset = 0
        while offset < len(data):
            tag, offset = decode_varint(data, offset)
            field_number = tag >> 3
            wire_type = tag & 0x07
            if wire_type == 0:
                value, offset = decode_varint(data, offset)
                if field_number == 2:
                    sequence_id = value
                elif field_number == 3:
                    generation_timestamp_ns = value
            elif wire_type == 2:
                length, offset = decode_varint(data, offset)
                raw = data[offset : offset + length]
                offset += length
                if field_number == 1:
                    resp_data = raw
                elif field_number == 4:
                    device_id = raw.decode("utf-8")
            elif wire_type == 5:
                offset += 4  # Skip unknown 32-bit fields
            elif wire_type == 1:
                offset += 8  # Skip unknown 64-bit fields
            else:
                break
        return cls(
            data=resp_data,
            sequence_id=sequence_id,
            generation_timestamp_ns=generation_timestamp_ns,
            device_id=device_id,
        )
