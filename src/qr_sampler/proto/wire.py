"""Primitive protobuf wire-format codec — the single home of varint/tag math.

These helpers implement the proto3 wire primitives used by the hand-written
message stubs in :mod:`qr_sampler.proto.entropy_service_pb2`:

- Varint fields: tag = ``(field_number << 3 | 0)``, then LEB128-encoded value
- Length-delimited fields: tag = ``(field_number << 3 | 2)``, then varint
  length, then raw bytes
- Default-valued fields (0, empty bytes, empty string) are omitted from the
  wire

Everything that needs to touch raw wire bytes (message stubs, tests,
example servers) imports these public names; no private copies exist
elsewhere in the package.
"""

from __future__ import annotations


def encode_varint(value: int) -> bytes:
    """Encode an unsigned integer as a protobuf varint (LEB128).

    Args:
        value: Non-negative integer to encode.

    Returns:
        LEB128-encoded bytes.
    """
    parts: list[int] = []
    while value > 0x7F:
        parts.append((value & 0x7F) | 0x80)
        value >>= 7
    parts.append(value & 0x7F)
    return bytes(parts)


def decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """Decode a varint from bytes at the given offset.

    Args:
        data: Raw bytes.
        offset: Starting position.

    Returns:
        Tuple of (decoded_value, new_offset).
    """
    result = 0
    shift = 0
    while True:
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, offset


def encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encode a protobuf field tag.

    Args:
        field_number: The proto field number (1-based).
        wire_type: 0 = varint, 2 = length-delimited.

    Returns:
        Varint-encoded tag bytes.
    """
    return encode_varint((field_number << 3) | wire_type)
