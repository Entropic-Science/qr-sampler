"""Tests for the single protobuf wire format (FROZEN GATE).

Validates the primitive codec in ``qr_sampler.proto.wire``, the
hand-written ``EntropyRequest`` / ``EntropyResponse`` message stubs, and
the transport-level ``encode_request`` / ``decode_reply`` seam that the
gRPC entropy source rides — enabling interoperability with any standard
gRPC server (including ``grpcurl`` and the user's ``qrng.QuantumRNG``).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from qr_sampler.entropy.qgrpc.transport import decode_reply, encode_request
from qr_sampler.exceptions import EntropyUnavailableError
from qr_sampler.proto.entropy_service_pb2 import EntropyRequest, EntropyResponse
from qr_sampler.proto.purity_service_pb2 import DrawRequest, DrawResponse
from qr_sampler.proto.wire import decode_fixed64, decode_varint, encode_fixed64, encode_varint

# ---------------------------------------------------------------------------
# Varint encoding/decoding
# ---------------------------------------------------------------------------


class TestVarint:
    """Test low-level varint encoding and decoding."""

    @pytest.mark.parametrize(
        ("value", "expected_bytes"),
        [
            (0, b"\x00"),
            (1, b"\x01"),
            (127, b"\x7f"),
            (128, b"\x80\x01"),
            (300, b"\xac\x02"),
            (16384, b"\x80\x80\x01"),
            (20480, b"\x80\xa0\x01"),
        ],
    )
    def test_encode_known_values(self, value: int, expected_bytes: bytes) -> None:
        assert encode_varint(value) == expected_bytes

    @pytest.mark.parametrize(
        ("encoded", "expected_value"),
        [
            (b"\x00", 0),
            (b"\x01", 1),
            (b"\x7f", 127),
            (b"\x80\x01", 128),
            (b"\xac\x02", 300),
            (b"\x80\xa0\x01", 20480),
        ],
    )
    def test_decode_known_values(self, encoded: bytes, expected_value: int) -> None:
        value, offset = decode_varint(encoded, 0)
        assert value == expected_value
        assert offset == len(encoded)

    def test_roundtrip(self) -> None:
        for v in [0, 1, 42, 127, 128, 255, 256, 1000, 20480, 65535, 2**20]:
            encoded = encode_varint(v)
            decoded, _ = decode_varint(encoded, 0)
            assert decoded == v


# ---------------------------------------------------------------------------
# EntropyRequest wire format
# ---------------------------------------------------------------------------


class TestEntropyRequestWireFormat:
    """Test that EntropyRequest produces standard protobuf encoding."""

    def test_empty_request_serializes_to_empty(self) -> None:
        """Proto3: all-default-valued fields produce empty bytes."""
        req = EntropyRequest(bytes_needed=0, sequence_id=0)
        assert req.SerializeToString() == b""

    def test_known_encoding_bytes_needed_only(self) -> None:
        """Field 1, varint 100 -> tag=0x08, value=0x64."""
        req = EntropyRequest(bytes_needed=100)
        wire = req.SerializeToString()
        # Tag: field_number=1, wire_type=0 -> (1<<3)|0 = 0x08
        # Value: 100 = 0x64
        assert wire == b"\x08\x64"

    def test_known_encoding_both_fields(self) -> None:
        """bytes_needed=100, sequence_id=42."""
        req = EntropyRequest(bytes_needed=100, sequence_id=42)
        wire = req.SerializeToString()
        # Field 1: tag=0x08, value=100 (0x64)
        # Field 2: tag=0x10, value=42 (0x2a)
        assert wire == b"\x08\x64\x10\x2a"

    def test_known_encoding_large_varint(self) -> None:
        """bytes_needed=20480 requires multi-byte varint."""
        req = EntropyRequest(bytes_needed=20480)
        wire = req.SerializeToString()
        # Tag: 0x08, Value: 20480 = 0x80 0xa0 0x01
        assert wire == b"\x08\x80\xa0\x01"

    def test_roundtrip(self) -> None:
        req = EntropyRequest(bytes_needed=20480, sequence_id=99)
        wire = req.SerializeToString()
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 20480
        assert decoded.sequence_id == 99

    def test_roundtrip_defaults(self) -> None:
        req = EntropyRequest()
        wire = req.SerializeToString()
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 0
        assert decoded.sequence_id == 0

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown fields should be silently skipped."""
        # Valid EntropyRequest(bytes_needed=100) + an unknown field 5 varint=99
        wire = b"\x08\x64" + b"\x28\x63"
        decoded = EntropyRequest.FromString(wire)
        assert decoded.bytes_needed == 100
        assert decoded.sequence_id == 0

    def test_from_string_empty_bytes(self) -> None:
        decoded = EntropyRequest.FromString(b"")
        assert decoded.bytes_needed == 0
        assert decoded.sequence_id == 0


# ---------------------------------------------------------------------------
# EntropyResponse wire format
# ---------------------------------------------------------------------------


class TestEntropyResponseWireFormat:
    """Test that EntropyResponse produces standard protobuf encoding."""

    def test_empty_response_serializes_to_empty(self) -> None:
        resp = EntropyResponse()
        assert resp.SerializeToString() == b""

    def test_known_encoding_data_only(self) -> None:
        """Field 1 (bytes), 3 bytes of data."""
        resp = EntropyResponse(data=b"\xaa\xbb\xcc")
        wire = resp.SerializeToString()
        # Tag: field_number=1, wire_type=2 -> (1<<3)|2 = 0x0a
        # Length: 3 = 0x03
        # Data: \xaa\xbb\xcc
        assert wire == b"\x0a\x03\xaa\xbb\xcc"

    def test_known_encoding_all_fields(self) -> None:
        """All four fields populated."""
        resp = EntropyResponse(
            data=b"\x42",
            sequence_id=1,
            generation_timestamp_ns=1000,
            device_id="dev1",
        )
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b"\x42"
        assert decoded.sequence_id == 1
        assert decoded.generation_timestamp_ns == 1000
        assert decoded.device_id == "dev1"

    def test_roundtrip(self) -> None:
        resp = EntropyResponse(
            data=b"\x00\x01\x02\x03\x04" * 100,
            sequence_id=12345,
            generation_timestamp_ns=1_700_000_000_000_000_000,
            device_id="dragonfly-0",
        )
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == resp.data
        assert decoded.sequence_id == 12345
        assert decoded.generation_timestamp_ns == 1_700_000_000_000_000_000
        assert decoded.device_id == "dragonfly-0"

    def test_roundtrip_defaults(self) -> None:
        resp = EntropyResponse()
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b""
        assert decoded.sequence_id == 0
        assert decoded.generation_timestamp_ns == 0
        assert decoded.device_id == ""

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown fields should be silently skipped."""
        resp = EntropyResponse(data=b"\x42", sequence_id=1)
        wire = resp.SerializeToString()
        # Append an unknown field 10, varint=99
        wire += b"\x50\x63"
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == b"\x42"
        assert decoded.sequence_id == 1

    def test_from_string_empty_bytes(self) -> None:
        decoded = EntropyResponse.FromString(b"")
        assert decoded.data == b""
        assert decoded.sequence_id == 0

    def test_large_data_payload(self) -> None:
        """Test with the typical 20KB entropy payload."""
        payload = bytes(range(256)) * 80  # 20480 bytes
        resp = EntropyResponse(data=payload, device_id="test")
        wire = resp.SerializeToString()
        decoded = EntropyResponse.FromString(wire)
        assert decoded.data == payload
        assert len(decoded.data) == 20480


# ---------------------------------------------------------------------------
# Cross-compatibility with generic wire-format decoder
# ---------------------------------------------------------------------------


class TestCrossCompatibility:
    """Verify that EntropyRequest wire output is decodable by a generic
    protobuf field-1 extractor (the pattern the qgrpc transport relies on for
    protocol-agnostic gRPC).
    """

    def test_request_field1_is_varint(self) -> None:
        """EntropyRequest.SerializeToString() field 1 should be decodable
        as a generic varint.
        """
        req = EntropyRequest(bytes_needed=20480)
        wire = req.SerializeToString()
        # Parse the tag
        tag, offset = decode_varint(wire, 0)
        field_number = tag >> 3
        wire_type = tag & 0x07
        assert field_number == 1
        assert wire_type == 0  # varint
        value, _ = decode_varint(wire, offset)
        assert value == 20480

    def test_response_field1_is_length_delimited(self) -> None:
        """EntropyResponse.SerializeToString() field 1 should be decodable
        as a generic length-delimited bytes extraction.
        """
        payload = b"\xde\xad\xbe\xef"
        resp = EntropyResponse(data=payload, sequence_id=1)
        wire = resp.SerializeToString()
        # Parse the tag
        tag, offset = decode_varint(wire, 0)
        field_number = tag >> 3
        wire_type = tag & 0x07
        assert field_number == 1
        assert wire_type == 2  # length-delimited
        length, offset = decode_varint(wire, offset)
        assert length == 4
        assert wire[offset : offset + length] == payload

    def test_request_with_only_field1_produces_minimal_wire(self) -> None:
        """A request with only bytes_needed set should produce the same wire
        bytes regardless of whether it's an EntropyRequest or a generic
        'field 1 varint' encoder.
        """
        req = EntropyRequest(bytes_needed=100)
        wire = req.SerializeToString()
        # Manually construct: tag(1, varint) + varint(100)
        expected = encode_varint((1 << 3) | 0) + encode_varint(100)
        assert wire == expected


# ---------------------------------------------------------------------------
# Transport-level request encoding (commitment nonce rides in sequence_id)
# ---------------------------------------------------------------------------


class TestTransportRequestEncoding:
    """``encode_request`` — the single request encoder the transport uses."""

    def test_encode_small(self) -> None:
        """tag 0x08 (field 1, varint), value 100 = 0x64."""
        assert encode_request(100) == b"\x08\x64"

    def test_encode_large(self) -> None:
        """20480 = 0x5000 -> LEB128: 0x80 0xa0 0x01."""
        assert encode_request(20480) == b"\x08\x80\xa0\x01"

    def test_encode_zero_is_empty(self) -> None:
        """Zero byte count produces empty bytes (proto3 default omission)."""
        assert encode_request(0) == b""

    def test_encode_with_sequence_id(self) -> None:
        """Non-zero sequence_id appends field 2 (varint)."""
        assert encode_request(100, 42) == b"\x08\x64\x10\x2a"

    def test_encode_zero_sequence_id_is_byte_identical_to_nonce_less(self) -> None:
        """Zero nonce must produce the exact nonce-less request bytes."""
        assert encode_request(100) == b"\x08\x64"
        assert encode_request(100, 0) == b"\x08\x64"

    def test_encode_roundtrips_through_message_class(self) -> None:
        """Encoded requests parse back through EntropyRequest."""
        for n, seq in ((256, 1), (10000, 2**62), (64, 0x7FFFFFFFFFFFFFFF)):
            msg = EntropyRequest.FromString(encode_request(n, seq))
            assert msg.bytes_needed == n
            assert msg.sequence_id == seq


# ---------------------------------------------------------------------------
# Transport-level response decoding
# ---------------------------------------------------------------------------


def _mock_response(data: bytes) -> bytes:
    """Field 1 = length-delimited bytes."""
    return b"\x0a" + encode_varint(len(data)) + data


class TestTransportResponseDecoding:
    """``decode_reply`` — the single response decoder the transport uses."""

    def test_decode_extracts_all_fields(self) -> None:
        """Decoder returns (payload, sequence_id echo, generation_ts)."""
        msg = EntropyResponse(
            data=b"\xab" * 8,
            sequence_id=777,
            generation_timestamp_ns=123456789,
        )
        payload, seq, gen_ts = decode_reply(msg.SerializeToString())
        assert payload == b"\xab" * 8
        assert seq == 777
        assert gen_ts == 123456789

    def test_decode_defaults_absent_fields_to_zero(self) -> None:
        """Servers that don't echo sequence_id yield (payload, 0, 0)."""
        payload, seq, gen_ts = decode_reply(_mock_response(b"\x01\x02"))
        assert payload == b"\x01\x02"
        assert seq == 0
        assert gen_ts == 0

    def test_decode_with_extra_fields(self) -> None:
        """Payload extraction works even when other fields come first."""
        # Field 2 (varint): tag=0x10, value=42; then field 1 (bytes) "abc".
        wire = b"\x10\x2a" + b"\x0a\x03abc"
        assert decode_reply(wire)[0] == b"abc"

    def test_decode_missing_payload_raises(self) -> None:
        """Should raise when field 1 bytes is missing (or on empty input)."""
        with pytest.raises(EntropyUnavailableError, match="field 1"):
            decode_reply(b"\x10\x2a")  # only a varint field 2
        with pytest.raises(EntropyUnavailableError, match="field 1"):
            decode_reply(b"")

    def test_decode_last_field1_occurrence_wins(self) -> None:
        """Repeated field 1: LAST occurrence wins (pb2/proto3 semantics).

        Recorded behavior change #6 — the old hand-rolled decoder kept the
        FIRST occurrence. Byte-identical for every real server (field 1 is
        sent exactly once); this pin documents the chosen semantics.
        """
        wire = _mock_response(b"first") + _mock_response(b"last")
        assert decode_reply(wire)[0] == b"last"


class TestQbertResponseShape:
    """Pin decode behaviour against the production qbert qrng.proto.

    RandomResponse: field 1 = bytes data, field 2 = uint64 timestamp
    (epoch MICROseconds), field 3 = string device_id. The decoder reads
    field 2 into its sequence_id slot (documented collision — can never
    match a nonce) and must skip the wire-type-2 device_id cleanly.
    """

    def test_qbert_response_decodes_payload_and_skips_device_id(self) -> None:
        payload = b"\xaa" * 16
        timestamp_us = 1_781_159_892_384_000
        device_id = b"qbert-device-01"
        wire = (
            b"\x0a"
            + encode_varint(len(payload))
            + payload  # field 1, bytes
            + b"\x10"
            + encode_varint(timestamp_us)  # field 2, varint
            + b"\x1a"
            + encode_varint(len(device_id))
            + device_id  # field 3, str
        )
        decoded_payload, seq, gen_ts = decode_reply(wire)
        assert decoded_payload == payload
        # The documented collision: field 2 lands in the sequence_id slot.
        assert seq == timestamp_us
        # device_id is wire-type 2 at field 3 — skipped, not misread as ts.
        assert gen_ts == 0

    def test_qbert_timestamp_never_verifies_as_echo(self) -> None:
        """A 63-bit nonce can't collide with an epoch-us timestamp here."""
        nonce = 0x7FEDCBA987654321
        timestamp_us = 1_781_159_892_384_000
        assert nonce != timestamp_us


# ---------------------------------------------------------------------------
# gRPC stubs (client stub + servicer registration)
# ---------------------------------------------------------------------------


class TestGrpcStubs:
    """The hand-written client stub / servicer helpers stay wire-compatible."""

    def test_stub_binds_method_paths_and_codec(self) -> None:
        from unittest.mock import MagicMock

        from qr_sampler.proto.entropy_service_pb2_grpc import EntropyServiceStub

        channel = MagicMock()
        EntropyServiceStub(channel)
        unary_args = channel.unary_unary.call_args
        stream_args = channel.stream_stream.call_args
        assert unary_args[0][0] == "/qr_entropy.EntropyService/GetEntropy"
        assert stream_args[0][0] == "/qr_entropy.EntropyService/StreamEntropy"

        # The registered (de)serializers round-trip real messages.
        ser = unary_args.kwargs["request_serializer"]
        deser = unary_args.kwargs["response_deserializer"]
        req = EntropyRequest(bytes_needed=64, sequence_id=9)
        assert EntropyRequest.FromString(ser(req)) == req
        resp = EntropyResponse(data=b"\x42" * 4, sequence_id=9)
        assert deser(resp.SerializeToString()) == resp

    def test_servicer_registration(self) -> None:
        pytest.importorskip("grpc", reason="grpcio not installed")
        from unittest.mock import MagicMock

        from qr_sampler.proto.entropy_service_pb2_grpc import (
            EntropyServiceServicer,
            add_EntropyServiceServicer_to_server,
        )

        server = MagicMock()
        add_EntropyServiceServicer_to_server(EntropyServiceServicer(), server)
        server.add_generic_rpc_handlers.assert_called_once()

    def test_default_servicer_methods_are_unimplemented(self) -> None:
        pytest.importorskip("grpc", reason="grpcio not installed")
        from unittest.mock import MagicMock

        from qr_sampler.proto.entropy_service_pb2_grpc import EntropyServiceServicer

        servicer = EntropyServiceServicer()
        with pytest.raises(NotImplementedError):
            servicer.GetEntropy(EntropyRequest(), MagicMock())
        with pytest.raises(NotImplementedError):
            servicer.StreamEntropy(iter(()), MagicMock())


# ---------------------------------------------------------------------------
# Fixed64 encoding/decoding (the double codec for qr_purity draws)
# ---------------------------------------------------------------------------


class TestFixed64:
    """Test the low-level fixed64 (IEEE-754 double, little-endian) codec."""

    @pytest.mark.parametrize(
        ("value", "expected_bytes"),
        [
            (0.0, b"\x00\x00\x00\x00\x00\x00\x00\x00"),
            (1.0, b"\x00\x00\x00\x00\x00\x00\xf0\x3f"),
            (0.5, b"\x00\x00\x00\x00\x00\x00\xe0\x3f"),
            (-1.0, b"\x00\x00\x00\x00\x00\x00\xf0\xbf"),
        ],
    )
    def test_encode_known_values(self, value: float, expected_bytes: bytes) -> None:
        assert encode_fixed64(value) == expected_bytes

    @pytest.mark.parametrize(
        ("encoded", "expected_value"),
        [
            (b"\x00\x00\x00\x00\x00\x00\xf0\x3f", 1.0),
            (b"\x00\x00\x00\x00\x00\x00\xe0\x3f", 0.5),
            (b"\x00\x00\x00\x00\x00\x00\xf0\xbf", -1.0),
        ],
    )
    def test_decode_known_values(self, encoded: bytes, expected_value: float) -> None:
        value, offset = decode_fixed64(encoded, 0)
        assert value == expected_value
        assert offset == 8

    def test_decode_respects_offset(self) -> None:
        data = b"\xff\xff" + encode_fixed64(0.5)
        value, offset = decode_fixed64(data, 2)
        assert value == 0.5
        assert offset == 10

    def test_nan_roundtrips(self) -> None:
        """NaN survives encode/decode (payload bits preserved by struct)."""
        encoded = encode_fixed64(float("nan"))
        assert len(encoded) == 8
        value, offset = decode_fixed64(encoded, 0)
        assert math.isnan(value)
        assert offset == 8

    def test_roundtrip(self) -> None:
        for v in [0.0, 1e-10, 1.0 - 1e-10, 0.25, -3.5, 1e308, -1e-308, math.pi]:
            encoded = encode_fixed64(v)
            decoded, _ = decode_fixed64(encoded, 0)
            assert decoded == v


# ---------------------------------------------------------------------------
# DrawRequest wire format (qr_purity.DrawRequest)
# ---------------------------------------------------------------------------


class TestDrawRequestWireFormat:
    """Test that DrawRequest produces standard protobuf encoding."""

    def test_empty_request_serializes_to_empty(self) -> None:
        """Proto3: all-default-valued fields produce empty bytes."""
        req = DrawRequest(sequence_id=0, source_id="", block_bytes=0)
        assert req.SerializeToString() == b""

    def test_known_encoding_sequence_id_only(self) -> None:
        """Field 1, varint 42 -> tag=0x08, value=0x2a."""
        req = DrawRequest(sequence_id=42)
        assert req.SerializeToString() == b"\x08\x2a"

    def test_known_encoding_all_fields(self) -> None:
        """sequence_id=42, source_id='dev', block_bytes=2 MiB."""
        req = DrawRequest(sequence_id=42, source_id="dev", block_bytes=2_097_152)
        wire = req.SerializeToString()
        # Field 1: tag=0x08, value=42 (0x2a)
        # Field 2: tag=0x12, len=3, "dev"
        # Field 3: tag=0x18, value=2097152 = 0x80 0x80 0x80 0x01
        assert wire == b"\x08\x2a" + b"\x12\x03dev" + b"\x18\x80\x80\x80\x01"

    def test_roundtrip(self) -> None:
        req = DrawRequest(sequence_id=0x7FFFFFFFFFFFFFFF, source_id="dragonfly-0", block_bytes=1024)
        decoded = DrawRequest.FromString(req.SerializeToString())
        assert decoded == req

    def test_roundtrip_defaults(self) -> None:
        decoded = DrawRequest.FromString(DrawRequest().SerializeToString())
        assert decoded.sequence_id == 0
        assert decoded.source_id == ""
        assert decoded.block_bytes == 0

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown varint/fixed64/len-delimited fields are silently skipped."""
        wire = (
            b"\x08\x2a"  # field 1 varint = 42
            + b"\x28\x63"  # unknown field 5 varint
            + b"\x31"
            + b"\x00" * 8  # unknown field 6 fixed64
            + b"\x3a\x02ab"  # unknown field 7 length-delimited
        )
        decoded = DrawRequest.FromString(wire)
        assert decoded.sequence_id == 42
        assert decoded.source_id == ""
        assert decoded.block_bytes == 0

    def test_from_string_empty_bytes(self) -> None:
        decoded = DrawRequest.FromString(b"")
        assert decoded == DrawRequest()


# ---------------------------------------------------------------------------
# DrawResponse wire format (qr_purity.DrawResponse)
# ---------------------------------------------------------------------------


class TestDrawResponseWireFormat:
    """Test that DrawResponse produces standard protobuf encoding."""

    def test_empty_response_serializes_to_empty(self) -> None:
        assert DrawResponse().SerializeToString() == b""

    def test_known_encoding_u_only(self) -> None:
        """Field 1 (double), wire type 1 -> tag=(1<<3)|1=0x09 + 8 LE bytes."""
        resp = DrawResponse(u=0.5)
        assert resp.SerializeToString() == b"\x09" + b"\x00\x00\x00\x00\x00\x00\xe0\x3f"

    def test_known_encoding_selected_fields(self) -> None:
        """u=1.0, sequence_id=7, coherence_valid=True."""
        resp = DrawResponse(u=1.0, sequence_id=7, coherence_valid=True)
        wire = resp.SerializeToString()
        # Field 1: tag=0x09 + fixed64(1.0)
        # Field 3: tag=0x18, value=7
        # Field 7: tag=0x38, value=1
        assert wire == (b"\x09" + b"\x00\x00\x00\x00\x00\x00\xf0\x3f" + b"\x18\x07" + b"\x38\x01")

    def test_roundtrip_all_fields(self) -> None:
        resp = DrawResponse(
            u=0.734,
            z=0.625,
            sequence_id=987654321,
            generation_timestamp_ns=1_700_000_000_000_000_000,
            source_id="dragonfly-0",
            coherence_z=4.25,
            coherence_valid=True,
            purity_label="quantum/intact/raw/amplified:2097152/qf:99+/QV",
            integrated_bytes=2_097_152,
            integrator="bit_z",
            coherence_r=0.125,
        )
        decoded = DrawResponse.FromString(resp.SerializeToString())
        assert decoded == resp

    def test_roundtrip_defaults(self) -> None:
        decoded = DrawResponse.FromString(DrawResponse().SerializeToString())
        assert decoded == DrawResponse()
        assert decoded.u == 0.0
        assert decoded.coherence_valid is False

    def test_last_field_occurrence_wins(self) -> None:
        """Repeated scalar field: LAST occurrence wins (proto3 semantics)."""
        first = DrawResponse(u=0.25).SerializeToString()
        last = DrawResponse(u=0.75).SerializeToString()
        decoded = DrawResponse.FromString(first + last)
        assert decoded.u == 0.75

    def test_from_string_skips_unknown_fields(self) -> None:
        """Unknown fields of every wire type are silently skipped."""
        wire = DrawResponse(u=0.5, integrator="bit_z").SerializeToString()
        wire += b"\x60\x63"  # unknown field 12 varint
        wire += b"\x69" + b"\x00" * 8  # unknown field 13 fixed64
        wire += b"\x72\x02xy"  # unknown field 14 length-delimited
        wire += b"\x7d" + b"\x00" * 4  # unknown field 15 fixed32
        decoded = DrawResponse.FromString(wire)
        assert decoded.u == 0.5
        assert decoded.integrator == "bit_z"

    def test_from_string_empty_bytes(self) -> None:
        assert DrawResponse.FromString(b"") == DrawResponse()

    def test_clamped_u_boundaries_roundtrip_exactly(self) -> None:
        """The server's clamp bounds survive the wire bit-for-bit."""
        for u in (1e-10, 1.0 - 1e-10):
            decoded = DrawResponse.FromString(DrawResponse(u=u).SerializeToString())
            assert decoded.u == u


# ---------------------------------------------------------------------------
# Proto byte-identity (cross-repo pin with Qbert0G)
# ---------------------------------------------------------------------------

#: sha256 over the CRLF->LF-normalized bytes of ``purity_service.proto``.
#: The SAME constant is pinned in Qbert0G's ``tests/test_purity_service.py``,
#: so both repos' copies are transitively byte-identical without cross-repo
#: file access. Never change one pin without the other.
PURITY_PROTO_SHA256 = "738af813b298c5ee8f161afb0e69cfe44f12e0dbb9fe0d3cfc754e05574c0bc3"


class TestPurityProtoIdentity:
    """``purity_service.proto`` is byte-identical to Qbert0G's copy."""

    def test_proto_file_sha256_pin(self) -> None:
        proto_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "qr_sampler"
            / "proto"
            / "purity_service.proto"
        )
        normalized = proto_path.read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(normalized).hexdigest() == PURITY_PROTO_SHA256


# ---------------------------------------------------------------------------
# PurityService gRPC stubs (client stub + servicer registration)
# ---------------------------------------------------------------------------


class TestPurityGrpcStubs:
    """The hand-written PurityService stub / servicer helpers stay wire-compatible."""

    def test_stub_binds_method_paths_and_codec(self) -> None:
        from unittest.mock import MagicMock

        from qr_sampler.proto.purity_service_pb2_grpc import PurityServiceStub

        channel = MagicMock()
        PurityServiceStub(channel)
        unary_args = channel.unary_unary.call_args
        stream_args = channel.stream_stream.call_args
        assert unary_args[0][0] == "/qr_purity.PurityService/GetDraw"
        assert stream_args[0][0] == "/qr_purity.PurityService/StreamDraws"

        # The registered (de)serializers round-trip real messages.
        ser = unary_args.kwargs["request_serializer"]
        deser = unary_args.kwargs["response_deserializer"]
        req = DrawRequest(sequence_id=9, source_id="dev", block_bytes=64)
        assert DrawRequest.FromString(ser(req)) == req
        resp = DrawResponse(u=0.5, z=1.0, sequence_id=9)
        assert deser(resp.SerializeToString()) == resp

    def test_servicer_serializers_roundtrip(self) -> None:
        pytest.importorskip("grpc", reason="grpcio not installed")
        from unittest.mock import MagicMock, patch

        from qr_sampler.proto import purity_service_pb2_grpc as grpc_mod
        from qr_sampler.proto.purity_service_pb2_grpc import (
            PurityServiceServicer,
            add_PurityServiceServicer_to_server,
        )

        # The server-side (de)serializer pair mirrors the client stub's.
        req = DrawRequest(sequence_id=3, source_id="dev")
        assert grpc_mod._draw_request_deserializer(req.SerializeToString()) == req
        resp = DrawResponse(u=0.5, sequence_id=3)
        assert grpc_mod._draw_response_serializer(resp) == resp.SerializeToString()

        server = MagicMock()
        with patch("grpc.method_handlers_generic_handler") as handler_fn:
            add_PurityServiceServicer_to_server(PurityServiceServicer(), server)
            assert handler_fn.call_args[0][0] == "qr_purity.PurityService"
        server.add_generic_rpc_handlers.assert_called_once()

    def test_servicer_registration(self) -> None:
        pytest.importorskip("grpc", reason="grpcio not installed")
        from unittest.mock import MagicMock

        from qr_sampler.proto.purity_service_pb2_grpc import (
            PurityServiceServicer,
            add_PurityServiceServicer_to_server,
        )

        server = MagicMock()
        add_PurityServiceServicer_to_server(PurityServiceServicer(), server)
        server.add_generic_rpc_handlers.assert_called_once()

    def test_default_servicer_methods_are_unimplemented(self) -> None:
        pytest.importorskip("grpc", reason="grpcio not installed")
        from unittest.mock import MagicMock

        from qr_sampler.proto.purity_service_pb2_grpc import PurityServiceServicer

        servicer = PurityServiceServicer()
        with pytest.raises(NotImplementedError):
            servicer.GetDraw(DrawRequest(), MagicMock())
        with pytest.raises(NotImplementedError):
            servicer.StreamDraws(iter(()), MagicMock())
