"""Tests for the qgrpc transport modes (server streaming, bidi streaming).

The unary path is exercised throughout ``test_qgrpc_source.py``; this
module covers the two streaming dispatch modes and the bidi session's
correlation machinery.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from qr_sampler.exceptions import EntropyUnavailableError
from tests.test_entropy.qgrpc_util import (
    encode_mock_response,
    make_bidi_source,
    make_mocked_source,
)

pytest.importorskip("grpc.aio", reason="grpcio not installed")


class TestServerStreaming:
    """Tests for server_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in server_streaming mode."""
        # channel.stream_stream() -> a callable producing a stream call
        # object with .read() and .cancel().
        mock_stream_call = AsyncMock()
        mock_stream_call.read = AsyncMock(return_value=encode_mock_response(b"\x55" * 50))
        mock_stream_call.cancel = MagicMock()
        mock_stream_handle = MagicMock(return_value=mock_stream_call)

        source, _ = make_mocked_source(
            stream_handle=mock_stream_handle, grpc_mode="server_streaming"
        )
        yield source
        source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Server streaming should return data from the stream."""
        data = source.get_random_bytes(50)
        assert len(data) == 50
        assert data == b"\x55" * 50

    def test_stream_end_raises(self) -> None:
        """If the stream ends unexpectedly (read returns None), should raise."""
        mock_stream_call = AsyncMock()
        mock_stream_call.read = AsyncMock(return_value=None)
        mock_stream_call.cancel = MagicMock()
        mock_stream_handle = MagicMock(return_value=mock_stream_call)

        source, _ = make_mocked_source(
            stream_handle=mock_stream_handle,
            grpc_mode="server_streaming",
            grpc_retry_count=0,
        )
        try:
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)
        finally:
            source.close()


class TestBidiStreaming:
    """Tests for bidi_streaming transport mode."""

    @pytest.fixture()
    def source(self) -> Any:
        """Create a QuantumGrpcSource in bidi_streaming mode."""
        source, mock_stream_handle = make_bidi_source(b"\xcc" * 64)
        source._mock_stream_handle = mock_stream_handle  # type: ignore[attr-defined]
        yield source
        source.close()

    def test_fetch_returns_correct_bytes(self, source: Any) -> None:
        """Bidi streaming should return data from the persistent stream."""
        data = source.get_random_bytes(64)
        assert len(data) == 64
        assert data == b"\xcc" * 64

    def test_stream_reuses_call(self, source: Any) -> None:
        """Bidi streaming should reuse the same call object."""
        source.get_random_bytes(64)
        source.get_random_bytes(64)
        # The stream handle should only be invoked once (session reused).
        assert source._mock_stream_handle.call_count == 1

    def test_concurrent_prefetches_correlate_by_nonce(self, source: Any) -> None:
        """Two in-flight prefetches must each get their own response."""
        t1 = source.prefetch(64, nonce=1111)
        t2 = source.prefetch(64, nonce=2222)
        assert t1 is not None and t2 is not None
        # Redeem out of order: correlation must not mix them up.
        data2 = source.get_random_bytes_with_ticket(64, t2)
        data1 = source.get_random_bytes_with_ticket(64, t1)
        assert data1 == b"\xcc" * 64
        assert data2 == b"\xcc" * 64
        assert t1.echo_verified is True
        assert t2.echo_verified is True

    def test_bidi_stream_end_resets(self) -> None:
        """If bidi stream ends (read returns None), the session resets."""
        source, _ = make_bidi_source(None, grpc_retry_count=0)
        try:
            with pytest.raises(EntropyUnavailableError):
                source.get_random_bytes(10)
            # The bidi session should have been reset to None.
            assert source._transport._bidi_session is None
        finally:
            source.close()
