"""Hand-written gRPC client stubs for the purity (server-draw) service.

Provides ``PurityServiceStub`` with both unary (``GetDraw``) and
bidirectional streaming (``StreamDraws``) RPCs. These stubs are
compatible with both sync and async (``grpc.aio``) channels.

If the proto definition changes, update these stubs or regenerate with
``grpc_tools.protoc``.
"""

from __future__ import annotations

from typing import Any


def _draw_request_serializer(request: Any) -> bytes:
    """Serialize a DrawRequest to bytes."""
    result: bytes = request.SerializeToString()
    return result


def _draw_response_deserializer(data: bytes) -> Any:
    """Deserialize bytes to a DrawResponse."""
    from qr_sampler.proto.purity_service_pb2 import DrawResponse

    return DrawResponse.FromString(data)


def _draw_request_deserializer(data: bytes) -> Any:
    """Deserialize bytes to a DrawRequest."""
    from qr_sampler.proto.purity_service_pb2 import DrawRequest

    return DrawRequest.FromString(data)


def _draw_response_serializer(response: Any) -> bytes:
    """Serialize a DrawResponse to bytes."""
    result: bytes = response.SerializeToString()
    return result


class PurityServiceStub:
    """gRPC client stub for the PurityService.

    Supports both sync and async (``grpc.aio``) channels. Provides two
    RPC methods:

    - ``GetDraw``: unary request-response
    - ``StreamDraws``: bidirectional streaming

    Args:
        channel: A gRPC Channel or async Channel instance.
    """

    def __init__(self, channel: Any) -> None:
        self.GetDraw = channel.unary_unary(
            "/qr_purity.PurityService/GetDraw",
            request_serializer=_draw_request_serializer,
            response_deserializer=_draw_response_deserializer,
        )
        self.StreamDraws = channel.stream_stream(
            "/qr_purity.PurityService/StreamDraws",
            request_serializer=_draw_request_serializer,
            response_deserializer=_draw_response_deserializer,
        )


class PurityServiceServicer:
    """Base class for PurityService server implementations.

    Override the methods in this class to implement the service.
    Used by tests' stub servers and third parties.
    """

    def GetDraw(  # noqa: N802
        self,
        request: Any,
        context: Any,
    ) -> Any:
        """Unary RPC: single request -> single response."""
        import grpc

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StreamDraws(  # noqa: N802
        self,
        request_iterator: Any,
        context: Any,
    ) -> Any:
        """Bidirectional streaming RPC."""
        import grpc

        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_PurityServiceServicer_to_server(  # noqa: N802
    servicer: PurityServiceServicer,
    server: Any,
) -> None:
    """Register a ``PurityServiceServicer`` with a gRPC server.

    Args:
        servicer: The service implementation.
        server: A ``grpc.Server`` or ``grpc.aio.Server`` instance.
    """
    from grpc import stream_stream_rpc_method_handler, unary_unary_rpc_method_handler

    rpc_method_handlers = {
        "GetDraw": unary_unary_rpc_method_handler(
            servicer.GetDraw,
            request_deserializer=_draw_request_deserializer,
            response_serializer=_draw_response_serializer,
        ),
        "StreamDraws": stream_stream_rpc_method_handler(
            servicer.StreamDraws,
            request_deserializer=_draw_request_deserializer,
            response_serializer=_draw_response_serializer,
        ),
    }
    from grpc import method_handlers_generic_handler

    generic_handler = method_handlers_generic_handler(
        "qr_purity.PurityService",
        rpc_method_handlers,
    )
    server.add_generic_rpc_handlers((generic_handler,))
