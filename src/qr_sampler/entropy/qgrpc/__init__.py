"""Quantum-gRPC entropy source package (``qgrpc`` — cannot shadow ``grpc``).

Decomposition of the former monolithic ``entropy/quantum.py``:

- :mod:`~qr_sampler.entropy.qgrpc.channel` — background loop + channel
  lifecycle.
- :mod:`~qr_sampler.entropy.qgrpc.transport` — wire codec + unary /
  server-streaming / bidi dispatch.
- :mod:`~qr_sampler.entropy.qgrpc.breaker` — pure adaptive-P99 circuit
  breaker.
- :mod:`~qr_sampler.entropy.qgrpc.preprobe` — TCP-connect fast-fail probe.
- :mod:`~qr_sampler.entropy.qgrpc.source` — the ``QuantumGrpcSource``
  facade composing the above (plus ``PrefetchTicket``).
"""

from qr_sampler.entropy.qgrpc.source import PrefetchTicket, QuantumGrpcSource

__all__ = [
    "PrefetchTicket",
    "QuantumGrpcSource",
]
