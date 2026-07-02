"""vLLM engine adapter package.

``qr_sampler.engines.vllm:VLLMAdapter`` is the target of both shipped
entry points (``vllm.logits_processors`` and
``qr_sampler.engine_adapters``) — this re-export keeps those strings
valid across the package's internal layout:

- :mod:`~qr_sampler.engines.vllm.adapter` — ``VLLMAdapter`` (the vLLM V1
  LogitsProcessor contract: batch state, tensor conversion, one-hot
  forcing).
- :mod:`~qr_sampler.engines.vllm.telemetry` — rolling per-stage
  sampling-cost aggregation.
"""

from qr_sampler.engines.vllm.adapter import VLLMAdapter

__all__ = [
    "VLLMAdapter",
]
