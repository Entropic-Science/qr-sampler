"""Backward-compatible re-export of QRSamplerLogitsProcessor.

The implementation has moved to ``qr_sampler.engines.vllm.VLLMAdapter``.
This module re-exports ``VLLMAdapter`` under the original name to preserve
the ``vllm.logits_processors`` entry point and any direct imports.

Side-effect: importing this module also imports
``qr_sampler.connectors.modal.vllm_patches``, which applies vLLM monkey-
patches on import. This entry point is loaded by vLLM inside EngineCore_DP0
during engine init — i.e. in the same Python process that later runs
``GPUModelRunner.init_fp8_kv_scales``, and BEFORE Modal's snapshot is
taken — so any class-method patch installed here is part of the
snapshotted heap and survives restore. See vllm_patches.py docstring for
the active patches and the conditions under which to remove them.
"""

from qr_sampler.connectors.modal import vllm_patches  # noqa: F401  -- import for side effect
from qr_sampler.engines.vllm import VLLMAdapter as QRSamplerLogitsProcessor

__all__ = ["QRSamplerLogitsProcessor"]
