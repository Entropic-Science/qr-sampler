"""vLLM 0.17.0 monkey-patches for the Modal deployment.

Auto-applied at import time. The canonical loading path is the side-effect
import in ``qr_sampler.processor`` — vLLM discovers that module via the
``vllm.logits_processors`` entry point during engine init (inside
EngineCore_DP0, the GPU worker subprocess where ``GPUModelRunner`` lives),
which fires BEFORE the snapshot is taken and BEFORE ``init_fp8_kv_scales``
is ever called by ``gpu_worker.wake_up()``. The patched class method is
part of the Python heap that Modal snapshots, so it survives restore.

Every patch must:

* be idempotent — re-entering ``install_vllm_patches()`` (e.g. on a future
  ``vllm.general_plugins`` migration, or a hot-reload) detects its marker
  and short-circuits;
* gracefully no-op when vLLM is not installed (qr-sampler local dev / CI),
  so importing this module never raises on hosts without GPU support;
* emit one ``vllm.<patch_name>.patch_installed`` event at install time and
  one ``vllm.<patch_name>.patched`` event on every invocation, so the
  cold-start log carries actionable evidence that the patch is live.

Active patches
--------------

``_install_fp8_kv_scales_patch``
    Handles Qwen3 GDN's ``List[Tensor]`` KV cache layout in
    ``vllm.v1.worker.gpu_model_runner.GPUModelRunner.init_fp8_kv_scales``.

    Upstream body (vllm/v1/worker/gpu_model_runner.py:783-825 in v0.17.0)
    iterates ``self.kv_caches`` and calls ``cache_tensor.zero_()`` on each
    entry, assuming flat ``torch.Tensor``. Qwen3 GDN stores some entries
    as ``List[Tensor]`` (one tensor per attention head group) because the
    Gated Delta Network layers have a different KV layout than the
    standard transformer blocks. ``.zero_()`` raises
    ``AttributeError: 'list' object has no attribute 'zero_'`` on the
    list branch.

    Captured in Phase 3 iter-02 (artifacts/iter-02.log:1248-1256) as the
    POST /wake_up 500 traceback after Modal GPU snapshot restore. The
    patch dispatches on entry type — ``None``, ``Tensor``, or ``list`` —
    and zeroes the contained tensors when the entry is a list. The
    scale-init logic at upstream lines 800-825 is unaffected; we
    reproduce it verbatim rather than falling through to the original
    (which would re-execute the buggy KV-zeroing loop and re-raise).

When to remove
--------------

Drop this whole module — plus the side-effect import in
``qr_sampler/processor.py`` — once vLLM ships a release whose
``init_fp8_kv_scales`` dispatches on KV entry type upstream. A
``vllm.fp8_kv_scales.patched`` event with ``list_count=0`` for several
consecutive cold-starts is the early signal that upstream's layout flipped
back to a flat tensor (the patch is now no-op but still safe).
"""

from __future__ import annotations

import logging

_log = logging.getLogger("qr_sampler.vllm_patches")


def install_vllm_patches() -> None:
    """Apply every qr-sampler vLLM monkey-patch. Idempotent."""
    _install_fp8_kv_scales_patch()


def _install_fp8_kv_scales_patch() -> None:
    """Replace ``GPUModelRunner.init_fp8_kv_scales`` with a List[Tensor]-aware version.

    The replacement reproduces the upstream body (vllm/v1/worker/
    gpu_model_runner.py:783-825 in v0.17.0) verbatim except for the
    KV-zeroing loop, where it dispatches on entry type. Counts of each
    layout (tensor / list / none / unknown) are logged on every call as a
    ``vllm.fp8_kv_scales.patched`` event so a future upstream layout
    shift is immediately greppable.

    Patch is dormant when ``--kv-cache-dtype fp8`` is not set — upstream's
    own ``cache_dtype.startswith("fp8")`` gate (line 792) is preserved.
    """
    try:
        import vllm.v1.worker.gpu_model_runner as _gmr
    except ImportError:
        _log.debug(
            "vllm.v1.worker.gpu_model_runner not importable; fp8_kv_scales patch skipped",
            extra={
                "event": "vllm.fp8_kv_scales.patch_skipped",
                "reason": "vllm_missing",
            },
        )
        return

    runner_cls = _gmr.GPUModelRunner
    if not hasattr(runner_cls, "init_fp8_kv_scales"):
        _log.warning(
            "GPUModelRunner.init_fp8_kv_scales not present in this vLLM build; "
            "patch lost its target — re-read upstream and pick a new patch point",
            extra={
                "event": "vllm.fp8_kv_scales.patch_skipped",
                "reason": "method_missing",
            },
        )
        return

    if getattr(runner_cls.init_fp8_kv_scales, "_qr_patched", False):
        return

    import torch

    @torch.inference_mode()
    def _patched_init_fp8_kv_scales(self: object) -> None:
        from vllm.model_executor.layers.attention import Attention, MLAAttention

        log = logging.getLogger("qr_sampler.vllm_patches.fp8_kv_scales")

        cache_dtype = getattr(getattr(self, "cache_config", None), "cache_dtype", "")
        if not cache_dtype.startswith("fp8"):
            return

        kv_caches = getattr(self, "kv_caches", [])

        tensor_count = 0
        list_count = 0
        none_count = 0
        unknown_count = 0
        for cache_tensor in kv_caches:
            if cache_tensor is None:
                none_count += 1
            elif isinstance(cache_tensor, torch.Tensor):
                cache_tensor.zero_()
                tensor_count += 1
            elif isinstance(cache_tensor, list):
                list_count += 1
                for sub in cache_tensor:
                    if isinstance(sub, torch.Tensor):
                        sub.zero_()
            else:
                unknown_count += 1

        k_attr_names = ("_k_scale", "k_scale")
        v_attr_names = ("_v_scale", "v_scale")
        attn_layers = self.compilation_config.static_forward_context  # type: ignore[attr-defined]
        for _name, module in attn_layers.items():
            if isinstance(module, (Attention, MLAAttention)):
                k_scale_val, v_scale_val = 1.0, 1.0
                for attr in k_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(k_scale_val)
                for attr in v_attr_names:
                    if hasattr(module, attr):
                        param = getattr(module, attr)
                        if isinstance(param, torch.Tensor):
                            param.fill_(v_scale_val)

        log.info(
            "fp8_kv_scales patched ran: tensor=%d list=%d none=%d unknown=%d cache_dtype=%s",
            tensor_count,
            list_count,
            none_count,
            unknown_count,
            cache_dtype,
            extra={
                "event": "vllm.fp8_kv_scales.patched",
                "tensor_count": tensor_count,
                "list_count": list_count,
                "none_count": none_count,
                "unknown_count": unknown_count,
                "cache_dtype": cache_dtype,
            },
        )

    _patched_init_fp8_kv_scales._qr_patched = True  # type: ignore[attr-defined]
    runner_cls.init_fp8_kv_scales = _patched_init_fp8_kv_scales
    _log.info(
        "vllm fp8_kv_scales patch installed on GPUModelRunner",
        extra={"event": "vllm.fp8_kv_scales.patch_installed"},
    )


# Side-effect: install patches at module import. Any failure here (partial
# vLLM install, unexpected upstream signature change) MUST NOT break the
# enclosing logits-processor discovery — we catch and log instead.
# Mirrors qr_sampler/engines/vllm.py:731-743 ("guarded by try/except so
# non-Modal contexts still import this module cleanly").
try:
    install_vllm_patches()
except Exception as _err:  # noqa: BLE001 -- defence-in-depth catch is intentional
    _log.warning(
        "vllm_patches install raised %s: %s — LP discovery proceeding without patches",
        type(_err).__name__,
        _err,
        extra={
            "event": "vllm.patches.install_failed",
            "error_type": type(_err).__name__,
            "error_msg": str(_err),
        },
    )
