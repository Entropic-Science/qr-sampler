"""Import-hygiene guard: ``import qr_sampler`` is 100% side-effect-free.

Asserts that importing ``qr_sampler`` and ``qr_sampler.engines.vllm``
(the ``vllm.logits_processors`` entry-point target that vLLM eagerly
imports during plugin discovery):

1. does NOT open any sockets attributable to qr_sampler's own code, and
2. does NOT monkey-patch anything on the ``vllm`` package.

Why this matters
----------------
A gRPC channel opened at module-import time breaks any host that
snapshots/forks the process (the socket survives only as a dangling fd),
and it makes ``import qr_sampler`` block on the network.
``QuantumGrpcSource`` therefore only opens its channel lazily inside the
first per-token fetch. This test pins that invariant so a future
regression — e.g. someone moving channel init back to ``__init__``
"for warm-up" — fails at unit-test time.

The no-monkey-patch half pins the removal of the legacy Modal-era import
side effects (the ``processor.py`` → ``vllm_patches`` chain and the
mm-probe patch call at the bottom of ``engines/vllm.py``): plugin
discovery must import our modules without mutating vLLM's.

Implementation notes
--------------------
* We don't replace ``socket.socket`` (the stdlib's ``ssl`` module
  subclasses it at module-import time, so a replacement triggers a
  cascade of TypeErrors). Instead we monkey-patch
  ``socket.socket.__init__`` to count calls.
* Pre-warm asyncio + ssl BEFORE installing the counter so their
  import-time socket creations aren't attributed to us.
* For the patch guard we plant a minimal fake ``vllm`` module tree in
  ``sys.modules`` (vLLM is not installed in dev/test), snapshot every
  fake module's ``__dict__``, force a fresh qr_sampler import, and
  assert the snapshots are unchanged.
"""

from __future__ import annotations

import contextlib
import importlib
import socket
import sys
import types
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


def _is_qr_sampler(name: str) -> bool:
    return name == "qr_sampler" or name.startswith("qr_sampler.")


@pytest.fixture
def _fresh_qr_sampler() -> Iterator[None]:
    """Pop all ``qr_sampler*`` modules so imports re-run their side effects.

    Stashes any pre-existing ``qr_sampler*`` modules and restores them
    after the test, so a re-import here does NOT invalidate class
    identities (e.g. ``VLLMAdapter``) cached by tests that ran earlier
    in the same pytest session.
    """
    stashed = {name: mod for name, mod in sys.modules.items() if _is_qr_sampler(name)}
    for name in list(stashed):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in [n for n in list(sys.modules) if _is_qr_sampler(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(stashed)


@pytest.fixture
def _socket_call_counter(_fresh_qr_sampler: None) -> Iterator[list[tuple[object, ...]]]:
    """Yield a list capturing every ``socket.socket.__init__`` invocation
    while qr_sampler is (re-)imported."""
    import asyncio  # noqa: F401 — pre-warm stdlib
    import ssl  # noqa: F401

    with contextlib.suppress(ImportError):
        import httpx  # noqa: F401  # transitively imports urllib3 etc.
    with contextlib.suppress(ImportError):
        import urllib3  # noqa: F401  # _has_ipv6 AF_INET6 capability probe

    calls: list[tuple[object, ...]] = []
    orig_init = socket.socket.__init__

    def _wrapper(self: socket.socket, *args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))
        orig_init(self, *args, **kwargs)

    socket.socket.__init__ = _wrapper  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        socket.socket.__init__ = orig_init  # type: ignore[method-assign]


def test_vllm_adapter_import_opens_no_sockets(
    _socket_call_counter: list[tuple[object, ...]],
) -> None:
    """``import qr_sampler.engines.vllm`` MUST NOT open any new sockets
    attributable to qr_sampler.

    A non-zero socket count here indicates either:
    1. A regression in ``QuantumGrpcSource`` (gRPC channel opened at
       ``__init__``/import time), or
    2. A new third-party dep dragged in at module-import time that opens
       sockets and wasn't pre-warmed in the fixture above.

    If (2), pre-warm the offending lib in the fixture and document why
    in a comment. NEVER suppress this assertion by raising the
    threshold — it's the only line of defence against an import-time
    socket leak.
    """
    importlib.import_module("qr_sampler.engines.vllm")

    assert _socket_call_counter == [], (
        f"qr_sampler.engines.vllm import opened a socket. Calls: {_socket_call_counter}."
    )


def test_top_level_import_opens_no_sockets(
    _socket_call_counter: list[tuple[object, ...]],
) -> None:
    """Same invariant for ``import qr_sampler`` — the package root must be
    equally inert (no sockets, no file writes, no patches)."""
    importlib.import_module("qr_sampler")

    assert _socket_call_counter == [], (
        f"qr_sampler import opened a socket. Calls: {_socket_call_counter}."
    )


# ---------------------------------------------------------------------------
# No-monkey-patch guard
# ---------------------------------------------------------------------------

_FAKE_VLLM_MODULES = (
    "vllm",
    "vllm.v1",
    "vllm.v1.sample",
    "vllm.v1.sample.logits_processor",
    "vllm.v1.worker",
    "vllm.v1.worker.gpu_model_runner",
)


@pytest.fixture
def _fake_vllm_tree(_fresh_qr_sampler: None) -> Iterator[dict[str, types.ModuleType]]:
    """Plant a minimal fake ``vllm`` module tree in ``sys.modules``.

    vLLM is not installed in the dev/test environment, so without this
    the ``from vllm.v1.sample.logits_processor import LogitsProcessor``
    shim in ``engines/vllm.py`` falls back to ``object`` and any
    would-be patch code silently no-ops — leaving nothing to observe.
    Planting the tree makes an import-time mutation of vLLM visible.
    """
    stashed = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "vllm" or name.startswith("vllm.")
    }
    for name in list(stashed):
        sys.modules.pop(name, None)

    fakes: dict[str, types.ModuleType] = {}
    for name in _FAKE_VLLM_MODULES:
        mod = types.ModuleType(name)
        fakes[name] = mod
        sys.modules[name] = mod
        parent_name, _, child = name.rpartition(".")
        if parent_name:
            setattr(fakes[parent_name], child, mod)

    class LogitsProcessor:  # minimal stand-in for the V1 LP base
        pass

    class GPUModelRunner:  # historical monkey-patch target
        def init_fp8_kv_scales(self) -> None:
            pass

    lp_mod = fakes["vllm.v1.sample.logits_processor"]
    lp_mod.LogitsProcessor = LogitsProcessor  # type: ignore[attr-defined]
    runner_mod = fakes["vllm.v1.worker.gpu_model_runner"]
    runner_mod.GPUModelRunner = GPUModelRunner  # type: ignore[attr-defined]

    try:
        yield fakes
    finally:
        for name in _FAKE_VLLM_MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(stashed)


def test_import_applies_no_vllm_monkey_patches(
    _fake_vllm_tree: dict[str, types.ModuleType],
) -> None:
    """``import qr_sampler`` + ``import qr_sampler.engines.vllm`` must not
    mutate ANY attribute of the ``vllm`` package.

    Pins the removal of the Modal-era import side effects: the
    ``processor.py`` → ``vllm_patches`` chain (patched
    ``GPUModelRunner.init_fp8_kv_scales``) and the mm-probe patch call
    that used to run at the bottom of ``engines/vllm.py``.
    """
    before = {name: dict(vars(mod)) for name, mod in _fake_vllm_tree.items()}
    runner_mod = _fake_vllm_tree["vllm.v1.worker.gpu_model_runner"]
    runner_cls = runner_mod.GPUModelRunner  # type: ignore[attr-defined]
    init_fp8_before = runner_cls.init_fp8_kv_scales

    importlib.import_module("qr_sampler")
    importlib.import_module("qr_sampler.engines.vllm")

    for name, mod in _fake_vllm_tree.items():
        after = dict(vars(mod))
        assert after.keys() == before[name].keys(), (
            f"import qr_sampler added/removed attributes on {name}: "
            f"{set(after) ^ set(before[name])}"
        )
        changed = {key for key in after if after[key] is not before[name][key]}
        assert not changed, f"import qr_sampler rebound attributes on {name}: {changed}"

    assert runner_cls.init_fp8_kv_scales is init_fp8_before, (
        "import qr_sampler monkey-patched GPUModelRunner.init_fp8_kv_scales"
    )
