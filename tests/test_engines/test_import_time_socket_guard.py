"""Phase 2 R15(a): import-time socket guard for ``qr_sampler.engines.vllm``.

Asserts that ``import qr_sampler.engines.vllm`` (the entry-point target
that vLLM eagerly imports during plugin discovery) does NOT open any
sockets attributable to qr_sampler's own code.

Why this matters
----------------
When Modal's ``enable_memory_snapshot=True`` is active, every file
descriptor open at snap=True time gets frozen into the snapshot. A live
socket in the snapshot survives only as a dangling fd after restore —
the kernel side of the socket is gone but Python still holds a handle,
and the first ``send`` / ``recv`` returns ``EBADF``. The classic case is
a gRPC channel opened at module-import: it survives the snapshot in a
broken state and the first per-token entropy fetch on the post-restore
hot path bombs out.

Phase 2's audit (R8) confirmed ``QuantumGrpcSource._ensure_channel``
sets ``_channel_initialized = False`` at construction and only opens
the gRPC channel inside ``apply()`` (the first per-token call). This
test pins that invariant so any future regression — e.g. someone moving
the channel init back to ``__init__`` "for warm-up" — fails at unit
test time, not at the first snapshot-restored cold start.

Implementation notes
--------------------
* We don't replace ``socket.socket`` (the stdlib's ``ssl`` module
  subclasses it at module-import time, so a replacement triggers a
  cascade of TypeErrors). Instead we monkey-patch
  ``socket.socket.__init__`` to count calls.
* Pre-warm asyncio + ssl + httpx + urllib3 BEFORE installing the
  counter so their import-time socket creations (e.g. urllib3's
  ``_has_ipv6`` AF_INET6 capability probe) aren't attributed to us.
* The guard runs in the same process as the test (no subprocess). Python
  caches modules; if a previous test already imported
  ``qr_sampler.engines.vllm`` we re-import the module to force the
  side effect path to fire fresh — necessary because the
  Phase 2 R3 import-time hook (``_install_mm_probe_skip_patch``) is
  idempotent and silently returns on second invocation.
"""

from __future__ import annotations

import contextlib
import importlib
import socket
import sys

import pytest


@pytest.fixture
def _socket_call_counter() -> list[tuple[object, ...]]:
    """Yield a list that captures every ``socket.socket.__init__`` invocation
    while qr_sampler is (re-)imported.

    The fixture:
    * Pre-warms third-party libraries whose own import-time socket calls
      are NOT attributable to qr_sampler — most notably urllib3's
      ``_has_ipv6`` AF_INET6 capability probe.
    * Stashes any pre-existing ``qr_sampler*`` modules and restores them
      after the test, so a re-import here does NOT invalidate
      ``VLLMAdapter`` references cached by tests that ran earlier in the
      same pytest session.
    """
    import asyncio  # noqa: F401 — pre-warm stdlib
    import ssl  # noqa: F401

    with contextlib.suppress(ImportError):
        import httpx  # noqa: F401  # transitively imports urllib3 etc.
    with contextlib.suppress(ImportError):
        import urllib3  # noqa: F401

    # Stash any already-imported qr_sampler modules so we can restore them
    # post-test. Without this restore, later tests in the same session
    # that imported ``VLLMAdapter`` at module-level top see a *different*
    # class object after our forced re-import — breaking ``is`` checks.
    def _is_qr_sampler(name: str) -> bool:
        return name == "qr_sampler" or name.startswith("qr_sampler.")

    stashed = {name: mod for name, mod in sys.modules.items() if _is_qr_sampler(name)}
    for name in list(stashed):
        sys.modules.pop(name, None)

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
        # Restore the original qr_sampler module objects so other tests
        # see the same class identities they imported.
        for name in [n for n in list(sys.modules) if _is_qr_sampler(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(stashed)


def test_vllm_adapter_import_opens_no_sockets(
    _socket_call_counter: list[tuple[object, ...]],
) -> None:
    """``import qr_sampler.engines.vllm`` MUST NOT open any new sockets
    attributable to qr_sampler.

    A non-zero socket count here indicates either:
    1. A regression in ``QuantumGrpcSource`` (gRPC channel back at
       ``__init__`` time — would break snapshot integrity), or
    2. A new third-party dep dragged in at module-import time that opens
       sockets and wasn't pre-warmed in this test.

    If (2), pre-warm the offending lib in the fixture above and document
    why in a comment. NEVER suppress this assertion by raising the
    threshold — it's the only line of defence against a snapshot-time
    socket leak.
    """
    importlib.import_module("qr_sampler.engines.vllm")

    assert _socket_call_counter == [], (
        "qr_sampler.engines.vllm import opened a socket — Phase 2 R8 "
        f"audit violated. Calls: {_socket_call_counter}. See R8 / R15(a) "
        "in plan.md for context."
    )


def test_vllm_processor_import_opens_no_sockets(
    _socket_call_counter: list[tuple[object, ...]],
) -> None:
    """Same invariant for ``qr_sampler.processor`` — the actual import
    path vLLM's entry-point loader traverses (qr_sampler.processor →
    qr_sampler.engines.vllm)."""
    importlib.import_module("qr_sampler.processor")

    assert _socket_call_counter == [], (
        "qr_sampler.processor import opened a socket — entry-point "
        f"discovery path is leaking. Calls: {_socket_call_counter}."
    )
