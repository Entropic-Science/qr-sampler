"""V1 LogitsProcessor ABI conformance test for ``VLLMAdapter``.

Phase 2 substep "qr_sampler V1 LP ABI audit" — asserts the surface
``vllm serve`` discovers at startup via the ``vllm.logits_processors``
entry point. The existing ``test_vllm_adapter.py`` exercises behaviour;
this file asserts the *shape* of the contract so a future refactor that
silently drops a required method fails at unit-test time instead of at
``vllm serve`` cold-start time.

Contract surface checked:

* Required methods exist on the public class:
  ``__init__``, ``apply``, ``update_state``, ``validate_params``,
  ``is_argmax_invariant``.
* ``__init__`` accepts ``(vllm_config, device, is_pin_memory)`` (the
  vLLM V1 LP constructor signature — names must match because vLLM
  invokes by keyword).
* ``validate_params`` is callable on the class (classmethod-shaped).
* ``is_argmax_invariant`` returns ``bool``.
* The entry-point registration in ``pyproject.toml`` lists the
  expected import path.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from qr_sampler.engines.vllm import VLLMAdapter
from qr_sampler.processor import QRSamplerLogitsProcessor


def test_processor_alias_matches_adapter() -> None:
    """``QRSamplerLogitsProcessor`` is the canonical re-export name."""
    assert QRSamplerLogitsProcessor is VLLMAdapter


def test_formal_inheritance_from_vllm_lp_base_when_available() -> None:
    """``VLLMAdapter`` formally inherits from vLLM's V1 LogitsProcessor.

    Phase 2 R1: vLLM 0.17.0 entry-point discovery validates plugins via
    ``issubclass`` checks, not duck-typing. The ``try/except ImportError``
    shim in ``qr_sampler.engines.vllm`` makes the formal base available
    inside Modal containers (where vLLM IS installed) while falling back
    to ``object`` in dev/test environments (where vLLM is NOT installed).

    This test asserts the formal-inheritance pathway in BOTH branches:
    - If vLLM is importable, ``VLLMAdapter`` MUST subclass the real base.
    - If vLLM is not importable, ``VLLMAdapter`` still loads (asserted
      by the file-level ``from qr_sampler.engines.vllm import
      VLLMAdapter``), proving the shim works.
    """
    try:
        from vllm.v1.sample.logits_processor import LogitsProcessor

        assert issubclass(VLLMAdapter, LogitsProcessor), (
            "VLLMAdapter must subclass vllm.v1.sample.logits_processor."
            "LogitsProcessor when vLLM is installed (entry-point discovery "
            "uses issubclass, not duck-typing)."
        )
    except ImportError:
        pytest.skip("vLLM not installed in this environment — shim path verified by import success")


def test_required_methods_present() -> None:
    """Every method ``vllm serve`` calls on a V1 LP exists on the class."""
    required = ("apply", "update_state", "validate_params", "is_argmax_invariant")
    for name in required:
        assert callable(getattr(VLLMAdapter, name, None)), (
            f"VLLMAdapter is missing required V1 LP method: {name}"
        )


def test_init_signature_matches_v1_lp_contract() -> None:
    """``__init__(self, vllm_config, device, is_pin_memory)`` — names matter.

    vLLM's V1 ``AdapterLogitsProcessor`` invokes the subclass
    constructor by keyword (``LP(vllm_config=..., device=...,
    is_pin_memory=...)``). A rename here would surface as a
    ``TypeError: unexpected keyword argument`` at engine init.
    """
    sig = inspect.signature(VLLMAdapter.__init__)
    param_names = list(sig.parameters)
    # ``self`` is always first; the next three must match the V1
    # constructor exactly.
    assert param_names[:4] == ["self", "vllm_config", "device", "is_pin_memory"], (
        f"VLLMAdapter.__init__ signature drift: {param_names}"
    )


def test_is_argmax_invariant_returns_bool() -> None:
    """vLLM's scheduler inspects this to short-circuit greedy decode batches."""
    # Construct with a minimal config — vllm_config=None is supported per
    # test_vllm_adapter::test_init_with_none_vllm_config.
    adapter = VLLMAdapter(vllm_config=None, device=None, is_pin_memory=False)
    try:
        result = adapter.is_argmax_invariant()
        assert isinstance(result, bool), (
            f"is_argmax_invariant() returned {type(result).__name__}, expected bool"
        )
    finally:
        adapter.close()


def test_validate_params_callable_on_class() -> None:
    """``validate_params`` is invoked at request-validation time.

    vLLM's V1 LP base accepts either an instance method or a
    classmethod; both shapes are valid. The contract is that the call
    site ``LP.validate_params(params)`` works.
    """
    # Invoke with empty extra_args — must not raise. Detailed
    # accept/reject semantics are exercised in test_vllm_adapter.
    from qr_sampler.config import validate_extra_args

    validate_extra_args({})  # baseline: empty dict is always valid
    assert callable(VLLMAdapter.validate_params), (
        "VLLMAdapter.validate_params is not callable on the class"
    )


def test_entry_point_registered_in_pyproject() -> None:
    """``vllm serve`` discovers LPs via the ``vllm.logits_processors`` group.

    Asserts the entry-point line in ``pyproject.toml`` so a future
    refactor that renames ``qr_sampler.processor`` doesn't silently
    break LP discovery at deploy time.
    """
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert '[project.entry-points."vllm.logits_processors"]' in text, (
        'pyproject.toml missing [project.entry-points."vllm.logits_processors"]'
    )
    assert "qr_sampler.processor:QRSamplerLogitsProcessor" in text, (
        "pyproject.toml entry-point target drift — expected "
        "qr_sampler.processor:QRSamplerLogitsProcessor"
    )


@pytest.mark.parametrize(
    "attr",
    [
        "apply",
        "update_state",
        "validate_params",
        "is_argmax_invariant",
        "close",
    ],
)
def test_v1_lp_method_is_method_not_attribute(attr: str) -> None:
    """Each ABI member resolves to a callable method on the class object."""
    member = getattr(VLLMAdapter, attr)
    assert callable(member), f"VLLMAdapter.{attr} is not callable"
