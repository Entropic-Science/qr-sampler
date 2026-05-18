"""Unit tests for the rolling-secret bearer verifier in ``vllm_serve.py``.

Acceptance (plan §"qr-sampler — Modal app.py cleanup + rolling-secret bearer
verifier"):

- ``_verify_bearer`` accepts the first, last, and middle entry from a
  3-secret vector (signer-uses-first / verifier-accepts-any rotation).
- Rejects an unknown token.
- Rejects the empty string.
- Constant-time: the implementation uses ``hmac.compare_digest`` rather than
  a Python ``==`` short-circuit. Verified by source inspection — the per-
  request branch count does not depend on which entry matched (the loop runs
  to completion even after a match).

The module under test imports FastAPI, which is available in the modal
connector toolchain (the GPU container ships it). When the dev environment
lacks fastapi, these tests skip cleanly rather than fail-load.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

_vllm_serve = pytest.importorskip(
    "qr_sampler.connectors.modal.vllm_serve",
    reason="fastapi not installed in this environment",
)


@pytest.fixture
def three_entry_vector(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Three randomly chosen, distinguishable secrets under SERVICE_TOKEN_SECRETS."""
    secrets = ["first-secret-AAA", "middle-secret-BBB", "last-secret-CCC"]
    monkeypatch.setenv("SERVICE_TOKEN_SECRETS", ",".join(secrets))
    importlib.reload(_vllm_serve)
    return secrets


@pytest.fixture
def empty_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERVICE_TOKEN_SECRETS", raising=False)
    importlib.reload(_vllm_serve)


def test_accepts_first_entry(three_entry_vector: list[str]) -> None:
    assert _vllm_serve._verify_bearer(three_entry_vector[0]) is True


def test_accepts_middle_entry(three_entry_vector: list[str]) -> None:
    assert _vllm_serve._verify_bearer(three_entry_vector[1]) is True


def test_accepts_last_entry(three_entry_vector: list[str]) -> None:
    assert _vllm_serve._verify_bearer(three_entry_vector[2]) is True


def test_rejects_unknown_token(three_entry_vector: list[str]) -> None:
    assert _vllm_serve._verify_bearer("not-in-the-vector") is False


def test_rejects_empty_string(three_entry_vector: list[str]) -> None:
    assert _vllm_serve._verify_bearer("") is False


def test_rejects_when_vector_empty(empty_vector: None) -> None:
    """Closed-by-default: no secret provisioned → every token rejected."""
    assert _vllm_serve._verify_bearer("anything") is False


def test_vector_drops_blank_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Extra commas / whitespace must not become accepted empty secrets."""
    monkeypatch.setenv("SERVICE_TOKEN_SECRETS", " good-token , , ,, ")
    importlib.reload(_vllm_serve)
    assert _vllm_serve._accepted_bearer_secrets() == ["good-token"]
    assert _vllm_serve._verify_bearer("good-token") is True
    assert _vllm_serve._verify_bearer("") is False
    assert _vllm_serve._verify_bearer(" ") is False


def test_uses_compare_digest_not_equality() -> None:
    """Constant-time smoke check: source must call hmac.compare_digest.

    We assert on the implementation, not on timing — the latter is flaky in
    CI. compare_digest's contract guarantees branch-count parity for
    equal-length inputs, which is the property the plan calls for.
    """
    source = inspect.getsource(_vllm_serve._verify_bearer)
    assert "compare_digest" in source, (
        "_verify_bearer must use hmac.compare_digest for constant-time compare"
    )
    assert "==" not in source.replace("!=", ""), (
        "_verify_bearer must not fall back to == on the bearer token"
    )


class _StubRequest:
    """Minimal FastAPI Request shim — only ``headers.get`` is exercised."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def test_gate_fails_closed_when_vector_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty SERVICE_TOKEN_SECRETS with no opt-in must reject with 503.

    Regression guard for the previous open-by-default behavior. The hardening
    is: a deploy that forgets to provision a secret should be rejected loudly
    (503) rather than silently expose the GPU endpoint.
    """
    from fastapi import HTTPException

    monkeypatch.delenv("SERVICE_TOKEN_SECRETS", raising=False)
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_INFERENCE", raising=False)
    importlib.reload(_vllm_serve)

    with pytest.raises(HTTPException) as excinfo:
        _vllm_serve._check_vllm_api_key(_StubRequest(headers={"authorization": "Bearer x"}))
    assert excinfo.value.status_code == 503


def test_gate_allows_unauth_with_explicit_optin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALLOW_UNAUTHENTICATED_INFERENCE=1 is the documented escape hatch for
    smoke-tests / local dev. Any other value (including '0', 'true', empty)
    must still fail-closed — only the literal '1' opts in."""
    monkeypatch.delenv("SERVICE_TOKEN_SECRETS", raising=False)
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED_INFERENCE", "1")
    importlib.reload(_vllm_serve)

    _vllm_serve._check_vllm_api_key(_StubRequest())  # must not raise


def test_gate_optin_rejects_non_one_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Guard against 'truthy' accidents: only '1' opts in, not 'true' / '0' / ''."""
    from fastapi import HTTPException

    monkeypatch.delenv("SERVICE_TOKEN_SECRETS", raising=False)
    for accidental in ("0", "true", "TRUE", "yes", ""):
        monkeypatch.setenv("ALLOW_UNAUTHENTICATED_INFERENCE", accidental)
        importlib.reload(_vllm_serve)
        with pytest.raises(HTTPException) as excinfo:
            _vllm_serve._check_vllm_api_key(_StubRequest())
        assert excinfo.value.status_code == 503, (
            f"ALLOW_UNAUTHENTICATED_INFERENCE={accidental!r} must NOT open the gate"
        )


def test_gate_optin_ignored_when_secrets_provisioned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a secret IS provisioned, the opt-in must not bypass bearer verification.

    Defense-in-depth: a stale ALLOW_UNAUTHENTICATED_INFERENCE=1 left over from
    a smoke test must not silently disable bearer auth in production.
    """
    from fastapi import HTTPException

    monkeypatch.setenv("SERVICE_TOKEN_SECRETS", "real-secret")
    monkeypatch.setenv("ALLOW_UNAUTHENTICATED_INFERENCE", "1")
    importlib.reload(_vllm_serve)

    with pytest.raises(HTTPException) as excinfo:
        _vllm_serve._check_vllm_api_key(_StubRequest())
    assert excinfo.value.status_code == 401


def test_gate_accepts_valid_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVICE_TOKEN_SECRETS", "real-secret")
    monkeypatch.delenv("ALLOW_UNAUTHENTICATED_INFERENCE", raising=False)
    importlib.reload(_vllm_serve)

    _vllm_serve._check_vllm_api_key(
        _StubRequest(headers={"authorization": "Bearer real-secret"})
    )  # must not raise
