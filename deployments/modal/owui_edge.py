"""Auth-proxy FastAPI app in front of Open WebUI on Modal.

Sits between the public ``chat.entropic.science`` hostname and the internal
``owui`` container. For every incoming request:

1. Reads the ``entropic_session`` cookie (Domain=.entropic.science).
2. Validates it by calling ``entropic.science/api/account/me`` with the
   cookie forwarded (cached for 30 s, keyed on a sha-256 of the cookie value).
3. On success, attaches three trusted headers and proxies to the upstream
   OWUI container:
       X-Trusted-Email
       X-Trusted-Account-Id
       X-Trusted-Display-Name
4. On failure, returns ``302 → entropic.science/account/sign-in?next=<original>``.

OWUI is configured with ``WEBUI_AUTH=true``,
``WEBUI_AUTH_TRUSTED_EMAIL_HEADER=X-Trusted-Email``, and
``WEBUI_AUTH_TRUSTED_NAME_HEADER=X-Trusted-Display-Name``, so it auto-provisions
its per-user record from the headers we inject and never sees the raw cookie.

The trust boundary
------------------
This proxy is the *only* thing that talks to OWUI; OWUI does not expose its
own ``WEBUI_AUTH=false`` surface. So the trusted-header injection is safe
because no external caller can reach OWUI's port directly. If the deployment
topology changes to expose OWUI publicly, the headers must be stripped from
inbound requests in the OWUI image's own ingress (e.g. a nginx sidecar) —
the OWUI base image does not strip them on its own.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse, StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger("qr_sampler.modal.owui_edge")

# Cookie + endpoint contract — kept in sync with entropic.science/lib/sessionAuth.ts.
_SESSION_COOKIE = "entropic_session"
_WHOAMI_PATH = "/account/me"

# The trusted-header names OWUI expects (matches WEBUI_AUTH_TRUSTED_* env vars).
_TRUSTED_EMAIL_HEADER = "X-Trusted-Email"
_TRUSTED_ACCOUNT_HEADER = "X-Trusted-Account-Id"
_TRUSTED_NAME_HEADER = "X-Trusted-Display-Name"

# Request headers we never forward to the upstream OWUI container. The trusted
# headers go in `_TRUSTED_*` so a malicious upstream caller cannot inject them.
_HOP_BY_HOP_REQUEST_HEADERS = frozenset(
    {
        "host",
        "connection",
        "transfer-encoding",
        "upgrade",
        "te",
        "trailers",
        "proxy-authorization",
        "proxy-authenticate",
        "expect",
        # Strip any inbound claim — only the proxy may set these.
        _TRUSTED_EMAIL_HEADER.lower(),
        _TRUSTED_ACCOUNT_HEADER.lower(),
        _TRUSTED_NAME_HEADER.lower(),
    }
)
_HOP_BY_HOP_RESPONSE_HEADERS = frozenset(
    {
        "connection",
        "transfer-encoding",
        "upgrade",
        "keep-alive",
        "proxy-authenticate",
        "trailers",
        "te",
    }
)


class _SessionCache:
    """30s TTL cache from sha-256(cookie_value) -> (account-dict or None)."""

    def __init__(self, ttl_seconds: float = 30.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, dict[str, Any] | None]] = {}

    def _key(self, cookie_value: str) -> str:
        return hashlib.sha256(cookie_value.encode("utf-8")).hexdigest()

    def get(self, cookie_value: str) -> tuple[bool, dict[str, Any] | None]:
        entry = self._entries.get(self._key(cookie_value))
        if entry is None:
            return False, None
        deadline, account = entry
        if time.monotonic() > deadline:
            self._entries.pop(self._key(cookie_value), None)
            return False, None
        return True, account

    def set(self, cookie_value: str, account: dict[str, Any] | None) -> None:
        self._entries[self._key(cookie_value)] = (
            time.monotonic() + self._ttl,
            account,
        )


def _sign_in_redirect(request: Request) -> RedirectResponse:
    api_base_url = os.environ["ENTROPIC_API_BASE_URL"]
    site_origin = api_base_url.removesuffix("/api").rstrip("/")
    if not site_origin:
        site_origin = "https://entropic.science"
    original = str(request.url)
    return RedirectResponse(
        url=f"{site_origin}/account/sign-in?next={quote(original, safe='')}",
        status_code=302,
    )


async def _whoami(
    client: httpx.AsyncClient,
    cookie_value: str,
) -> dict[str, Any] | None:
    api_base_url = os.environ["ENTROPIC_API_BASE_URL"].rstrip("/")
    try:
        response = await client.get(
            f"{api_base_url}{_WHOAMI_PATH}",
            headers={"cookie": f"{_SESSION_COOKIE}={cookie_value}"},
            timeout=httpx.Timeout(3.0),
        )
    except httpx.HTTPError as exc:
        logger.warning("whoami call failed: %s", exc)
        return None
    if response.status_code != 200:
        return None
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict) or body.get("status") != "signed-in":
        return None
    account = body.get("account")
    if not isinstance(account, dict):
        return None
    return account


def _forward_request_headers(headers: httpx.Headers, account: dict[str, Any]) -> dict[str, str]:
    """Construct the upstream header set with trusted-user headers attached."""
    out: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in _HOP_BY_HOP_REQUEST_HEADERS:
            continue
        out[key] = value
    email = str(account.get("email", "")).strip()
    account_id = str(account.get("id", "")).strip()
    display_name = str(account.get("displayName") or account.get("display_name") or email).strip()
    if not email or not account_id:
        # Should not reach here — whoami returned a malformed account.
        raise RuntimeError("Validated account is missing email or id")
    out[_TRUSTED_EMAIL_HEADER] = email
    out[_TRUSTED_ACCOUNT_HEADER] = account_id
    out[_TRUSTED_NAME_HEADER] = display_name
    return out


def _scrub_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _HOP_BY_HOP_RESPONSE_HEADERS
    }


def build_app() -> FastAPI:
    cache = _SessionCache()
    upstream_url = os.environ.get("OWUI_UPSTREAM_URL", "http://owui:8080")
    proxy_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    whoami_client = httpx.AsyncClient()

    app = FastAPI(title="owui-edge (entropic.science auth proxy)")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await proxy_client.aclose()
        await whoami_client.aclose()

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(path: str, request: Request) -> Response:
        cookie_value = request.cookies.get(_SESSION_COOKIE)
        if not cookie_value:
            return _sign_in_redirect(request)

        hit, cached = cache.get(cookie_value)
        if hit:
            account = cached
        else:
            account = await _whoami(whoami_client, cookie_value)
            cache.set(cookie_value, account)

        if account is None:
            return _sign_in_redirect(request)

        upstream_path = f"/{path}" if path else "/"
        upstream_target = f"{upstream_url.rstrip('/')}{upstream_path}"

        try:
            upstream_headers = _forward_request_headers(request.headers, account)
        except RuntimeError as exc:
            logger.error("invalid account from whoami: %s", exc)
            return _sign_in_redirect(request)

        body_iter: AsyncIterator[bytes] | None = None
        if request.method not in ("GET", "HEAD"):
            body_iter = request.stream()

        upstream_response = await proxy_client.send(
            proxy_client.build_request(
                method=request.method,
                url=upstream_target,
                params=dict(request.query_params),
                headers=upstream_headers,
                content=body_iter,
            ),
            stream=True,
        )

        async def _body_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream_response.aiter_raw():
                    yield chunk
            finally:
                await upstream_response.aclose()

        return StreamingResponse(
            _body_stream(),
            status_code=upstream_response.status_code,
            headers=_scrub_response_headers(upstream_response.headers),
            media_type=upstream_response.headers.get("content-type"),
        )

    return app


app = build_app()
