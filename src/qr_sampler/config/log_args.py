"""Request arguments that control logging and are not sampler configuration.

``qr_token_log`` asks for the per-request token log and ``qr_request_id``
names its file (see :mod:`qr_sampler.logging.token_log`). They travel in
``extra_args`` next to the real ``qr_*`` overrides, but they never reach
:class:`~qr_sampler.config.model.QRSamplerConfig`: a logged request resolves
to the same configuration, the same config hash and the same shared amplifier
as an unlogged one, so asking for the log cannot change what is sampled.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

LOG_FLAG_KEY = "qr_token_log"
REQUEST_ID_KEY = "qr_request_id"
LOG_ARG_KEYS: frozenset[str] = frozenset({LOG_FLAG_KEY, REQUEST_ID_KEY})

_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@=+-]{0,199}$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


class TokenLogArgError(ValueError):
    """``qr_token_log`` or ``qr_request_id`` has a value that cannot be used."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise TokenLogArgError(f"{LOG_FLAG_KEY} must be true or false, got {value!r}")


def parse_log_args(extra_args: Mapping[str, Any] | None) -> tuple[bool, str | None]:
    """``(log wanted, request id)`` from a request's ``extra_args``.

    The id becomes a file name, so it is held to letters, digits and
    ``._:/@=+-`` (``/`` is written as ``__``) and may not contain ``..``.

    Raises:
        TokenLogArgError: On a non-boolean ``qr_token_log``, a request id that
            is not a safe file name, or a log request with no id.
    """
    if not extra_args:
        return False, None
    wanted = _as_bool(extra_args.get(LOG_FLAG_KEY, False))
    raw_id = extra_args.get(REQUEST_ID_KEY)
    request_id: str | None = None
    if raw_id is not None:
        if not isinstance(raw_id, str) or not _REQUEST_ID.fullmatch(raw_id) or ".." in raw_id:
            raise TokenLogArgError(
                f"{REQUEST_ID_KEY} must be 1-200 characters of letters, digits and "
                "'._:/@=+-', starting with a letter or digit and without '..'"
            )
        request_id = raw_id
    if wanted and request_id is None:
        raise TokenLogArgError(f"{LOG_FLAG_KEY} needs {REQUEST_ID_KEY} to name the log file")
    return wanted, request_id


__all__ = [
    "LOG_ARG_KEYS",
    "LOG_FLAG_KEY",
    "REQUEST_ID_KEY",
    "TokenLogArgError",
    "parse_log_args",
]
