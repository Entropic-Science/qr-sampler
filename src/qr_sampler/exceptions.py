"""Exception hierarchy for qr-sampler.

All exceptions derive from QRSamplerError, enabling broad catch patterns
at the application boundary while allowing fine-grained handling internally.
"""


class QRSamplerError(Exception):
    """Base exception for all qr-sampler errors."""


class EntropyUnavailableError(QRSamplerError):
    """No entropy source can provide bytes.

    Raised when the primary entropy source fails and either no fallback
    is configured or the fallback also fails.
    """


class ConfigValidationError(QRSamplerError):
    """Configuration field validation failed.

    Raised when per-request extra_args contain invalid keys, attempt to
    override non-overridable infrastructure fields, or fail type validation.

    Deliberately NOT a ``ValueError``: pydantic validators convert any
    ``ValueError`` raised inside them into a pydantic ``ValidationError``,
    and this type's contract is to propagate RAW out of
    ``QRSamplerConfig``'s ``@field_validator``/``@model_validator`` hooks
    so the config surface fails with one exception type. The vLLM
    request-boundary variant that DOES need ``ValueError`` semantics is
    :class:`RequestRejectedError` below.
    """


class RequestRejectedError(ConfigValidationError, ValueError):
    """Per-request rejection at the vLLM API boundary.

    Raised by ``VLLMAdapter.validate_params`` (called in the API-server
    process via ``SamplingParams._validate_logits_processors``). Also a
    ``ValueError`` because vLLM's OpenAI server maps ``ValueError`` from
    params validation to a clean per-request 400 — any other exception
    type surfaces as an opaque 500 InternalServerError (verified against
    vLLM 0.24.0 on the deployment). Subclasses
    :class:`ConfigValidationError` so existing callers and tests that
    catch the config-surface type keep working. Never raised inside
    pydantic validators (a ``ValueError`` there would be swallowed into a
    pydantic ``ValidationError`` — see :class:`ConfigValidationError`).
    """


class SignalAmplificationError(QRSamplerError):
    """Signal amplification computation failed.

    Raised when the amplifier receives invalid input (e.g., empty bytes)
    or encounters a numerical error during z-score computation.
    """


class TokenSelectionError(QRSamplerError):
    """Token selection failed.

    Raised when no candidate tokens survive top-k and top-p filtering,
    making it impossible to select a token from the CDF.
    """
