"""The cross-repo seam — the only surface downstream consumers may import.

``qr_sampler`` ships as a vLLM logits-processor plugin, but it also has a
second, non-vLLM consumer: the ``qr-llm-qthought`` service imports the
:class:`~qr_sampler.qthought.QthoughtRoller` entropy stack directly (no vLLM,
no GPU) to drive its case-frame grammar. Every internal module boundary in
this package (``config.py`` vs. ``config/``, ``presets.py``, ``qthought.py``,
``entropy/...``) is free to move during a qr-sampler-internal refactor
**as long as this module's** ``__all__`` **keeps re-exporting the same
names** — that is the whole point of a contract module: it decouples "what
qthought imports" from "how qr-sampler is laid out inside".

Rules for anyone editing this file:

* Pure re-export, no logic. If you need to adapt a name, fix it at the
  source, not here.
* Only widen ``__all__`` (add names) in lockstep with a real qthought need;
  do not export something "just in case".
* Bump :data:`CONTRACT_VERSION` on any breaking change to this surface (a
  removed name, a changed signature, a changed field set) — qthought's
  ``qr_qthought.__init__`` asserts this value at import and fails loudly on
  a mismatch, so a stale sibling checkout cannot silently drift.
* ``tests/test_contract.py`` pins ``__all__``, the three qthought preset
  dicts, and the ``QthoughtRoller`` / ``ChoiceProvenance`` shapes this module
  re-exports — it is the drift guard for everything below.
"""

from __future__ import annotations

from qr_sampler.config import (
    BUILTIN_PRESETS,
    PRESET_QTHOUGHT,
    PRESET_QTHOUGHT_THINK,
    PRESET_QTHOUGHT_VOICE,
    QRSamplerConfig,
    resolve_config,
    resolve_preset,
)
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.fallback import FallbackEntropySource
from qr_sampler.entropy.mock import MockUniformSource
from qr_sampler.exceptions import ConfigValidationError, EntropyUnavailableError
from qr_sampler.qthought import BindSpec, ChoiceProvenance, IntRange, QthoughtRoller

#: Bumped on any breaking change to this module's ``__all__`` — a removed
#: name, a changed ``QthoughtRoller``/``ChoiceProvenance`` shape, or a changed
#: qthought preset dict. qthought asserts this at import time.
CONTRACT_VERSION = 1

__all__ = [  # noqa: RUF022 -- grouped by concern (roller/config/entropy/exceptions), not alphabetized
    "CONTRACT_VERSION",
    # roller + provenance
    "QthoughtRoller",
    "ChoiceProvenance",
    "BindSpec",
    "IntRange",
    # config + presets
    "QRSamplerConfig",
    "resolve_config",
    "resolve_preset",
    "BUILTIN_PRESETS",
    "PRESET_QTHOUGHT",
    "PRESET_QTHOUGHT_THINK",
    "PRESET_QTHOUGHT_VOICE",
    # entropy primitives
    "EntropySource",
    "MockUniformSource",
    "FallbackEntropySource",
    # exceptions
    "EntropyUnavailableError",
    "ConfigValidationError",
]
