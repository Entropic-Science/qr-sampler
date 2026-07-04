"""Coherence-gated temperature strategy (spec FR-S4, Amendment 1.2).

Composes over an inner temperature strategy (default ``fixed``) and adds a
temperature boost driven by the cross-device coherence statistic that rides
each server-integrated draw's ``DrawMeta``:

    b   = coherence_t_boost_max * max(0, r)     iff  coherence_valid
                                                     and coherence_z >= threshold
    b_ema <- alpha * b + (1 - alpha) * b_ema

The pipeline feeds ``DrawMeta`` through the duck-typed ``observe_draw_meta``
hook right after the fetch; since temperature is pipeline stage 1 and the
fetch is stage 2, the meta observed at token *t* first affects token *t+1*
— the one-draw lag is structural, not simulated.

The boost is applied UPSTREAM of the inner strategy by shifting its
base-temperature field on a per-token config copy (see :data:`_BASE_FIELD`).
Because temperature division is selector stage 1 — upstream of
top-k/softmax/min-p/top-p — a widened distribution also widens which tokens
survive truncation. Inner strategies without a known base field get the
boost added to their result temperature instead (documented fallback).

Fail-safe contract: first token, missing meta, or ANY internal gate failure
yields exactly the inner strategy's unmodified result (``T_base``); no
exceptions from the gate machinery ever escape ``compute_temperature``.

This module deliberately imports nothing from ``qr_sampler.entropy`` — the
meta object is duck-read via ``getattr`` so the temperature layer stays
decoupled from the transport layer.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final

from qr_sampler.temperature.base import TemperatureResult, TemperatureStrategy
from qr_sampler.temperature.registry import TemperatureStrategyRegistry

if TYPE_CHECKING:
    import numpy as np

    from qr_sampler.config import QRSamplerConfig

_logger = logging.getLogger("qr_sampler")

#: Base-temperature config field per builtin inner strategy. The boost is
#: added to this field on a config copy BEFORE the inner strategy runs, so
#: it participates in the inner's own formula and clamping. Strategies not
#: listed here get the boost post-added to their result temperature.
_BASE_FIELD: Final[dict[str, str]] = {
    "fixed": "fixed_temperature",
    "edt": "edt_base_temp",
    "hvh_drift": "hvh_t_base",
}

#: Snap-to-zero floor for the EMA boost. A geometric EMA never reaches 0.0
#: in floating point, so without a floor a single gate-open event would keep
#: ``gate_open`` true (and the status file churning) for thousands of tokens
#: while the boost is physically meaningless (~1e-30). Below this floor the
#: EMA is snapped to exactly 0.0, closing the gate.
_EMA_FLOOR: Final[float] = 1e-6


class CoherenceGateStrategy(TemperatureStrategy):
    """Coherence-gated boost composed over an inner temperature strategy.

    Per-request state (EMA boost, last observed draw meta, lazily built
    inner instance) lives on the instance — engine adapters build a fresh
    instance per request (the ``_RequestState`` pattern), like
    ``hvh_drift``.
    """

    def __init__(self, vocab_size: int) -> None:
        """Initialize gate state.

        Args:
            vocab_size: Model vocabulary size, forwarded to the inner
                strategy's constructor when it takes one (e.g. EDT).
        """
        self._vocab_size = vocab_size
        self._ema_boost: float = 0.0
        self._last_meta: Any | None = None
        self._inner: TemperatureStrategy | None = None
        self._inner_name: str = ""

    def observe_draw_meta(self, meta: Any) -> None:
        """Record the latest draw's metadata (duck-typed pipeline hook).

        Called by the pipeline after EVERY draw-mode fetch: with the
        ``DrawMeta`` on success, with ``None`` on a degraded (failed) draw.
        The ``None`` case clears the stored meta so a PurityService outage
        hard-resets the gate on the next token instead of replaying stale
        coherence evidence. The meta is only stored here; all reads happen
        (fail-safe) inside :meth:`compute_temperature` on the NEXT token.

        Args:
            meta: A ``DrawMeta``-like object exposing ``coherence_z``,
                ``coherence_valid``, and ``coherence_r`` attributes, or
                ``None`` to signal a degraded draw.
        """
        self._last_meta = meta

    def compute_temperature(self, logits: np.ndarray, config: QRSamplerConfig) -> TemperatureResult:
        """Compute the (possibly boosted) temperature via the inner strategy.

        Args:
            logits: 1-D logit array for the current token.
            config: Active configuration providing the ``coherence_*`` gate
                parameters and the inner strategy's own parameters.

        Returns:
            The inner strategy's ``TemperatureResult`` with its diagnostics
            passed through (critically including ``hvh_drift``'s per-token
            ``min_p``) plus the gate keys ``strategy``/``inner``/
            ``gate_open``/``gate_boost``/``coherence_z``/``coherence_valid``.
            Every failure path returns exactly the unmodified inner result.
        """
        inner = self._get_inner(config)

        # Gate math is fully fail-safe: any problem (malformed meta,
        # bad config values, ...) collapses the boost to 0.0 so the
        # inner result below is exactly T_base.
        coherence_z: float | None = None
        coherence_valid = False
        try:
            meta = self._last_meta
            if meta is None:
                # First token / degraded draw: no evidence — hard reset.
                self._ema_boost = 0.0
            else:
                coherence_valid = bool(meta.coherence_valid)
                coherence_z = float(meta.coherence_z)
                boost = 0.0
                if coherence_valid and coherence_z >= config.coherence_threshold:
                    boost = config.coherence_t_boost_max * max(0.0, float(meta.coherence_r))
                alpha = config.coherence_ema_alpha
                self._ema_boost = alpha * boost + (1.0 - alpha) * self._ema_boost
                if self._ema_boost < _EMA_FLOOR:
                    # Geometric decay never reaches 0.0 on its own — snap,
                    # so the gate genuinely re-closes after an open event.
                    self._ema_boost = 0.0
        except Exception:
            _logger.warning("coherence_gate: gate evaluation failed; boost reset", exc_info=True)
            self._ema_boost = 0.0
            coherence_z = None
            coherence_valid = False

        b_ema = self._ema_boost

        # Run the inner strategy — boosted config copy when the gate has
        # any accumulated boost, the caller's config verbatim otherwise.
        # If the BOOSTED run fails for any reason, fall back to the
        # unmodified inner run (exactly T_base). An unboosted inner
        # failure is an inner-strategy bug and propagates as such.
        if b_ema > 0.0:
            try:
                result = self._run_boosted(inner, logits, config, b_ema)
            except Exception:
                _logger.warning(
                    "coherence_gate: boosted inner compute failed; using T_base", exc_info=True
                )
                b_ema = self._ema_boost = 0.0
                result = inner.compute_temperature(logits, config)
        else:
            result = inner.compute_temperature(logits, config)

        return TemperatureResult(
            temperature=result.temperature,
            shannon_entropy=result.shannon_entropy,
            diagnostics={
                **result.diagnostics,
                "strategy": "coherence_gate",
                "inner": self._inner_name,
                "gate_open": b_ema > 0.0,
                "gate_boost": b_ema,
                "coherence_z": coherence_z,
                "coherence_valid": coherence_valid,
            },
        )

    def _run_boosted(
        self,
        inner: TemperatureStrategy,
        logits: np.ndarray,
        config: QRSamplerConfig,
        b_ema: float,
    ) -> TemperatureResult:
        """Run *inner* with the boost applied pre- or post-compute."""
        field = _BASE_FIELD.get(self._inner_name)
        if field is not None:
            inner_cfg = config.model_copy(update={field: getattr(config, field) + b_ema})
            return inner.compute_temperature(logits, inner_cfg)
        # Unknown base field (third-party inner): post-add the boost to the
        # result's temperature — it then bypasses the inner's own clamps,
        # which is why the builtin table above is preferred.
        result = inner.compute_temperature(logits, config)
        return TemperatureResult(
            temperature=result.temperature + b_ema,
            shannon_entropy=result.shannon_entropy,
            diagnostics=result.diagnostics,
        )

    def _get_inner(self, config: QRSamplerConfig) -> TemperatureStrategy:
        """Build (lazily, once) the composed inner strategy.

        The inner is built on first compute because the config is only
        available then. An unresolvable or self-referential
        ``coherence_inner_strategy`` falls back to ``fixed`` (logged) so
        the gate never takes sampling down.
        """
        inner = self._inner
        if inner is None:
            name = str(getattr(config, "coherence_inner_strategy", "fixed") or "fixed")
            if name == "coherence_gate":
                _logger.warning(
                    "coherence_gate: self-referential inner strategy; falling back to 'fixed'"
                )
                name = "fixed"
            try:
                inner = TemperatureStrategyRegistry.build(
                    SimpleNamespace(temperature_strategy=name), self._vocab_size
                )
            except Exception:
                _logger.warning(
                    "coherence_gate: inner strategy %r failed to build; falling back to 'fixed'",
                    name,
                    exc_info=True,
                )
                name = "fixed"
                inner = TemperatureStrategyRegistry.build(
                    SimpleNamespace(temperature_strategy=name), self._vocab_size
                )
            self._inner = inner
            self._inner_name = name
        return inner
