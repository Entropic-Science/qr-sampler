"""
title: QR-Sampler Parameters
author: qr-sampler
author_url: https://github.com/alchemystack/qr-sampler
version: 0.1.0
license: MIT
description: Injects qr-sampler per-request parameters into vLLM requests. Configure sampling parameters via Valves to control external-entropy-driven token selection.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Filter:
    """Open WebUI filter that injects qr_* parameters into vLLM requests.

    When qr-sampler is loaded as a vLLM logits processor, it reads per-request
    overrides from extra fields in the request body (prefixed with ``qr_``).
    This filter adds those fields via the ``inlet()`` hook so that Open WebUI
    users can control sampling parameters through the Valves UI without
    manually editing API calls.

    Parameter flow::

        Open WebUI chat -> inlet() adds qr_* keys -> vLLM /v1/chat/completions
        -> SamplingParams.extra_args -> qr-sampler resolve_config()

    Toggle ``enable_qr_sampling`` to False to disable parameter injection
    entirely (requests pass through unmodified).
    """

    class Valves(BaseModel):
        """Admin-configurable qr-sampler parameters.

        Each field maps to a ``qr_*`` key that qr-sampler's
        ``resolve_config()`` accepts. Only per-request-overridable fields
        are exposed here; infrastructure settings (gRPC address, fallback
        mode, etc.) are controlled by environment variables on the vLLM
        container.
        """

        # --- Filter control ---
        priority: int = Field(
            default=0,
            description="Filter execution priority (lower runs first).",
        )
        enable_qr_sampling: bool = Field(
            default=True,
            description=(
                "Master switch. When False, no qr_* parameters are injected "
                "and requests pass through unmodified."
            ),
        )

        # --- Token selection ---
        top_k: int = Field(
            default=0,
            description="Top-k filtering: keep only the k most probable tokens (0 disables).",
        )
        top_p: float = Field(
            default=1.0,
            description="Nucleus sampling threshold (1.0 disables).",
        )

        # --- Temperature ---
        temperature_strategy: str = Field(
            default="fixed",
            description="Temperature strategy: 'fixed' or 'edt' (entropy-dependent).",
        )
        fixed_temperature: float = Field(
            default=0.7,
            description="Constant temperature when strategy is 'fixed'.",
        )
        edt_base_temp: float = Field(
            default=0.8,
            description="Base coefficient for EDT strategy.",
        )
        edt_exponent: float = Field(
            default=0.5,
            description="Power-law exponent for EDT strategy.",
        )
        edt_min_temp: float = Field(
            default=0.1,
            description="EDT temperature floor.",
        )
        edt_max_temp: float = Field(
            default=2.0,
            description="EDT temperature ceiling.",
        )

        # --- Signal amplification ---
        signal_amplifier_type: str = Field(
            default="zscore_mean",
            description="Signal amplification algorithm.",
        )
        sample_count: int = Field(
            default=20480,
            description="Number of entropy bytes to fetch per token.",
        )

        # --- Logging ---
        log_level: str = Field(
            default="summary",
            description="Logging verbosity: 'none', 'summary', or 'full'.",
        )
        diagnostic_mode: bool = Field(
            default=False,
            description="Store all token records in memory for analysis.",
        )

    class UserValves(BaseModel):
        """Per-user qr-sampler preset selector.

        Open WebUI renders this as a dropdown / toggle in each user's
        per-filter settings (it introspects pydantic ``Literal`` fields).
        Distinct from ``Valves``: ``UserValves`` is editable by every
        user, while ``Valves`` is admin-only.
        """

        preset: Literal["creative_sampling", "normal_t1"] = Field(
            default="normal_t1",
            description=(
                "Token sampling preset. 'creative_sampling' is EXPERIMENTAL, "
                "based on the V6_HVD_R01_01 winner from createmp-evalsuite "
                "(hvh_drift family with per-sequence EMA state and dynamic min-p). "
                "'normal_t1' is the vanilla T=1 baseline (quantum entropy still "
                "drives selection)."
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    # Fields that are part of qr-sampler config (everything except filter-
    # internal fields like ``priority`` and ``enable_qr_sampling``).
    _QR_FIELDS: frozenset = frozenset(
        {
            "signal_amplifier_type",
            "sample_count",
            "temperature_strategy",
            "fixed_temperature",
            "edt_base_temp",
            "edt_exponent",
            "edt_min_temp",
            "edt_max_temp",
            "top_k",
            "top_p",
            "log_level",
            "diagnostic_mode",
        }
    )

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Inject qr_* parameters into the request body before it reaches vLLM.

        Each Valve value is added as a top-level ``qr_<field>`` key.  vLLM
        maps unknown top-level keys to ``SamplingParams.extra_args``, and
        qr-sampler's ``resolve_config()`` picks them up from there.

        Args:
            body: The request body (messages, model, stream, etc.).
            __user__: Optional user information from Open WebUI.

        Returns:
            The request body with qr_* keys injected (or unmodified if
            sampling is disabled).
        """
        if not self.valves.enable_qr_sampling:
            return body

        valve_dict = self.valves.model_dump()
        for field_name in self._QR_FIELDS:
            body[f"qr_{field_name}"] = valve_dict[field_name]

        user_valves = None
        if __user__ is not None:
            user_valves = (
                __user__.get("valves")
                if isinstance(__user__, dict)
                else getattr(__user__, "valves", None)
            )
        if user_valves is not None:
            preset_name = getattr(user_valves, "preset", None)
            if preset_name is None and isinstance(user_valves, dict):
                preset_name = user_valves.get("preset")
            if preset_name:
                body["qr_preset"] = preset_name

        return body

    async def outlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Pass-through — no post-processing needed.

        Args:
            body: The response body from vLLM.
            __user__: Optional user information from Open WebUI.

        Returns:
            The response body unmodified.
        """
        return body
