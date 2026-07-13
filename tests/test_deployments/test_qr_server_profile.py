"""Structural gate for the ``qr-server`` self-containment deployment profile.

``deployments/qr-server/`` is the box-level layout that collapses the two former
per-app ``*-vllm`` + ``*-qbert0g`` units into ONE shared vLLM engine and ONE
shared Qbert0G daemon (spec §4.1, AC-1). These are deployment artifacts, not
importable code, so nothing else in the suite touches them — this test asserts
the profile exists, the systemd units are structurally sane, the load-bearing
serialization flag is present (``--max-num-seqs 1``, spec §3.2 / AC-7), and the
shared entropy config pins the single-draw-card + coherence-reference layout
(spec §4.2).

``systemd-analyze verify`` is not available on the dev box (Windows), so the
``.service`` check is a structural parse rather than a real systemd lint.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_PROFILE = Path(__file__).resolve().parents[2] / "deployments" / "qr-server"

_REQUIRED_FILES = (
    "qbert0g.service",
    "qr-sampler-vllm.service",
    "qbert0g.config.yaml.example",
    "qr-server.env.example",
    "README.md",
)


def _sections(unit_text: str) -> set[str]:
    """Return the ``[Section]`` headers declared in a systemd unit."""
    return {
        line.strip()[1:-1]
        for line in unit_text.splitlines()
        if line.strip().startswith("[") and line.strip().endswith("]")
    }


class TestProfileFilesExist:
    def test_profile_dir_exists(self) -> None:
        assert _PROFILE.is_dir(), f"missing deployment profile: {_PROFILE}"

    def test_all_required_files_present(self) -> None:
        missing = [name for name in _REQUIRED_FILES if not (_PROFILE / name).is_file()]
        assert missing == [], f"qr-server profile missing files: {missing}"


class TestSystemdUnitsParse:
    def test_units_have_required_sections(self) -> None:
        for unit in ("qbert0g.service", "qr-sampler-vllm.service"):
            text = (_PROFILE / unit).read_text(encoding="utf-8")
            sections = _sections(text)
            assert {"Unit", "Service", "Install"} <= sections, (
                f"{unit} missing systemd sections; found {sections}"
            )
            assert "ExecStart=" in text, f"{unit} has no ExecStart"

    def test_vllm_unit_running_batch_is_env_driven(self) -> None:
        """--max-num-seqs is env-driven via $SHARED_MAX_NUM_SEQS (2026-07).

        Was a hard-pinned ``--max-num-seqs 1`` for the fairness/serialization
        guarantee (AC-7); relaxed once daemon-side draw failover was disabled
        so a busy draw card waits instead of mis-routing. The value comes
        from the profile ``.env`` (default 4 for quantum serving; 16 is the
        PRNG-research bypass setting) so it must NOT be a hard-coded literal.
        """
        text = (_PROFILE / "qr-sampler-vllm.service").read_text(encoding="utf-8")
        assert "--max-num-seqs ${SHARED_MAX_NUM_SEQS}" in text
        assert "--max-num-seqs 1" not in text, "must not hard-pin the running batch"
        # Loopback-only OpenAI endpoint on the shared port.
        assert "--host 127.0.0.1 --port 8000" in text
        # qthought's propose_speech tool call needs the xml parser; harmless to owui.
        assert "--tool-call-parser qwen3_xml" in text
        # Pipeline (not tensor) parallel across the 4 no-NVLink cards.
        assert "--pipeline-parallel-size 4" in text

    def test_vllm_unit_wires_entropy_health_middleware(self) -> None:
        """GET /health/entropy must exist on the stock vLLM API server.

        The passive status-file reader (qr_sampler.engines.vllm.health) is
        what the OWUI setup guard / qr-status chip / no-silent-PRNG banner
        probe. The argv form MUST be ``module.callable`` (vLLM rsplits on
        '.'); a ``module:callable`` typo crashes only after minutes of
        engine init, so pin the exact string here.
        """
        text = (_PROFILE / "qr-sampler-vllm.service").read_text(encoding="utf-8")
        flag = "--middleware qr_sampler.engines.vllm.health.entropy_health_middleware"
        assert flag in text
        assert ":entropy_health_middleware" not in text  # the crash-after-3-min typo
        # The referenced callable must actually exist and be a coroutine
        # function (vLLM applies functions via app.middleware('http')).
        import inspect

        from qr_sampler.engines.vllm.health import entropy_health_middleware

        assert inspect.iscoroutinefunction(entropy_health_middleware)

    def test_qbert0g_unit_serves_shared_socket(self) -> None:
        text = (_PROFILE / "qbert0g.service").read_text(encoding="utf-8")
        assert "qbert0g serve" in text


class TestSharedEntropyConfig:
    """The shared config pins the single-draw-card + coherence-reference layout."""

    def _config(self) -> dict:
        raw = (_PROFILE / "qbert0g.config.yaml.example").read_text(encoding="utf-8")
        loaded = yaml.safe_load(raw)
        assert isinstance(loaded, dict)
        return loaded

    def test_only_dragonfly0_is_drawable(self) -> None:
        cfg = self._config()
        assert cfg["integration"]["sources"] == ["dragonfly-0"], (
            "dragonfly-1 must never be a drawable source"
        )

    def test_dragonfly1_is_coherence_reference_only(self) -> None:
        cfg = self._config()
        pair = cfg["coherence"]["pair"]
        assert "dragonfly-1" in pair
        assert "dragonfly-1" not in cfg["integration"]["sources"]

    def test_draw_failover_disabled(self) -> None:
        # Single drawable card: daemon-side device failover must be OFF so a
        # busy dragonfly-0 makes a draw WAIT for the card rather than
        # mis-routing it to the coherence-only dragonfly-1 (which has no draw
        # fingerprint → the FAILED_PRECONDITION storm / silent PRNG degrade
        # that slowed the box to a crawl, 2026-07). Pairs with the env-driven
        # --max-num-seqs relaxation in qr-sampler-vllm.service.
        cfg = self._config()
        assert cfg["server"]["failover_enabled"] is False, (
            "draws must never fail over to the coherence-only card"
        )

    def test_freshness_and_non_reuse_pins(self) -> None:
        cfg = self._config()
        fresh = cfg["freshness"]
        assert fresh["flush_device_buffer"] is True
        assert fresh["allow_pooling"] is False
        assert fresh["allow_pregeneration"] is False

    def test_raw_post_processing(self) -> None:
        cfg = self._config()
        assert cfg["post_processing"]["mode"] == "raw"

    def test_request_cap_covers_integration_block(self) -> None:
        cfg = self._config()
        assert cfg["limits"]["max_bytes_per_request"] >= 2_097_152
        assert cfg["limits"]["max_bytes_per_request"] >= cfg["integration"]["block_bytes"]


class TestSharedConfigMirrorsQbert0G:
    """The canonical config is kept byte-identical to the Qbert0G-repo copy.

    The sibling checkout is not guaranteed present (CI may build one repo at a
    time), so the mirror assertion is skipped when the sibling copy is absent —
    the local invariant pins above still hold unconditionally.
    """

    def test_byte_identical_when_sibling_present(self) -> None:
        sibling = (
            Path(__file__).resolve().parents[3]
            / "Entropic-Science"
            / "Qbert0G"
            / "deployments"
            / "qr-server"
            / "qbert0g.config.yaml.example"
        )
        if not sibling.is_file():
            return
        here = (_PROFILE / "qbert0g.config.yaml.example").read_bytes()
        assert here == sibling.read_bytes(), (
            "qr-server qbert0g config drifted from the Qbert0G canonical copy"
        )
