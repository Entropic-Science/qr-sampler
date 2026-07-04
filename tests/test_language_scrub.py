"""Language-relocation gate (QPI refactor 2026-07, FR-S6).

The research narrative — and its vocabulary — lives in the Qbert0G repo.
qr-sampler describes itself in neutral terms ("weak-signal integration",
"entropy purity verification"). This test walks the installed source tree
and fails if the relocated language ever creeps back into ``src/``.

Scope is deliberately ``src/qr_sampler/`` only: stale build artifacts
(``build/``, ``*.egg-info``) and non-shipping files are out of scope, and
generated proto stubs (``*_pb2*``) are excluded by name.
"""

from __future__ import annotations

from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "qr_sampler"

_CHECKED_SUFFIXES = {".py", ".yaml", ".md"}

_FORBIDDEN = "consciousness"


def _scrubbed_files() -> list[Path]:
    files = []
    for path in sorted(_SRC_ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in _CHECKED_SUFFIXES:
            continue
        if "_pb2" in path.name:
            continue
        if "egg-info" in str(path):
            continue
        files.append(path)
    return files


class TestLanguageScrub:
    def test_src_tree_exists_and_is_nonempty(self) -> None:
        files = _scrubbed_files()
        assert len(files) > 20, "scrub walk found suspiciously few files"

    def test_no_relocated_language_in_src(self) -> None:
        offenders: list[str] = []
        for path in _scrubbed_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            if _FORBIDDEN in text.lower():
                offenders.append(str(path.relative_to(_SRC_ROOT)))
        assert offenders == [], (
            f"relocated language ({_FORBIDDEN!r}) found in src/: {offenders}; "
            "the research narrative lives in the Qbert0G repo — use neutral "
            "phrasing here (see AGENTS.md invariant 20)"
        )
