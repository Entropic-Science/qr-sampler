#!/usr/bin/env python
"""One-command verification for qr-sampler.

Runs every oracle this repo is gated on, in order. CI and pre-commit invoke
this same script (with ``--only``) so the check list cannot drift between
local runs, hooks, and CI.

Usage::

    python scripts/check.py                 # run everything
    python scripts/check.py --only lint,types
    python scripts/check.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Ordered oracle table. Keys are ``--only`` selectors; values are the exact
#: commands (run from the repo root, using this interpreter's environment).
CHECKS: dict[str, list[str]] = {
    "lint": [sys.executable, "-m", "ruff", "check", "."],
    "format": [sys.executable, "-m", "ruff", "format", "--check", "."],
    "types": [sys.executable, "-m", "mypy", "--strict", "src/"],
    "security": [sys.executable, "-m", "bandit", "-c", "pyproject.toml", "-r", "src/", "-q"],
    "tests": [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=src/qr_sampler"],
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of checks to run (default: all). "
        f"Available: {', '.join(CHECKS)}",
    )
    parser.add_argument("--list", action="store_true", help="List available checks and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, cmd in CHECKS.items():
            print(f"{name}: {' '.join(cmd[1:])}")
        return 0

    if args.only is None:
        selected = list(CHECKS)
    else:
        selected = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = [name for name in selected if name not in CHECKS]
        if unknown:
            parser.error(f"unknown check(s): {', '.join(unknown)} (available: {', '.join(CHECKS)})")

    failed: list[str] = []
    for name in selected:
        cmd = CHECKS[name]
        print(f"\n=== {name}: {' '.join(cmd[1:])} ===", flush=True)
        result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print(f"OK: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
