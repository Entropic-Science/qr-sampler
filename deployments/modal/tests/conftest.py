"""Conftest for ``deployments/modal/tests/``.

These tests sit outside the default ``testpaths`` (``tests/``) and import
``deployments.modal.vllm_serve`` as a top-level namespace package. Pytest's
auto-discovery sets ``rootdir`` from the nearest ``pyproject.toml``, but the
repo root is not always on ``sys.path`` when this directory is invoked
directly via ``pytest deployments/modal/tests/``. Insert it explicitly so the
``deployments.*`` import resolves the same way it does inside the Modal
container.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
