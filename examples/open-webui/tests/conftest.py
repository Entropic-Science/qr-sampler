"""Shared loaders for OWUI-plugin tests.

The plugin files live outside the `qr_sampler` package (under
`examples/open-webui/`) and are loaded by Open WebUI as standalone modules.
We mirror that loading shape in tests so test imports don't depend on
turning the directory into a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PLUGIN_DIR = _HERE.parent


def _load(name: str, filename: str) -> Any:
    """Load a sibling module file by absolute path, caching in `sys.modules`."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _PLUGIN_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_modal_warmth() -> Any:
    return _load("qr_sampler_owui_modal_warmth", "_modal_warmth.py")


def load_profile() -> Any:
    return _load("qr_sampler_owui_profile", "entropic_science_profile.py")


def load_filter() -> Any:
    # Profile must be importable as a sibling first so the filter's relative
    # fallback finds it.
    load_modal_warmth()
    load_profile()
    return _load("qr_sampler_owui_filter_cold_start", "qr_sampler_filter.py")


def load_pipe() -> Any:
    load_modal_warmth()
    load_profile()
    return _load("qr_sampler_owui_pipe_cold_start", "qr_comparison_pipe.py")
