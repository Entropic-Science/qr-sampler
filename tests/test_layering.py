"""Layering guard: the sampling library never imports the tooling layer.

``qr_sampler``'s import graph is layered (AGENTS.md): the library layers
(``core``, ``entropy``, ``amplification``, ``temperature``, ``selection``,
``logging``, ``config``, ``engines``, plus the ``qthought`` roller,
``contract``, ``telemetry``, ``proto``, and ``exceptions``) must never reach
the operator tooling layer (``cli``, ``profiles``, ``templates``) — tooling
imports the library, never the reverse.

This test walks the *static* import graph (AST, so lazy function-local
imports are caught too) from every library module and asserts no tooling
module is reachable. It runs as part of ``python scripts/check.py`` via the
``tests`` oracle.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
PACKAGE = "qr_sampler"

#: Library layers that must never (transitively) import the tooling layer.
LIBRARY_ROOTS = (
    "core",
    "entropy",
    "amplification",
    "temperature",
    "selection",
    "logging",
    "config",
    "engines",
)

#: Operator tooling: reachable FROM the CLI, never from the library.
FORBIDDEN_PREFIXES = (
    f"{PACKAGE}.cli",
    f"{PACKAGE}.profiles",
    f"{PACKAGE}.templates",
)


def _module_name(path: Path) -> str:
    """``src/qr_sampler/entropy/qgrpc/source.py`` -> dotted module name."""
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _all_modules() -> dict[str, Path]:
    return {_module_name(p): p for p in (SRC / PACKAGE).rglob("*.py")}


def _imports_of(module: str, path: Path, known: set[str]) -> set[str]:
    """Every ``qr_sampler``-internal module statically imported by *module*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    is_package = path.name == "__init__.py"
    edges: set[str] = set()

    def _add(target: str) -> None:
        if target.startswith(PACKAGE) and target in known:
            edges.add(target)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ""
            else:
                # Resolve ``from ..x import y`` against this module's package.
                anchor = module.split(".")
                if not is_package:
                    anchor = anchor[:-1]
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join(anchor + ([node.module] if node.module else []))
            _add(base)
            for alias in node.names:
                # ``from x import y`` where y is itself a submodule.
                _add(f"{base}.{alias.name}")
    return edges


def test_library_layers_never_reach_tooling() -> None:
    modules = _all_modules()
    known = set(modules)
    graph = {name: _imports_of(name, path, known) for name, path in modules.items()}

    roots = [
        name
        for name in modules
        if any(name.startswith(f"{PACKAGE}.{layer}") for layer in LIBRARY_ROOTS)
    ]
    assert roots, "no library modules found — did the package layout move?"

    # BFS the import graph from every library module.
    reachable: set[str] = set()
    frontier = list(roots)
    while frontier:
        current = frontier.pop()
        if current in reachable:
            continue
        reachable.add(current)
        frontier.extend(graph.get(current, ()))

    violations = sorted(mod for mod in reachable if mod.startswith(FORBIDDEN_PREFIXES))
    assert not violations, (
        "tooling modules reachable from the library layers "
        f"(cli/profiles/templates must import the library, never the reverse): {violations}"
    )
