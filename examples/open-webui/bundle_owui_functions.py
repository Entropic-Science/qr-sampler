"""Bundle OWUI plugin source + sibling helpers into single-file JSON wrappers.

OWUI Functions are uploaded as standalone `.py` modules — there is no
multi-file import. The on-disk sources in this directory remain modular
(`qr_sampler_filter.py` + `qr_comparison_pipe.py` import sibling helpers
`_modal_warmth.py` and `entropic_science_profile.py`) so the test suite can
exercise them as a package, and so future operators can read the code without
scrolling through eight hundred lines.

For deployment, this script flattens each plugin: the helper modules are
inlined where the sibling-import block lives, producing one self-contained
`.py` body that we embed inside the OWUI JSON envelope.

Run from this directory:

    python bundle_owui_functions.py

Outputs `qr_sampler_filter.json` and `qr_comparison_pipe.json` in place.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_HELPER_FILES = ("_modal_warmth.py", "entropic_science_profile.py")

_FRONTMATTER_RE = re.compile(r'^"""\s*\n(.*?)\n"""', re.DOTALL)
_IMPORT_BLOCK_RE = re.compile(
    r"try:\s*\n\s*from \. import.*?(?=\n# -+\n)",
    re.DOTALL,
)


def _strip_module_header(src: str) -> str:
    """Drop the leading docstring + `from __future__ import annotations` line."""
    src = _FRONTMATTER_RE.sub("", src, count=1).lstrip()
    return src.replace("from __future__ import annotations\n\n", "", 1)


def _read_meta_from_docstring(src: str) -> dict[str, str]:
    """Extract `title`, `author`, `version`, `license`, `description`."""
    match = _FRONTMATTER_RE.match(src)
    if match is None:
        raise RuntimeError("plugin source is missing the OWUI-style docstring header")

    fields: dict[str, str] = {}
    for raw in match.group(1).splitlines():
        if ":" in raw:
            key, _, value = raw.partition(":")
            fields[key.strip()] = value.strip()
    for required in ("title", "author", "version", "license", "description"):
        if required not in fields:
            raise RuntimeError(f"plugin source missing `{required}:` in docstring header")
    return fields


def _inline_helpers(plugin_src: str, helpers: dict[str, str]) -> str:
    """Replace the `try: from . import ...` fallback block with synthetic modules.

    Each helper file's body is embedded as a string literal and `exec`'d into a
    freshly-created `types.ModuleType`. The plugin can then call
    `_modal_warmth.probe_warmth(...)` exactly as it does when imported as a
    sibling.
    """
    lines: list[str] = [
        "# --- bundled sibling helpers (inlined by bundle_owui_functions.py) ---",
        "import types as _bundle_types",
        "",
    ]
    for filename, src in helpers.items():
        module_name = filename.replace(".py", "")
        body = _strip_module_header(src)
        const_name = f"_BUNDLE_SOURCE_{module_name.upper()}"
        lines.extend(
            [
                f"{const_name} = {json.dumps(body)}",
                f"{module_name} = _bundle_types.ModuleType({module_name!r})",
                f"exec({const_name}, {module_name}.__dict__)",
                "",
            ]
        )
    bundled = "\n".join(lines)

    replaced = _IMPORT_BLOCK_RE.sub(lambda _m: bundled, plugin_src, count=1)
    if replaced == plugin_src:
        raise RuntimeError("could not locate sibling-import block in plugin source")
    return replaced


def _bundle(plugin_filename: str, function_id: str, function_name: str) -> dict:
    plugin_src = (_HERE / plugin_filename).read_text(encoding="utf-8")
    meta = _read_meta_from_docstring(plugin_src)

    helpers = {name: (_HERE / name).read_text(encoding="utf-8") for name in _HELPER_FILES}
    bundled_src = _inline_helpers(plugin_src, helpers)

    return {
        "id": function_id,
        "name": function_name,
        "meta": {
            "description": meta["description"],
            "manifest": {
                "title": meta["title"],
                "author": meta["author"],
                "version": meta["version"],
                "license": meta["license"],
            },
        },
        "content": bundled_src,
    }


def main() -> None:
    filter_envelope = _bundle(
        "qr_sampler_filter.py",
        "qr_sampler_parameters",
        "QR-Sampler Parameters",
    )
    pipe_envelope = _bundle(
        "qr_comparison_pipe.py",
        "qr_comparison_pipe",
        "QR vs PRNG Comparison",
    )

    (_HERE / "qr_sampler_filter.json").write_text(
        json.dumps([filter_envelope], indent=2) + "\n",
        encoding="utf-8",
    )
    (_HERE / "qr_comparison_pipe.json").write_text(
        json.dumps([pipe_envelope], indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote qr_sampler_filter.json and qr_comparison_pipe.json")


if __name__ == "__main__":
    main()
