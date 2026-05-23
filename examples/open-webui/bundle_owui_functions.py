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

    python bundle_owui_functions.py            # write bundles
    python bundle_owui_functions.py --check    # idempotency check; exit 1 on drift

Outputs `qr_sampler_filter.json` and `qr_comparison_pipe.json` in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

_HELPER_FILES = ("_modal_warmth.py", "entropic_science_profile.py")

_FRONTMATTER_RE = re.compile(r'^"""\s*\n(.*?)\n"""', re.DOTALL)
_LEADING_COMMENTS_RE = re.compile(r"^(?:#[^\n]*\n|\s*\n)+")
_IMPORT_BLOCK_RE = re.compile(
    r"try:\s*\n\s*from \. import.*?(?=\n# -+\n)",
    re.DOTALL,
)


def _strip_leading_comments(src: str) -> str:
    """Drop any leading ``#``-comment lines (e.g. file-level ``# ruff: noqa``).

    Ruff's file-level ``noqa`` directive lives ABOVE the module docstring;
    the bundler's frontmatter regex requires the docstring to be the first
    thing it sees, so the source has to be peeled past those comments before
    the regex matches. We drop the comments from the bundled output too —
    they only matter at lint time on the source, not in the embedded module.
    """
    return _LEADING_COMMENTS_RE.sub("", src, count=1)


def _strip_module_header(src: str) -> str:
    """Drop the leading docstring + `from __future__ import annotations` line."""
    src = _strip_leading_comments(src)
    src = _FRONTMATTER_RE.sub("", src, count=1).lstrip()
    return src.replace("from __future__ import annotations\n\n", "", 1)


def _read_meta_from_docstring(src: str) -> dict[str, str]:
    """Extract `title`, `author`, `version`, `license`, `description`."""
    match = _FRONTMATTER_RE.match(_strip_leading_comments(src))
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


def _bundle(
    plugin_filename: str,
    function_id: str,
    function_name: str,
    extra_meta: dict | None = None,
) -> dict:
    plugin_src = (_HERE / plugin_filename).read_text(encoding="utf-8")
    meta = _read_meta_from_docstring(plugin_src)

    helpers = {name: (_HERE / name).read_text(encoding="utf-8") for name in _HELPER_FILES}
    bundled_src = _inline_helpers(plugin_src, helpers)

    envelope_meta: dict = {
        "description": meta["description"],
        "manifest": {
            "title": meta["title"],
            "author": meta["author"],
            "version": meta["version"],
            "license": meta["license"],
        },
    }
    if extra_meta:
        envelope_meta.update(extra_meta)

    return {
        "id": function_id,
        "name": function_name,
        "meta": envelope_meta,
        "content": bundled_src,
    }


# Machine-readable mirror of `Filter.UserValves.preset`. Redundant with the
# embedded pydantic class but exposes the enum to filter-registry browsers
# that do not introspect the bundled `content` string. Kept here (not in
# qr_sampler_filter.py) so the .py file stays pure pydantic and the bundle
# script owns the JSON-side schema.
_FILTER_USER_VALVES_META: dict = {
    "user_valves": {
        "preset": {
            "type": "string",
            "enum": ["creative_sampling", "normal_t1"],
            "default": "creative_sampling",
            "description": (
                "Token sampling preset. 'creative_sampling' (default, "
                "experimental) uses HVH-Drift dynamic temperature "
                "(V6_HVD_R01_01 winner). 'normal_t1' is the vanilla T=1 "
                "baseline."
            ),
        },
    },
}


def _render_bundles() -> dict[str, str]:
    """Build both envelopes and return ``{relative_path: serialized_json}``."""
    filter_envelope = _bundle(
        "qr_sampler_filter.py",
        "qr_sampler_parameters",
        "QR-Sampler Parameters",
        extra_meta=_FILTER_USER_VALVES_META,
    )
    pipe_envelope = _bundle(
        "qr_comparison_pipe.py",
        "qr_comparison_pipe",
        "QR vs PRNG Comparison",
    )
    return {
        "qr_sampler_filter.json": json.dumps([filter_envelope], indent=2) + "\n",
        "qr_comparison_pipe.json": json.dumps([pipe_envelope], indent=2) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Render bundles in memory and diff against on-disk files; exit 1 "
            "with a one-line summary on any drift. Used by CI / plan-step "
            "verification to guarantee `python bundle_owui_functions.py` is "
            "idempotent and committed."
        ),
    )
    args = parser.parse_args(argv)

    rendered = _render_bundles()

    if args.check:
        drifted: list[str] = []
        for relpath, content in rendered.items():
            path = _HERE / relpath
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != content:
                drifted.append(relpath)
        if drifted:
            print(
                f"bundles drifted: {', '.join(drifted)} — "
                "run `python bundle_owui_functions.py` and commit the changes.",
                file=sys.stderr,
            )
            return 1
        print("bundles are up to date")
        return 0

    for relpath, content in rendered.items():
        (_HERE / relpath).write_text(content, encoding="utf-8")
    print("wrote qr_sampler_filter.json and qr_comparison_pipe.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
