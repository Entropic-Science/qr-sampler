# LEARNINGS

Cross-cutting notes about architectural pivots, observed runtime quirks, and
decisions whose rationale is too long for a commit message. Newest first.

## 2026-05-18 — qr-llm split, step R2: OWUIService class + fallback-visibility hook

**Context.** As part of the qr-llm-chat split (entropic.science → standalone
Modal-deployed chat), `connectors/modal/app.py` now owns a third `@app.cls`
— `OWUIService` — so `modal deploy -m qr_sampler.connectors.modal.app`
brings up the two vLLM samplers AND the Open WebUI surface as one unit.

**OWUI image declaration: `add_local_python_source("qr_llm_chat", copy=True)`.**
Three options considered:

* `add_local_python_source("qr_llm_chat", copy=True)` — chosen. Mirrors how
  the qr-sampler package itself is shipped on the vllm image (the line just
  above the OWUI block). Modal resolves the package at deploy time via
  Python's import machinery, so the qr-sampler test suite imports cleanly
  today even before R3 lands the canonical `src/qr_llm_chat/` layout.
* `pip_install_from_pyproject(...)` (suggested in the plan text) — would
  require a path-dep declaration in qr-sampler's own pyproject, coupling
  qr-sampler's published packaging to qr-llm-chat. Rejected.
* `add_local_dir(qr_llm_chat_root, "/repo").run_commands("pip install -e /repo")`
  — works without a sibling editable install on the deploy host, but more
  verbose and out of step with the existing qr_sampler shipping pattern.

**`OWUIService` is intentionally thin.** Five lines of body across two
`@modal.enter` methods plus one `@modal.asgi_app`. All heavy lifting
(admin bootstrap, OWUI Function envelope import, Pipe valve writing) lives
in the `qr_llm_chat.modal_entrypoint` module on the qr-llm-chat side, so
the OWUI lifecycle and the Modal class lifecycle stay 1:1 and the qr-sampler
repo does not depend on `qr-llm-chat` at import time.

**Snapshot-time network probes.** OWUI 0.9.5's `open_webui.config` reaches
out to `localhost:11434` (Ollama) and `huggingface.co:443` (sentence-
transformers cache freshness) at *module import*. If `OWUIService._pre_snapshot`
imports `open_webui.main` without first setting `ENABLE_OLLAMA_API=false`,
`OLLAMA_BASE_URLS=`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`, those
open TCP sockets freeze into the memory snapshot and produce undefined
behaviour on restore. These four env vars are declared on `_OWUI_IMAGE`
directly so they're baked into every container of this class — not relying
on the operator's Secret to set them.

**Fallback-visibility hook in `qr_sampler_filter.py:outlet()`.** When the
vLLM serve layer reports `qr_metadata.last_source_used == "system"` and
the configured primary (read live from `QR_ENTROPY_SOURCE_TYPE`) is
`quantum_grpc`, the filter emits one `status` warning per chat-id via
`__event_emitter__`. Without this hook a user has no UI signal when
quantum-source bytes silently fell back to urandom — fallback is operator-
relevant but it must also be user-observable so users can re-prompt when the
quantum primary is restored.

The hook runs *before* the email gate in `outlet()` so the OWUI-only deploy
profile (no entropic.science allowance metering) also surfaces the warning.

The hook is forward-compatible: it silently no-ops when `qr_metadata` is
absent from the response body. This means the OWUI bundle can be deployed
ahead of the vllm-serve layer that actually attaches the metadata, and the
warning will start firing automatically once both sides are upgraded.

**`bundle_owui_functions.py --check` flag added.** Verifies on-disk
bundles match what the script would render today — catches the case where
a developer edits `qr_sampler_filter.py` or `qr_comparison_pipe.py` without
re-running the bundler. Wired into the R2 plan's verification block.
