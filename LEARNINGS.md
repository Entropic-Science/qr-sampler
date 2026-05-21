# LEARNINGS

Cross-cutting notes about architectural pivots, observed runtime quirks, and
decisions whose rationale is too long for a commit message. Newest first.

Each entry that mirrors a Claude auto-memory record names it by ID. Do not
duplicate the auto-memory body verbatim -- these notes assume the reader
can resolve cross-references against the live auto-memory. See also
`../Entropic-Science/qr-llm-chat/LEARNINGS.md` for the OWUI-running half
of the same production stack (the active `enginecore_500_open` bug is
tracked there: `POST /v1/chat/completions` returns HTTP 500 with the
"EngineCore encountered an issue. See stack trace (above) for the root
cause." body, and the failure surfaces in the qr-llm-chat-side deploy
guard chat probe).

Cross-cutting auto-memory entries relevant to this repo:
`vllm_not_in_modal_python`, `vllm_model_arch_mismatch`,
`modal_add_python_layering_trap`, `feedback_owui_filter_cross_repo_drift`,
`qrng_colocation_constraint`, `qrng_tcp_preprobe`, `modal_secrets_layout`.

---

## Cross-repo integration contract (qr-sampler <-> qr-llm-chat)

This section is the authoritative record of the integration seam. The
qr-llm-chat LEARNINGS.md mirrors the OWUI-facing half (Pipe streaming
protocol, filter Valves shape); facts that constrain qr-sampler's own
code live here.

### QRNG wire contract (Cipherstone)

`qr_sampler.entropy.quantum.QuantumGrpcSource` uses a **protocol-agnostic**
wire decoder (no generated stubs). Field numbers MUST stay at 1 for both
the request `num_bytes` (uint32 / varint) and the response `data`
(bytes / length-delimited). Other proto fields can be added freely;
field 1 changes are breaking.

- Proto in production: package `qrng`, service `QuantumRNG`, method
  `GetRandomBytes(RandomRequest) returns (RandomResponse)` -- mapped via
  `QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes` (unary).
  `QR_GRPC_STREAM_METHOD_PATH` is the empty string (QRNG proto defines
  no streaming RPC).
- gRPC metadata: single entry, header `api-key` (configurable via
  `QR_GRPC_API_KEY_HEADER`), value = `QR_GRPC_API_KEY`. Bearer secret
  semantics: client never logs it; cloudflared never sees it (api-key
  is end-to-end between qr-sampler and the QRNG server).
- Transport: `cloudflared access tcp` sidecar binds loopback
  `127.0.0.1:50051`. The gRPC channel is `grpc.aio.insecure_channel(...)`
  and is safe **only because** the cloudflared sidecar runs in the same
  container bound to loopback. Do not change the channel construction
  to TLS unless cloudflared is removed.
- No `Healthcheck` RPC: liveness is observed implicitly via the per-token
  `GetRandomBytes` call; the circuit breaker + fallback wrapper handle
  degraded conditions.

### Per-request entropy-source override (`qr_entropy_source_type`)

Enables comparison mode (quantum vs system) on a single GPU without
renting a second one. Implementation contract:

- `src/qr_sampler/config.py`: `entropy_source_type` is in
  `_PER_REQUEST_FIELDS`.
- `src/qr_sampler/entropy/registry.py`: `all_sources()` helper used by
  `VLLMAdapter` during pre-init.
- `src/qr_sampler/engines/vllm.py`:
  - At adapter construction (called from `@modal.enter(snap=True)`), build
    a `dict[str, SamplingPipeline]` keyed by entropy-source-type, one
    entry per source listed in env `QR_PREINIT_ENTROPY_SOURCES` (default
    `"quantum_grpc,system"`).
  - `update_state(req)` reads `qr_entropy_source_type` from `extra_args`;
    defaults to env `QR_ENTROPY_SOURCE_TYPE` if absent; rejects unknown
    / un-preinit'd values with a clean 400.
  - `apply(logits)` looks up per-request pipeline by source key and runs
    `sample_token()`. **Just-in-time invariant preserved**: entropy
    fetched *after* logits are computed.
- Env: `QR_PREINIT_ENTROPY_SOURCES` -- comma-separated. Operators wanting
  only one source set this to e.g. `"quantum_grpc"`.

### Service-token format (filter + pipe sign their requests)

- Header: `X-Service-Token: <unix_ts>.<hmac>`
- `hmac = HMAC-SHA256(<secret>, unix_ts + path)`. The signer always uses
  the **first** entry of `SERVICE_TOKEN_SECRETS`; the verifier accepts a
  match against **any** entry. Rolling-secret rotation: prepend new ->
  redeploy at leisure -> remove old next deploy. No lockstep redeploy
  required.
- 60s timestamp window enforced server-side.
- `SERVICE_TOKEN_SECRETS` is **plural** and comma-separated. The filter's
  Valves field is also plural and follows the same pattern (signs with
  the first secret if multiple are passed).

### Snapshot integrity invariants (load-bearing)

These constrain how qr-sampler init code is written. Modal memory
snapshots in the production deploy depend on them.

- **No live gRPC channel captured in the snapshot.** `quantum_grpc`
  source already uses lazy channel creation. Do NOT "optimise" by moving
  channel construction into `__init__` -- it would freeze a dead socket
  into the snapshot, and the first post-restore request would fail. The
  auto-memory `qrng_tcp_preprobe` entry documents the related fast-fail
  TCP pre-probe that converts ~15 s gRPC retry timeouts into ~500 ms
  fallback engagement.
- **No process-relative state captured.** Avoid `os.getpid()`-based
  caches, in-process locks, or anything that assumes the process started
  fresh.
- **Secrets are mounted after restore** -- Modal honours this
  automatically. qr-sampler's config layer reads env at construction
  time, which is fine **as long as** construction happens inside
  `@modal.enter(snap=True)`, not at module import.

### Region pinning (auto-memory `qrng_colocation_constraint`)

vLLM `@app.cls` `region=` must stay in `["us-east", "us-west"]`.
Cipherstone's gRPC endpoint is east-US-hosted, and every generated token
issues an entropy RPC; cross-region backbone RTT becomes a per-token
cost. Do not relax without measuring.

### Modal Secret split (auto-memory `modal_secrets_layout`)

QRNG / Cipherstone vars live in **`qr-sampler-prod`**, NOT in
`qr-llm-chat-prod`. The vLLM containers (`VllmQrGemma`, `VllmQrQwen`) are
the gRPC consumers and only `qr-sampler-prod` is mounted on those
classes. Mounting QRNG vars in `qr-llm-chat-prod` on `OWUIService` would
be useless (OWUI doesn't call the gRPC client) and a leak hazard.
qr-llm-chat ships `scripts/dump_modal_secret.py` (mounts both secrets,
allow-list disclosure policy) as the operator-side read path.

---

## 2026-05-19 — vLLM deploy unblock: image deps, region widening, container-restore tolerance

After landing the qr-llm-chat split (R2) the first end-to-end smoke against
`modal deploy -m qr_sampler.connectors.modal.app` surfaced three distinct
failure modes in sequence. Each fix exposed the next, classic onion-peeling.

### Failure 1 — `ModuleNotFoundError: No module named 'pydantic'` in vLLM containers

`modal app logs <ap-id>` showed the GPU containers crashing immediately on
restore at `from qr_sampler.config import ...` (which imports pydantic).

**Root cause.** The `vllm_image` declaration was:

```python
vllm_image = modal.Image.from_dockerfile(
    str(Path(__file__).parent / "Dockerfile.vllm"),
    context_dir=str(_REPO_ROOT),
    add_python="3.12",
).add_local_python_source("qr_sampler", copy=True)
```

`add_python="3.12"` tells Modal to install Python 3.12 as a parallel
interpreter on top of the base image. `add_local_python_source` then ships
`/root/qr_sampler/` as importable from 3.12's site-packages. **But** the
Dockerfile's `pip install --no-cache-dir .` only installs to the BASE image's
Python (whatever vllm/vllm-openai:v0.6.6 ships with), not to Modal's added
3.12. So at container restore, 3.12 imports `/root/qr_sampler` but has no
pydantic to satisfy `qr_sampler.config`.

**Fix.** Add `.pip_install(...)` after `from_dockerfile(...)` listing
qr-sampler's runtime deps from `pyproject.toml` (`numpy`, `pydantic`,
`pydantic-settings`, `grpcio`, `protobuf`, `pyyaml`) plus `vllm_serve.py`
needs (`fastapi`, `httpx`). The `.pip_install` layer is built by Modal in
the add_python-selected Python, so the deps land where the import looks.

The pre-existing Dockerfile `pip install .` stays — it's harmless and useful
for any subprocess that uses the base image's Python (e.g. the cloudflared
sidecar's Python).

### Failure 2 — VllmQr* "waiting to be scheduled on GPU_H200 worker"

After fix 1, the GPU container loaded its code, but Modal's scheduler emitted:

> *Function VllmQrGemma.* is waiting to be scheduled on a GPU_H200 worker.
> Relaxing requirements (region=us-east-1 or setting regions=[us-east])
> may lead to faster scheduling.*

H200 capacity in `us-east-1` alone was insufficient. Curl-side symptom was
HTTP 303 + `__modal_function_call_id` query param + 60-90s timeout on the
polled URL — Modal's standard "function-call queued, poll for result"
pattern, which hangs when the function never gets scheduled.

**Fix.** Widen `_CLS_KWARGS["region"]` from `"us-east-1"` (str) to
`["us-east", "us-west"]` (list — the multi-region form Modal's hint
suggests). Both us-east and us-west are valid Modal region groups; the
scheduler picks any available H200 zone within them. QRNG entropy reaches
every container via Cloudflare's global edge, so there is no latency reason
to pin a specific zone. Staying US-only keeps weights in-region for the
`llm-weights` Volume and aligns egress with OWUI's billing zone.

### Failure 3 — `_qr_llm_chat_functions_dir()` raises in vLLM containers

After fixes 1+2, vLLM containers scheduled successfully and started loading.
The next restore failed earlier in the import chain with our own
RuntimeError: *"qr_llm_chat is not importable in the deploy host's Python
env."*

**Root cause.** Modal's `_container_entrypoint` does
`importlib.import_module("qr_sampler.connectors.modal.app")` to find the
`VllmQrGemma` class definition. That runs the ENTIRE module body, including:

```python
_OWUI_IMAGE = (
    modal.Image.debian_slim(...)
    ...
    .add_local_dir(
        str(_qr_llm_chat_functions_dir()),   # <-- evaluated at import time
        remote_path="/root/qr_llm_chat/functions",
        copy=True,
    )
)
```

`_qr_llm_chat_functions_dir()` calls `importlib.util.find_spec("qr_llm_chat")`
and raises if missing. The vLLM containers only ship `qr_sampler` (via
`add_local_python_source` on `vllm_image`, not `_OWUI_IMAGE`), so
qr_llm_chat is genuinely absent there. Pre-fix-1 this never surfaced because
the pydantic crash happened first.

**Fix.** Gate the hard-fail on `MODAL_TASK_ID` being unset — that env var
is set inside every Modal container at runtime and absent on the deploy
host. Container-side, the function returns a placeholder Path. Image
construction (`.add_local_dir(...)`) happens at deploy time on the host
where the real path is returned; container-side, the placeholder is
metadata that Modal never re-reads.

Lesson: any top-level code in `connectors/modal/app.py` that depends on
the deploy host's filesystem must tolerate running inside Modal's
container-restore import chain, where only the package's own files are
present.

### Cross-repo bug surfaced in passing — comparison-pipe URL

While diagnosing the above, an OWUI Playwright smoke surfaced
`Unexpected token 'm', "modal-http"... is not valid JSON` errors in
chat completions. Root cause: `qr_llm_chat.bootstrap_connections._normalize_url`
strips a trailing `/v1` before writing valve URLs, but
`qr_comparison_pipe._stream_completion` (and `_probe_warmth`) was appending
only `/chat/completions` (and `/models`) — landing on routes the FastAPI
dispatcher does not mount. Fixed on the qr-llm-chat side in the same session.

The URL convention is now explicit: bootstrap stores the bare host
(no /v1), the pipe + probe re-append `/v1/<route>`. This matches how
`bootstrap_connections._build_connection_state` already handles OWUI's
Connections (which DO want /v1 on the URL).

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
