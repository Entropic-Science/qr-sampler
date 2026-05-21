# LEARNINGS

Cross-cutting notes about architectural pivots, observed runtime quirks,
and decisions whose rationale is too long for a commit message.

Each entry that mirrors a Claude auto-memory record names it by ID. Do
not duplicate the auto-memory body verbatim — these notes assume the
reader can resolve cross-references against the live auto-memory. See
also `../Entropic-Science/qr-llm-chat/LEARNINGS.md` for the OWUI-running
half of the same production stack.

Cross-cutting auto-memory entries relevant to this repo:
`vllm_not_in_modal_python`, `vllm_model_arch_mismatch`,
`modal_add_python_layering_trap`, `feedback_owui_filter_cross_repo_drift`,
`qrng_colocation_constraint`, `qrng_tcp_preprobe`, `modal_secrets_layout`,
`vllm_modal_host_binding`, `vllm_cli_flag_churn`.

---

## Cross-repo integration contract

This section is the authoritative record of the integration seam.
`qr-llm-chat/LEARNINGS.md` mirrors the OWUI-facing half (Pipe streaming
protocol, filter Valves shape); facts that constrain qr-sampler's own
code live here.

### QRNG wire contract (Cipherstone)

`qr_sampler.entropy.quantum.QuantumGrpcSource` uses a
**protocol-agnostic** wire decoder (no generated stubs). Field numbers
MUST stay at 1 for both the request `num_bytes` (uint32 / varint) and
the response `data` (bytes / length-delimited). Other proto fields can
be added freely; field 1 changes are breaking.

- Proto in production: package `qrng`, service `QuantumRNG`, method
  `GetRandomBytes(RandomRequest) returns (RandomResponse)` — mapped
  via `QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes` (unary).
  `QR_GRPC_STREAM_METHOD_PATH` is the empty string (QRNG proto defines
  no streaming RPC).
- gRPC metadata: single entry, header `api-key` (configurable via
  `QR_GRPC_API_KEY_HEADER`), value = `QR_GRPC_API_KEY`. Bearer-secret
  semantics: client never logs it; cloudflared never sees it (api-key
  is end-to-end between qr-sampler and the QRNG server).
- Transport: `cloudflared access tcp` sidecar binds loopback
  `127.0.0.1:50051`. The gRPC channel is `grpc.aio.insecure_channel(...)`
  and is safe **only because** the cloudflared sidecar runs in the
  same container bound to loopback. Do not change channel construction
  to TLS unless cloudflared is removed.
- No `Healthcheck` RPC: liveness is observed implicitly via the
  per-token `GetRandomBytes` call; the circuit breaker + fallback
  wrapper handle degraded conditions.

### Per-request entropy-source override (`qr_entropy_source_type`)

Enables comparison mode (quantum vs system) on a single GPU without
renting a second one. Implementation contract:

- `src/qr_sampler/config.py`: `entropy_source_type` is in
  `_PER_REQUEST_FIELDS`.
- `src/qr_sampler/entropy/registry.py`: `all_sources()` helper used
  by `VLLMAdapter` during pre-init.
- `src/qr_sampler/engines/vllm.py`:
  - At adapter construction (called from `@modal.enter(snap=True)`),
    build a `dict[str, SamplingPipeline]` keyed by entropy-source-type,
    one entry per source listed in env `QR_PREINIT_ENTROPY_SOURCES`
    (default `"quantum_grpc,system"`).
  - `update_state(req)` reads `qr_entropy_source_type` from
    `extra_args`; defaults to env `QR_ENTROPY_SOURCE_TYPE` if absent;
    rejects unknown / un-preinit'd values with a clean 400.
  - `apply(logits)` looks up the per-request pipeline by source key
    and runs `sample_token()`. **Just-in-time invariant preserved**:
    entropy fetched *after* logits are computed.
- Env: `QR_PREINIT_ENTROPY_SOURCES` — comma-separated. Operators
  wanting only one source set this to e.g. `"quantum_grpc"`.

### Service-token format (filter + pipe sign their requests)

- Header: `X-Service-Token: <unix_ts>.<hmac>`.
- `hmac = HMAC-SHA256(<secret>, unix_ts + path)`. The signer always
  uses the **first** entry of `SERVICE_TOKEN_SECRETS`; the verifier
  accepts a match against **any** entry. Rolling-secret rotation:
  prepend new → redeploy at leisure → remove old next deploy. No
  lockstep redeploy required.
- 60 s timestamp window enforced server-side.
- `SERVICE_TOKEN_SECRETS` is **plural** and comma-separated. The
  filter's Valves field is also plural and follows the same pattern
  (signs with the first secret if multiple are passed).

### Snapshot integrity invariants (load-bearing)

These constrain how qr-sampler init code is written. Modal memory
snapshots in the production deploy depend on them.

- **No live gRPC channel captured in the snapshot.** `quantum_grpc`
  source already uses lazy channel creation. Do NOT "optimise" by
  moving channel construction into `__init__` — it would freeze a
  dead socket into the snapshot, and the first post-restore request
  would fail. Auto-memory `qrng_tcp_preprobe` documents the related
  fast-fail TCP pre-probe that converts ~15 s gRPC retry timeouts
  into ~500 ms fallback engagement.
- **No process-relative state captured.** Avoid `os.getpid()`-based
  caches, in-process locks, or anything that assumes the process
  started fresh.
- **Secrets are mounted after restore** — Modal honours this
  automatically. qr-sampler's config layer reads env at construction
  time, which is fine **as long as** construction happens inside
  `@modal.enter(snap=True)`, not at module import.

### Region pinning (auto-memory `qrng_colocation_constraint`)

vLLM `@app.cls` `region=` must stay in `["us-east", "us-west"]`.
Cipherstone's gRPC endpoint is east-US-hosted, and every generated
token issues an entropy RPC; cross-region backbone RTT becomes a
per-token cost. Do not relax without measuring.

### Modal Secret split (auto-memory `modal_secrets_layout`)

QRNG / Cipherstone vars live in **`qr-sampler-prod`**, NOT in
`qr-llm-chat-prod`. The vLLM containers (`VllmQrQwen`) are the gRPC
consumers and only `qr-sampler-prod` is mounted on those classes.
Mounting QRNG vars in `qr-llm-chat-prod` on `OWUIService` would be
useless (OWUI doesn't call the gRPC client) and a leak hazard.
qr-llm-chat ships `scripts/dump_modal_secret.py` (mounts both secrets,
allow-list disclosure policy) as the operator-side read path.

### Modal `@modal.web_server` needs `--host 0.0.0.0` (auto-memory `vllm_modal_host_binding`)

`@modal.web_server(port=N)` proxies inbound traffic via the
container's external network interface, NOT loopback. vllm serve must
bind `0.0.0.0` in the spawn argv or every external request silently
hangs (Modal's edge accepts TLS then never receives a response).
Internal `_start_and_sleep` / `_wake` probes can still hit
`127.0.0.1:8000` — they share the process's network namespace. See
the full discovery + symptom set in `qr-llm-chat/LEARNINGS.md` § *Modal
@modal.web_server needs --host 0.0.0.0* (the bug was diagnosed there
during iter-08).

### vLLM reasoning parser (iter-11)

The spawn argv passes `--reasoning-parser qwen3`. vLLM 0.17+ extracts
inline `<think>...</think>` blocks from Qwen3-family responses into a
separate `reasoning` field on the OpenAI chat-completion response,
distinct from `content`. Open WebUI 0.9.5 renders it as a collapsible
"Thinking" panel. The qr-llm-chat comparison Pipe's `_extract_delta_text`
reads both `delta.reasoning` (vLLM 0.17's actual field name) and
`delta.reasoning_content` (the spec name) for forward-compat.

---

## vLLM 0.17 / Modal-specific gotchas

### `add_python` + Dockerfile pip-install layering (auto-memory `modal_add_python_layering_trap` + `vllm_not_in_modal_python`)

`Image.from_dockerfile(...).add_python("3.12")` does NOT layer
Dockerfile-side pip installs into Modal's auto-injected Python 3.12.
The Dockerfile's `pip install` lands in the base image's Python;
Modal's 3.12 sees a different `site-packages` and can't import them.

Vllm itself was a victim — fixed by dropping `add_python` and adding
a `python` symlink in `Dockerfile.vllm` (auto-memory
`vllm_not_in_modal_python`). The fact pattern recurs whenever a new
Python dep is added; check whether it needs `.pip_install(...)` even
if it's already in the Dockerfile.

### vLLM model architecture mismatch (auto-memory `vllm_model_arch_mismatch`)

`Qwen/Qwen3.6-27B` has a populated `vision_config` in its HF config
(`architectures=['Qwen3_5ForConditionalGeneration']`). vLLM V1's
`profile_run` raises `StopIteration` when iterating
`vision_config.merge_size`. Affects vLLM 0.17.0 + 0.21.0. The
`_install_mm_probe_skip_patch` monkey-patch in `vllm_serve.py` flips
`mm_config.skip_mm_profiling=True` at `profile_run` entry. The patch
fires `vllm.mm.probe_skipped` when active; absence of that event on a
cold-start means the patch lost its hook.

### vLLM CLI flag churn (auto-memory `vllm_cli_flag_churn`)

vLLM removes CLI flags between minor releases (PR #21739 dropped
`--disable-log-requests` in v0.10/v0.11). iter-10 added an argv
validator in `connectors/modal/app.py:_start_and_sleep` that imports
`vllm.entrypoints.openai.cli_args:make_arg_parser` directly and
asserts every `--xxx` in the spawn cmd is in the parser's action
list. Fires `vllm.argv.validated` on the happy path; `vllm.argv.unrecognized`
+ `RuntimeError` on a real flag rename, naming the offending flag in
the traceback.

### Region pin + multi-region scheduling

`_CLS_KWARGS["region"]` is `["us-east", "us-west"]`. Single-region
pinning (`"us-east-1"`) hit "waiting to be scheduled on GPU_H200
worker" when H200 capacity was tight; the multi-region form lets the
scheduler pick any available H200 zone within the listed groups.
QRNG entropy reaches every container via Cloudflare's global edge,
so there is no per-token latency penalty for staying inside the
US-only group.

### Container-restore tolerance for top-level code

`Modal._container_entrypoint` runs the entire module body of
`connectors/modal/app.py` to find the `@app.cls` definitions. Any
top-level code that depends on the deploy host's filesystem must
tolerate running inside container restore where only the package's
own files are present. `_qr_llm_chat_functions_dir()` is the canonical
example — gated on `MODAL_TASK_ID` being unset (deploy host) vs set
(container).

---

## Cleanup history

`connectors/modal/vllm_serve.py` was reduced from 997 → 157 lines in
iter-10 by retiring the pre-iter-07 architecture
(`build_engine`/`build_app`/`build_dispatcher_for` +
`_SECRET_DIAG_*` allow-list + `_emit_modal_secret_diag`). Surviving
helpers: `_install_mm_probe_skip_patch` (still invoked from
`engines/vllm.py` at LP-import time) and the bearer-auth utilities
(`_accepted_bearer_secrets` / `_verify_bearer` / `_check_vllm_api_key`)
which remain test-pinned for the future in-container auth-gate
restoration. The current `@modal.web_server` + `vllm serve` subprocess
deployment does NOT route through the bearer helpers — vLLM owns the
FastAPI app — but the helpers' rolling-secret semantics are pinned by
`tests/connectors/modal/test_vllm_serve_bearer.py` against the next
time an auth gate lands in front of the engine.

For the iter-by-iter narrative of how the Modal deploy went from
broken to green (iter-01..iter-11, 2026-05-21..22), see
`../Entropic-Science/qr-llm-chat/CLEANUP_REPORT.md` and the matching
git history. The integration contract above is the load-bearing record;
the iteration journey is in commits.
