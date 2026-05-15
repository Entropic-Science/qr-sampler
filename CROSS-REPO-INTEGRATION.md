# Cross-Repo Integration — entropic.science chatbot

**Status**: Planned. Implementation has not started yet.
**Other repo**: `C:\Code\Entropic-Science\entropic.science\`
**Source-of-truth artefacts**: `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/` (PRD `requirements.md`, spec `spec.md`, plan `plan.md`).
**Last updated**: 2026-05-15 (Pre-flight decisions).

This file exists so a second Claude agent working in `qr-sampler` knows what is about to land here from the `entropic.science` side. It is a heads-up, not a directive. The authoritative design lives in `spec.md` in the other repo.

---

## ⚠ Heads-up — Pre-flight decisions on 2026-05-15

The user resolved the eight implementation assumptions from spec §11. **Four are material deviations from the version of this document captured just before** and have already been propagated inline below. Summary so anyone reading top-to-bottom catches them at a glance:

1. **Two models, not one.** Default: **Gemma 4 31B (Reasoning) FP8**. Selectable alternative: **Qwen 3.6 27B (Reasoning) FP8**. Both served from one `@app.cls` on one B200 as two `AsyncLLMEngine` siblings, dispatched on the request's `model` field. The OWUI model selector shows both. Comparison mode runs per-base-model and never mixes models across columns. (Previously this document said "Gemma 3 27B FP8 (fallback Gemma 2 27B FP8)".)
2. **Streaming everywhere, including comparison mode.** The comparison Pipe opens two `stream=true` SSEs in parallel and multiplexes them into one OWUI SSE response with the full dual-column markdown re-emitted per delta tick. Outlet debit fires on stream completion. (Previously this document said "Non-streaming in v1. Streaming dual-column is a v2 follow-up.")
3. **`max_containers=1`**, not 2. Up to 8 in-flight requests per container before queueing.
4. **Rolling-secret rotation for `SERVICE_TOKEN_SECRETS`** (note the plural). The site's `lib/serviceToken.ts` reads a comma-separated vector of accepted secrets; signs with the first entry, verifies against any. Rotation = prepend new → redeploy at leisure → remove old next deploy. The `examples/open-webui/qr_sampler_filter.py` Valves field is also plural and follows the same pattern (signs with the first secret if multiple are passed in).

Two further decisions that don't change this document materially but are worth knowing:
- §11.1 (firefly-1 reachability) — **(A) public-with-firewall accepted**. The entropic.science side will add Modal's egress IP range to the firefly-1 firewall before first deploy. Tailscale (B) deferred to v2.
- §11.7 (snapshot effectiveness on B200) — `enable_memory_snapshot=True` ships as default. Fallback on snapshot-restore failure is **pre-baked weights without snapshots** (~30–45 s cold start). **`keep_warm=1` on `VllmQr` is explicitly NOT a fallback** — escalate before reaching for warm pools.

Authoritative details live in `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/spec.md` §1.3, §5.4, §5.5, §5.6, §11 (now annotated "RESOLVED") and the matching plan.md Pre-flight step.

---

## 1. What is being built

The `entropic.science` site is launching its first app — a **quantum-random LLM chatbot** powered by qr-sampler. The user-facing surface is Open WebUI at `chat.entropic.science`. **Two reasoning-tuned models** are served from a single `@app.cls` on one Modal B200 GPU: **Gemma 4 31B (Reasoning) FP8** as the default and **Qwen 3.6 27B (Reasoning) FP8** as a selectable alternative. Both run via **vLLM**, each its own `AsyncLLMEngine`, with qr-sampler doing per-token entropy-driven sampling on whichever engine handles the request.

Two unusual product requirements drive the qr-sampler-side changes:

1. **Comparison mode** — a user-toggleable mode that runs the *same prompt, same model, same sampling params* twice, once with `quantum_grpc` entropy and once with `system` entropy, and renders both completions side-by-side. Goal: let users see whether the entropy source materially affects generation.
2. **Daily allowance metering** — `entropic.science` has its own auth + a 128k-weighted-tokens-per-day free allowance per account. The Open WebUI filter is the integration point that calls `entropic.science/api` to preflight and debit per-request.

---

## 2. Concrete qr-sampler changes that are planned

These are the files we expect to touch. Group by area:

### 2.1 Library — per-request entropy-source override

The only library-level change. Small (~50 LoC) but load-bearing — it's what makes comparison mode possible without renting a second GPU.

- **`src/qr_sampler/config.py`** — add `entropy_source_type` to `_PER_REQUEST_FIELDS`. Today it's a startup-only field; we need it overridable per request so the OWUI Pipe can fire two completions with different sources against the same vLLM instance.
- **`src/qr_sampler/entropy/registry.py`** — add an `all_sources()` helper used by `VLLMAdapter` during pre-init.
- **`src/qr_sampler/engines/vllm.py`**:
  - At adapter construction (called from `@modal.enter(snap=True)`, see §2.4), build a `dict[str, SamplingPipeline]` keyed by entropy-source-type, one entry per source listed in env `QR_PREINIT_ENTROPY_SOURCES` (default: `"quantum_grpc,system"`).
  - `update_state(req)` reads `qr_entropy_source_type` from `extra_args`; defaults to env `QR_ENTROPY_SOURCE_TYPE` if absent; rejects unknown / un-preinit'd values with a clean 400-shaped error.
  - `apply(logits)` looks up the per-request pipeline by source key and runs `sample_token()`. **The just-in-time invariant (entropy fetched *after* logits are computed) is preserved.**

New env var to document in the existing `engines/vllm.py` env contract section:
- `QR_PREINIT_ENTROPY_SOURCES` — comma-separated list of sources to pre-initialise. Default `"quantum_grpc,system"`. Operators wanting only one source set this to e.g. `"quantum_grpc"`.

### 2.2 OWUI filter — extend `examples/open-webui/qr_sampler_filter.py`

Keep all current `qr_*` parameter-injection behaviour. Add:

1. **Inlet** — read `__user__["email"]`, call `entropic.science/api/allowance/preflight` (signed with an HMAC service token, see §3), reject below threshold with markdown that renders next-refill-time + a `[Register interest →]` link to the site's waitlist page. Detects `qr_comparison_mode` in body metadata so the preflight gate uses the doubled cost.
2. **Outlet** — POST `prompt_tokens` + `completion_tokens` to `/api/allowance/debit`, then POST a thin record `(owui_chat_id, title, last_message_at, comparison_mode_used, weighted_tokens_total)` to `/api/conversations/upsert` so the site can show a per-account chat history list across devices.

New `Valves` (filter-config) fields:
- `api_base_url` (default `https://entropic.science/api`)
- `service_token_secret` (from env, never logged)
- `min_reserved_output_tokens` (default 128)
- `request_timeout_s` (default 5.0)

The existing `qr_sampler_filter.json` export needs to be regenerated.

### 2.3 OWUI Pipe — new file `examples/open-webui/qr_comparison_pipe.py` (+ `.json` export)

A new **manifold Pipe** whose `pipes()` returns one entry per base model — `gemma-4-31b-reasoning--qr-vs-prng` (default) and `qwen-3.6-27b-reasoning--qr-vs-prng`. Each entry, when selected from OWUI's model selector, runs the fan-out against the matching real base model. **Streaming dual-column** (Pre-flight decision 2026-05-15 — streaming everywhere). Behaviour:

1. Preflight allowance with `comparisonMode=True` (so the gate uses 2× cost).
2. Open two parallel `httpx.AsyncClient.stream("POST", ...)` to the same vLLM endpoint, both with `stream: true`. Bodies mirror the user's request but with `model` rewritten to the real base model (e.g. `gemma-4-31b-reasoning`), and `extra_body={"qr_entropy_source_type": "quantum_grpc"}` on one side, `{"qr_entropy_source_type": "system"}` on the other. All sampling params (`seed`, `temperature`, `top_p`, `top_k`, `max_tokens`) are forced identical between the two so only the entropy source differs.
3. Concurrent stream readers append to two in-memory buffers `L_buf` and `R_buf`. On each delta from either side, yield one OWUI SSE chunk whose `choices[0].delta.content` is the **full current dual-column markdown table** re-rendered with the latest buffers. (Re-emitting the full markdown each tick is the simplest sound rendering against OWUI's stream protocol; message size stays small.)
4. After both streams finish (or one fails — see error handling), yield a final SSE chunk with the complete markdown plus a `usage` block containing summed `prompt_tokens` + `completion_tokens`. Then debit once with `comparisonMode=true` and the summed token counts; upsert the conversation record once.
5. Error handling: if one side errors mid-stream, surface that side's error inline in its column ("[stream error: <code>]") and let the other side finish. Debit reflects tokens actually produced.

### 2.4 Modal deployment — new profile `deployments/modal/`

A new sibling to the existing `deployments/firefly-1/` Compose profile. **B200 GPU, dual-model, memory snapshots, 3-min idle, `max_containers=1`.**

Files:
- `app.py` — Modal entrypoint, declares five top-level objects:
  - `weights_volume = Volume.from_name("llm-weights", create_if_missing=True)` mounted at `/root/.cache/huggingface`. (Renamed from `gemma-weights` to reflect dual-model housing.)
  - `download_weights` — one-shot `@app.function` that downloads **both** model directories sequentially: `huggingface_hub.snapshot_download("google/gemma-4-31b-reasoning", revision=<pinned sha>)` and `huggingface_hub.snapshot_download("Qwen/Qwen-3.6-27B-Reasoning", revision=<pinned sha>)`. Invoked as `modal run deployments/modal/app.py::download_weights` once per model-version bump.
  - `VllmQr` — `@app.cls(gpu="B200", region="us-east-1", volumes={"/root/.cache/huggingface": weights_volume}, enable_memory_snapshot=True, container_idle_timeout=180, allow_concurrent_inputs=8, max_containers=1)`. The init method is decorated `@modal.enter(snap=True)` so Modal captures the post-init state to a snapshot. The init method constructs **two** vLLM `AsyncLLMEngine` siblings in the same process (one per real base model), and pre-initialises both qr-sampler entropy pipelines (`quantum_grpc` + `system`) **for each engine** — four warmed `(engine, source)` pipelines total. The ASGI dispatcher routes `POST /v1/chat/completions` on `request.json()["model"]` to the matching engine's router.
  - `owui_edge` — small FastAPI auth-proxy that reads the entropic.science session cookie, validates it, and injects trusted-user headers before proxying to OWUI.
  - `owui` — stock Open WebUI in trusted-header auth mode.
- `vllm_serve.py` — module imported by `VllmQr.load`. Constructs two engines:
  - Engine A: `--model google/gemma-4-31b-reasoning --served-model-name gemma-4-31b-reasoning --quantization fp8 --kv-cache-dtype fp8 --max-model-len 65536 --gpu-memory-utilization 0.45`
  - Engine B: `--model Qwen/Qwen-3.6-27B-Reasoning --served-model-name qwen-3.6-27b-reasoning --quantization fp8 --kv-cache-dtype fp8 --max-model-len 65536 --gpu-memory-utilization 0.45`
  - (0.45 + 0.45 = 0.90 total, mirroring the prior single-model budget on the 180 GB B200.)
- `owui_edge.py` — FastAPI auth-proxy.
- `Dockerfile.vllm` — base `vllm/vllm-openai:<tag with B200 + FP8 KV support>`, `pip install qr_sampler huggingface_hub`.
- `Dockerfile.owui` — base `ghcr.io/open-webui/open-webui:<pinned>`, `pip install httpx`.
- `.env.example` — every env var from §3 below.
- `modal_secrets.md` — secret-provisioning steps.
- `README.md` — full deploy walkthrough (provision secrets → `modal run download_weights` → `modal deploy` → first request pays the full init cost and produces the snapshot → subsequent cold starts are ~10–15 s).

**Critical snapshot integrity invariants** (these constrain how qr-sampler init code is written):

- **No live gRPC channel captured in the snapshot.** The `quantum_grpc` source already uses lazy channel creation (good). Do not "optimise" by moving channel construction into `__init__` — it would freeze a dead socket into the snapshot, and the first post-restore request would fail.
- **No process-relative state captured.** Avoid `os.getpid()`-based caches, in-process locks, or anything that assumes the process started fresh.
- **Secrets are mounted after restore** — Modal honours this automatically. The qr-sampler config layer reads env at construction time, which is fine *as long as* construction happens inside the `@modal.enter(snap=True)` method, not at module import.

### 2.5 Optional alias profile `deployments/entropic-science/`

A docker-compose profile that mirrors the Modal env so the stack can be smoke-tested locally against firefly-1 before deploying to Modal. Skippable if local parity isn't useful.

### 2.6 `examples/open-webui/README.md`

Document the new entropic.science integration: which env vars the filter + pipe need, how to install both Global Functions into OWUI on first deploy, where the API contract lives.

---

## 3. Env contract (binding inputs from the entropic.science side)

These are the env vars the Modal `VllmQr` container must have set. Group by purpose:

| Group | Var | Value | Notes |
|---|---|---|---|
| Entropy | `QR_ENTROPY_SOURCE_TYPE` | `quantum_grpc` | Default for non-comparison requests. |
| Entropy | `QR_PREINIT_ENTROPY_SOURCES` | `quantum_grpc,system` | NEW env, pre-loads both pipelines into the snapshot. |
| gRPC | `QR_GRPC_SERVER_ADDRESS` | `10.0.0.115:50051` | firefly-1 gRPC server. RFC 1918 — see §4. |
| gRPC | `QR_GRPC_MODE` | `unary` | firefly-1 supports unary only. |
| gRPC | `QR_GRPC_METHOD_PATH` | `/qrng.QuantumRNG/GetRandomBytes` | firefly-1 proto path. |
| gRPC | `QR_GRPC_STREAM_METHOD_PATH` | `""` | Disable streaming explicitly. |
| gRPC | `QR_GRPC_API_KEY` | from Modal Secret `firefly1-api-key` | |
| gRPC | `QR_GRPC_API_KEY_HEADER` | `api-key` | firefly-1 metadata header name. |
| Sample sizing | `QR_SAMPLE_COUNT` | `13312` | firefly-1's max per request (vs qr-sampler default 20480). |
| Fallback | `QR_FALLBACK_MODE` | `error` | Disable silent fallback to system — the comparison product depends on this. |
| Circuit | `QR_CB_*` | defaults | P99 × 1.5; 3-failure trip. |
| vLLM | `VLLM_API_KEY` | from Modal Secret `vllm-api-key` | owui uses this to call vllm internally. |
| Models | `VLLM_MODELS` | `gemma-4-31b-reasoning,qwen-3.6-27b-reasoning` | Comma-separated served-model-names. Engine init iterates this list to construct two `AsyncLLMEngine` siblings. |
| Models | `VLLM_DEFAULT_MODEL` | `gemma-4-31b-reasoning` | Default model in the OWUI selector. |
| Allowance | `ENTROPIC_API_BASE_URL` | `https://entropic.science/api` | filter + pipe consume. |
| Allowance | `SERVICE_TOKEN_SECRETS` | from Modal Secret `qr-sampler-prod` | **Plural, comma-separated** rolling-secret vector (Pre-flight §11.4). HMAC keys for `X-Service-Token`. |

Service-token format (filter and pipe both sign their requests):
- Header: `X-Service-Token: <unix_ts>.<hmac>`
- `hmac = HMAC-SHA256(<secret>, unix_ts + path)` — signer always uses the **first** entry of `SERVICE_TOKEN_SECRETS`; the entropic.science API verifier accepts a match against **any** entry. This is the rolling-secret rotation posture (Pre-flight §11.4): prepend new secret → redeploy at leisure → remove old next deploy. No lockstep redeploy required.
- 60s timestamp window enforced server-side.

---

## 4. Network shape (firefly-1 reachability)

firefly-1 lives at `10.0.0.115:50051` (RFC 1918). For Modal to reach it, two options were considered:

- **(A) Public exposure with TLS + the existing API-key auth + a firewall rule that only allows Modal's egress IP range.** ✅ **Resolved 2026-05-15 — (A) accepted.** The entropic.science side (the operator of firefly-1) will add Modal's egress IP range to the firefly-1 firewall before first deploy. No Tailscale sidecar work expected.
- **(B) Tailscale sidecar on the Modal container + a Tailscale node on the firefly-1 host.** Deferred to v2 hardening track.

---

## 5. What we expect the other agent NOT to change while this is in flight

To avoid merge surprises, please avoid the following while this work is in flight (or coordinate explicitly):

1. **The shape of `_PER_REQUEST_FIELDS` in `src/qr_sampler/config.py`** — we are about to add `entropy_source_type` to it. Concurrent changes risk a messy merge.
2. **`src/qr_sampler/engines/vllm.py` adapter init signature** — we are about to wire a `dict[str, SamplingPipeline]` of pre-initialised pipelines into adapter construction.
3. **`src/qr_sampler/entropy/quantum.py` channel construction** — keep it lazy (do not move `grpc.aio.insecure_channel` into `__init__`). Memory snapshots in §2.4 depend on this.
4. **`examples/open-webui/qr_sampler_filter.py` Valves shape** — we are adding four new `Valves` fields (`api_base_url`, `service_token_secret`, `min_reserved_output_tokens`, `request_timeout_s`). If you need to add more, please coordinate so we don't both rewrite the same file.

Everything else (new entropy sources, new engine adapters, statistical-test refactors, new profiles outside `deployments/modal/` and `deployments/entropic-science/`, CLI changes, doc rewrites elsewhere) is free territory and shouldn't conflict.

---

## 6. Verification commands the entropic.science side will run

When this lands, we'll verify from `C:\Code\qr-sampler\` with:

```bash
ruff format --check src/ tests/
ruff check src/ tests/
mypy --strict src/
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing   # 90% gate must hold
```

New tests we expect to add:

- `tests/test_engines/test_vllm_adapter_per_request_source.py` — two requests in one batch with different `qr_entropy_source_type`; assert each used the correct source. Plus rejection of un-preinit'd source values.
- `tests/test_open_webui_integration.py` — HTTP mock for entropic.science API; assert filter inlet+outlet sequence under (sufficient, insufficient, email-not-verified, model-unavailable) scenarios.
- Pipe tests under `tests/test_open_webui/test_qr_comparison_pipe.py` — pipe fans out two parallel calls, aggregates dual-column markdown, sends one comparison-flagged debit, honours preflight insufficient branch (does not invoke vLLM).

---

## 7. Open questions for the other agent

If you have opinions on any of these, please leave a note in this file (or a PR comment) and we'll fold them in:

1. **Should the comparison-mode adapter live in the OWUI Pipe (current plan) or as a new qr-sampler engine adapter?** Spec leaves the Pipe-layer fan-out as v1, **and the Pipe must stream** (Pre-flight §11.5 — streaming everywhere). If you see a reason to move the fan-out into qr-sampler's engine layer (e.g. latency benchmarks suggest the filter-layer fan-out adds unacceptable overhead even with streaming), please flag.
2. **Is `QR_PREINIT_ENTROPY_SOURCES` the right env-var name?** Open to better names. We picked it for symmetry with `QR_ENTROPY_SOURCE_TYPE`.
3. **OWUI version pin** — any preference for a specific Open WebUI tag known to work cleanly with trusted-header auth + Global Functions + manifold Pipes that stream?
4. **Pipe placement** — does `examples/open-webui/qr_comparison_pipe.py` feel like the right home, or should it live under `examples/open-webui/pipes/`? We'll match whatever convention you set.
5. **Manifold Pipe `pipes()` registration** — the comparison Pipe must register one entry per real base model (so it lights up in OWUI's selector once per model). We're planning to drive that off `VLLM_MODELS` env. If you'd prefer the Pipe declare its own static list and the operator update it on model changes, push back and we'll switch.

---

## 8. Where the source-of-truth design lives

- **Product requirements**: `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/requirements.md`
- **Technical spec** (full design): `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/spec.md` — sections §1.3, §5, §6 are the qr-sampler-relevant ones.
- **Implementation plan**: `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/plan.md` — the four qr-sampler steps are "qr-sampler library — per-request entropy-source override", "qr-sampler OWUI filter", "qr-sampler OWUI comparison Pipe", and "Modal deployment".

If you can't see those files (different machine), ping the entropic.science side and we'll mirror the relevant excerpts here.
