# Modal profile — `chat.entropic.science`

Hosts the quantum-random LLM chatbot for [entropic.science](https://entropic.science).

| Component | Modal object | Hardware | Idle timeout |
|---|---|---|---|
| Inference (dual-model) | `VllmQr` | 1× B200 (~180 GB HBM3e) | 3 min, `max_containers=1` |
| Auth proxy | `owui_edge` | CPU | always-warm (`keep_warm=1`) |
| Chat UI | `owui` (Open WebUI) | CPU | always-warm (`keep_warm=1`) |

The `VllmQr` container runs **two `AsyncLLMEngine` siblings** in one
process, both FP8, both 64k context — Gemma 4 31B Reasoning (default) and
Qwen 3.6 27B Reasoning (selectable). The OWUI model selector exposes both
real models plus two `--qr-vs-prng` comparison-Pipe pseudo-models.

Cold-start optimisation: pre-baked weights in a `modal.Volume` named
`llm-weights` + `enable_memory_snapshot=True`. First request after deploy
pays the full ~60–120 s init cost (×2 engines); subsequent cold starts
restore from snapshot in ~10–15 s. `keep_warm` on `VllmQr` is **not** an
option — always-on B200 cost is unacceptable.

## First-time setup

### 1. Provision Modal secrets

See [`modal_secrets.md`](./modal_secrets.md). You need two secrets:
`qr-sampler-prod` (runtime config + the rolling-secret `SERVICE_TOKEN_SECRETS`
vector) and `hf-token` (Hugging Face token, only used by `download_weights`).

### 2. Populate the weights volume

Run the one-shot download function. Downloads ~31 GB Gemma 4 31B Reasoning
FP8 + ~27 GB Qwen 3.6 27B Reasoning FP8 to the `llm-weights` Volume.

```bash
modal run deployments/modal/app.py::download_weights
```

Takes 10–20 min on a fast HF mirror. Re-run only on model-version bump.

#### At-deploy-time model fallbacks

If either FP8 weight set is unavailable at deploy time, fall back to the
next-newest FP8 reasoning model in the same family and document the
substitution. Edit `_HF_REPO_FOR_MODEL` in `vllm_serve.py` and the matching
constants in `app.py`:

| Default | Fallback |
|---|---|
| `google/gemma-4-31b-reasoning` | `google/gemma-3-27b-reasoning-fp8` (or community FP8 conversion) |
| `Qwen/Qwen-3.6-27B-Reasoning` | `Qwen/Qwen-3-27B-Reasoning-FP8` |

If vLLM lacks a stable tag with B200 + FP8 KV + dual-engine support, pin
`VLLM_BASE_TAG` in `Dockerfile.vllm` to a nightly known-good commit and
record the SHA in this section.

### 3. First deploy

```bash
modal deploy deployments/modal/app.py
```

The first request hitting `VllmQr.serve` pays the full init cost and Modal
captures the snapshot afterwards. Watch the Modal dashboard for the
"snapshot ready" event before measuring cold-start times.

### 4. Wire the custom domain

```bash
modal domain create chat.entropic.science --function owui_edge
```

Then add a CNAME on `entropic.science`'s DNS zone pointing
`chat.entropic.science` → `<modal-generated-cname>.modal.run`. The same-cookie
scope is satisfied because `chat.entropic.science` is a subdomain of
`entropic.science`, and the session cookie has `Domain=.entropic.science`.

### 5. firefly-1 reachability

Pre-flight decision §11.1: **(A) public-with-firewall** accepted. Before the
first `VllmQr` request reaches firefly-1, add Modal's egress IP range to
the firefly-1 firewall. Read Modal's current egress range from
[Modal's docs](https://modal.com/docs) or `modal config show` and propagate
to the firewall rules at the firefly-1 host. Tailscale (option B) is
deferred to v2.

### 6. Import the two OWUI Global Functions

Open `https://chat.entropic.science` while signed into a test entropic.science
account (the auth-proxy will let you through once your session is valid).
Then in OWUI's admin UI:

1. **Admin Panel → Functions → Import** → upload
   `examples/open-webui/qr_sampler_filter.json`. Toggle it to **Global** so
   every chat goes through allowance preflight + debit.
2. **Admin Panel → Functions → Import** → upload
   `examples/open-webui/qr_comparison_pipe.json`. This Pipe registers two
   pseudo-models (`gemma-4-31b-reasoning--qr-vs-prng` and
   `qwen-3.6-27b-reasoning--qr-vs-prng`) in OWUI's model selector.

Both functions read their config from OWUI's Valves UI; the defaults pick up
`SERVICE_TOKEN_SECRETS` and `ENTROPIC_API_BASE_URL` from the container env.

A future iteration may automate this via OWUI's import API — for v1 it's a
one-time manual step.

## Cold-start expectations

| Scenario | Expected end-to-end latency |
|---|---|
| First request after `modal deploy` (snapshot capture) | 60–120 s |
| Cold start, snapshot restored | 10–15 s |
| Warm container | < 50 ms to first token |
| Cold start, snapshot fallback active | 30–45 s |

The 2× cost of dual-engine init only happens during the **first** request
after a deploy (when Modal is recording the snapshot). Snapshot restore
already includes both engines, so subsequent cold starts pay the same
~10–15 s regardless of dual vs single model.

## Snapshot-failure fallback

Per Pre-flight §11.7: if memory-snapshot restore fails on B200 (CUDA-context
corruption or image-base incompatibility), drop to **pre-baked weights
without snapshots**:

1. Edit `app.py`: set `enable_memory_snapshot=False` on `VllmQr`.
2. `modal deploy deployments/modal/app.py`.
3. Cold starts are now ~30–45 s (weights still pre-baked, just no snapshot
   restore).

**Do NOT set `keep_warm=1` on `VllmQr`.** Always-on B200 cost is
unacceptable. If both snapshot and pre-baked-weight paths fail, escalate to
the user before reaching for warm-pool config.

## Invalidating the snapshot

The snapshot is keyed on the container image + the `@modal.enter(snap=True)`
method's effect. To force a fresh snapshot:

- Bumping any of these regenerates the snapshot automatically:
  - Anything in the `vllm_image` build (Dockerfile.vllm, qr-sampler source).
  - Anything in `vllm_serve.py`.
  - Either model's revision (re-run `download_weights` with new
    `GEMMA_REVISION` / `QWEN_REVISION` env vars first).
- To force regeneration without source changes:
  ```bash
  modal app stop qr-sampler-entropic
  modal volume rm llm-weights         # only if you want fresh weights too
  modal run deployments/modal/app.py::download_weights
  modal deploy deployments/modal/app.py
  ```

## Rolling-secret rotation (`SERVICE_TOKEN_SECRETS`)

See [`modal_secrets.md`](./modal_secrets.md) for the full procedure. TL;DR:

1. **Prepend** new secret to `SERVICE_TOKEN_SECRETS` on both Modal and
   entropic.science.
2. **Redeploy at leisure** — no lockstep required.
3. **Remove old** on the next routine deploy.

The signer always uses the first entry; the verifier accepts any. Adding a
new secret never breaks live traffic.

## Verification (manual, no CI in v1)

Acceptance criteria from `spec.md` §10.3:

1. `modal deploy deployments/modal/app.py` succeeds (no build / push errors).
2. `curl https://chat.entropic.science/healthz` returns `{"status":"ok"}` (the
   `owui_edge` proxy health endpoint).
3. Sign in with a test entropic.science account, send a prompt through OWUI,
   receive a response, and confirm a `debitAllowance` row appears in the dev
   DB's `allowance_ledger` table.
4. Switch to the `gemma-4-31b-reasoning--qr-vs-prng` model and observe two
   parallel columns of generation + a double-cost debit (one ledger row
   with `comparison_mode = true`).
5. Kill the warm container (or wait 4 min idle), send a new prompt, and
   verify Modal logs show "restored from snapshot" with end-to-end response
   in ≤ 20 s.

## Operations reference

| Action | Command |
|---|---|
| Deploy | `modal deploy deployments/modal/app.py` |
| One-shot weights download | `modal run deployments/modal/app.py::download_weights` |
| Tail logs (vLLM) | `modal app logs qr-sampler-entropic --function VllmQr` |
| Tail logs (auth proxy) | `modal app logs qr-sampler-entropic --function owui_edge` |
| Stop the app | `modal app stop qr-sampler-entropic` |
| Update a secret | `modal secret update qr-sampler-prod KEY=value` |
| List Modal volumes | `modal volume list` |

## Local-dev parity (optional)

A `deployments/entropic-science/` Compose profile mirrors this Modal env so
the stack can be smoke-tested locally against firefly-1 before pushing to
Modal. Skippable if firefly-1 firewall rules are already keyed to Modal's
egress IP range and you're confident in the Modal deploy.

## Source-of-truth design

- `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/spec.md` §1.3,
  §5.5, §5.6, §5.7.
- `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/requirements.md`
  §10 (resolved decisions).
- Cross-repo handshake: [`../../CROSS-REPO-INTEGRATION.md`](../../CROSS-REPO-INTEGRATION.md).
