# Modal profile — `VllmQr` GPU inference

Modal-side surface for the quantum-random LLM chatbot at
[entropic.science](https://entropic.science). This profile now hosts **only
the GPU inference container**. Open WebUI itself and its auth bridge live on
the entropic.science Replit deployment (see
`entropic.science/spec.md §3.2` — the `artifacts/open-webui/` artifact and
`middlewares/owuiAuthBridge.ts`).

| Component | Modal object | Hardware | Idle timeout |
|---|---|---|---|
| Inference (dual-model) | `VllmQr` | 1× B200 (~180 GB HBM3e) | 3 min, `max_containers=1` |

Topology, end-to-end:

```
Browser ── HTTPS ──▶ entropic.science (Replit Reserved VM)
                        │
                        ├── /api      → api-server artifact
                        ├── /chat     → owuiAuthBridge → open-webui artifact (127.0.0.1:8081)
                        └── (other)   → static-site artifact

  open-webui ── HTTPS ─▶ VllmQr.serve  (this Modal app)
                          │  https://<workspace>--qr-sampler-entropic-vllm-qr-serve.modal.run/v1
                          ▼
                         B200 GPU, dual AsyncLLMEngine, FP8 weights + FP8 KV
```

The `VllmQr` container runs **two `AsyncLLMEngine` siblings** in one
process, both FP8, both 64k context — Gemma 4 31B Reasoning (default) and
Qwen 3.6 27B Reasoning (selectable). The OWUI model selector (configured
inside the Replit-hosted OWUI artifact) exposes both real models plus two
`--qr-vs-prng` comparison-Pipe pseudo-models.

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

#### Model-substitution procedure

To change the served model, edit `VllmQrQwen.HF_REPO_ID` +
`VllmQrQwen.SERVED_MODEL_NAME` in `app.py`. The downloader (`download_weights`)
reads `HF_REPO_ID` directly; vLLM serve consumes it as the positional
`model_tag` argument. Older revisions of this README documented a
multi-model `_HF_REPO_FOR_MODEL` table that lived in `vllm_serve.py`
alongside `build_engine`; the per-model `@app.cls` rewrite + iter-10
cleanup removed both. Gemma is paused (see `app.py` module docstring)
until vLLM ships a release with both Gemma4 GDN support AND a workable
text-only MM-probe knob; restoring it is a code-only change.

`VLLM_BASE_TAG` is pinned to `v0.17.0` (V1 engine release with the
LP ABI the qr-sampler `VLLMAdapter` is built against).

If you need to roll the pin forward, prefer pinning to a vLLM tag whose
bundled transformers already knows the model architectures (which removes
the `--no-deps` override). Until then, bump the SHA pinned at the end of
the `transformers @ git+...@main` URL when a regression is suspected.

### 3. First deploy

```bash
modal deploy deployments/modal/app.py
```

The first request hitting `VllmQr.serve` pays the full init cost and Modal
captures the snapshot afterwards. Watch the Modal dashboard for the
"snapshot ready" event before measuring cold-start times.

Record the Modal-assigned URL (e.g. `https://<workspace>--qr-sampler-entropic-vllm-qr-serve.modal.run`)
and propagate it to the entropic.science Replit deployment as
`OPENAI_API_BASE_URL` (with `/v1` suffix) and `MODAL_BASE_URL` (no suffix).
See `entropic.science/spec.md §12.5` for the consolidated env-var list.

### 4. firefly-1 reachability

Pre-flight decision §11.1: **(A) public-with-firewall** accepted. Before the
first `VllmQr` request reaches firefly-1, add Modal's egress IP range to
the firefly-1 firewall. Read Modal's current egress range from
[Modal's docs](https://modal.com/docs) or `modal config show` and propagate
to the firewall rules at the firefly-1 host. Tailscale (option B) is
deferred to v2.

### 5. OWUI Global Functions (Replit-side, not Modal-side)

The two OWUI Global Functions (`qr_sampler_filter.py` and
`qr_comparison_pipe.py`) are imported into the **Replit-hosted** OWUI
artifact, not into a Modal container. See
`entropic.science/spec.md §12.6` (operator runbook) and
`qr-sampler/examples/open-webui/README.md` for the operator steps.

Both functions read their config from OWUI's Valves UI; the defaults pick up
`SERVICE_TOKEN_SECRETS` and `ENTROPIC_API_BASE_URL` from the OWUI container
env. The first entry of `SERVICE_TOKEN_SECRETS` is the signer; this Modal
app's `_verify_bearer` accepts any entry.

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

The branded cold-start indicator that users see in OWUI is emitted by the
qr-sampler OWUI plugin (`examples/open-webui/qr_sampler_filter.py` +
`qr_comparison_pipe.py`), gated on the `QR_INTEGRATION_PROFILE` env var —
not by this Modal app. See `examples/open-webui/README.md`.

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
new secret never breaks live traffic. `_verify_bearer` in `vllm_serve.py`
honours this contract symmetrically with the entropic.science api-server's
HMAC verifier.

## Verification (manual, no CI in v1)

Acceptance criteria from `entropic.science/spec.md §8.4`:

1. `modal deploy deployments/modal/app.py` succeeds (no build / push errors).
2. From the Replit deployment: `curl https://entropic.science/api/owui/upstream-health`
   returns `{"status":"ok"|"starting","checkedAt":"..."}`.
3. Sign in to entropic.science with a test account, click **Launch** on the
   quantum-random-llm app, send a prompt through OWUI, receive a response,
   and confirm a `debitAllowance` row appears in the dev DB's
   `allowance_ledger` table.
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
| Stop the app | `modal app stop qr-sampler-entropic` |
| Update a secret | `modal secret update qr-sampler-prod KEY=value` |
| List Modal volumes | `modal volume list` |

## Local-dev parity (optional)

A `deployments/entropic-science/` Compose profile mirrors this Modal env so
the stack can be smoke-tested locally against firefly-1 before pushing to
Modal. Skippable if firefly-1 firewall rules are already keyed to Modal's
egress IP range and you're confident in the Modal deploy.

## Source-of-truth design

- `entropic.science/.zenflow/tasks/qr-sampler-app-ui-ad88/spec.md` §3.2,
  §4.1, §4.4, §12.
- `entropic.science/.zenflow/tasks/qr-sampler-app-ui-ad88/requirements.md`
  (NFR-COST-1, NFR-SEC-*).
- Cross-repo handshake: [`../../CROSS-REPO-INTEGRATION.md`](../../CROSS-REPO-INTEGRATION.md).
