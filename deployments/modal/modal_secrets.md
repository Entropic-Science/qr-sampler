# Modal secrets — `qr-sampler-prod` and `hf-token`

Two Modal Secrets back this deployment. Provision them once per Modal
workspace; keys can be rotated independently afterwards.

The OWUI surface itself now lives on the entropic.science Replit deployment
(see `entropic.science/spec.md §3.2`), so OWUI-specific env vars
(`WEBUI_AUTH`, `WEBUI_AUTH_TRUSTED_*`, `OPENAI_API_BASE_URL`, `OPENAI_API_KEY`,
`OWUI_UPSTREAM_URL`, `ENABLE_SIGNUP`, …) are NOT held in Modal Secrets —
they're set as Replit Secrets on the entropic.science deployment per
`entropic.science/spec.md §12.5`.

## `qr-sampler-prod`

Holds runtime configuration for `VllmQr`. Created once and mounted into the
GPU container (see `app.py`).

```bash
modal secret create qr-sampler-prod \
  QR_ENTROPY_SOURCE_TYPE=quantum_grpc \
  QR_PREINIT_ENTROPY_SOURCES=quantum_grpc,system \
  QR_FALLBACK_MODE=error \
  QR_SAMPLE_COUNT=13312 \
  QR_GRPC_SERVER_ADDRESS=10.0.0.115:50051 \
  QR_GRPC_MODE=unary \
  QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes \
  QR_GRPC_STREAM_METHOD_PATH= \
  QR_GRPC_API_KEY=<firefly-1 api key> \
  QR_GRPC_API_KEY_HEADER=api-key \
  QR_GRPC_TIMEOUT_MS=5000 \
  VLLM_MODELS=gemma-4-31b-reasoning,qwen-3.6-27b-reasoning \
  VLLM_DEFAULT_MODEL=gemma-4-31b-reasoning \
  VLLM_MAX_MODEL_LEN=65536 \
  VLLM_GPU_MEMORY_UTILIZATION_PER_ENGINE=0.45 \
  ENTROPIC_API_BASE_URL=https://entropic.science/api \
  SERVICE_TOKEN_SECRETS=<random 32-byte base64>
```

### v1 (prototyping): system entropy until WARP-in-Modal lands

firefly-1 at `10.0.0.115:50051` is only reachable via the `cipherstone`
Cloudflare WARP tunnel. Modal containers can't reach RFC1918 addresses
without a WARP client *inside* the container — see the future-work block
in `Dockerfile.vllm`. The initial deploy therefore ships with
`QR_ENTROPY_SOURCE_TYPE=system` so chat works end-to-end without QRNG.
firefly-1 creds stay in the Secret (parked) so the future flip is a
single-line `modal secret update`:

```bash
# Override on first deploy:
modal secret update qr-sampler-prod \
  QR_ENTROPY_SOURCE_TYPE=system \
  QR_PREINIT_ENTROPY_SOURCES=system,quantum_grpc \
  QR_FALLBACK_MODE=fallback

# Once WARP-in-Modal is wired up, flip back to:
modal secret update qr-sampler-prod \
  QR_ENTROPY_SOURCE_TYPE=quantum_grpc \
  QR_PREINIT_ENTROPY_SOURCES=quantum_grpc,system \
  QR_FALLBACK_MODE=error
```

Note: in v1, the two `--qr-vs-prng` comparison-mode pseudo-models on
Replit's `MODELS_FILTERED` will fall back to system entropy on the
"quantum" side (because `QR_FALLBACK_MODE=fallback`), so comparison
mode is honest-but-uninteresting until firefly-1 is reachable. If you
want comparison-mode requests to hard-fail instead of silently fall
back, omit the two pseudo-model entries from Replit's
`MODELS_FILTERED` for v1.

Generate random values:

```bash
openssl rand -base64 32   # SERVICE_TOKEN_SECRETS entries
```

## `hf-token`

Held in a separate Secret because it's only needed by the one-shot
`download_weights` function — `VllmQr` should never have HF credentials at
request time.

```bash
modal secret create hf-token HF_TOKEN=<huggingface token>
```

## Rolling-secret rotation for `SERVICE_TOKEN_SECRETS`

`SERVICE_TOKEN_SECRETS` is a **comma-separated vector** (Pre-flight §11.4).
Three readers share one vector:

- **entropic.science api-server** — verifies `X-Service-Token` HMACs from the
  qr-sampler OWUI filter/pipe at `preflightAllowance` / `debitAllowance`.
- **The qr-sampler OWUI plugin** (`examples/open-webui/qr_sampler_filter.py`
  + `qr_comparison_pipe.py`) — signs `X-Service-Token` headers and uses the
  first entry as `Authorization: Bearer` for the Modal bearer.
- **`VllmQr` bearer verifier** in `deployments/modal/vllm_serve.py` —
  `_verify_bearer` accepts a constant-time `compare_digest` match against
  ANY entry.

The signer always uses the FIRST entry; every verifier accepts any. This
removes lockstep redeploy pain — adding a new secret never breaks live
traffic.

Procedure when rotating:

1. **Prepend** the new secret on every side:
   - Modal: `modal secret update qr-sampler-prod SERVICE_TOKEN_SECRETS=<new>,<old>`
   - entropic.science Replit Secret: `SERVICE_TOKEN_SECRETS=<new>,<old>` (same value).
2. **Redeploy at leisure** — either side first is fine. While both old and
   new are live, requests signed under either secret are accepted.
3. After both deploys have settled and you have verified the new secret is
   actually being used by traffic:
   - Modal: `modal secret update qr-sampler-prod SERVICE_TOKEN_SECRETS=<new>`
   - entropic.science Replit: same.
   - Redeploy both. Old secret is now removed.

No automated rotation in v1. Re-run this procedure once per quarter or
after any suspected leak.

## Unauthenticated-inference opt-in (`ALLOW_UNAUTHENTICATED_INFERENCE`)

By default the inference container **fails closed** when
`SERVICE_TOKEN_SECRETS` is unset — every request is rejected with `503` and
a configuration-error message. This guards against accidentally exposing the
GPU endpoint to the public internet if the secret slot is provisioned blank.

For smoke-tests / local-Modal experiments where bearer auth is genuinely
unwanted, set the explicit escape hatch:

```bash
ALLOW_UNAUTHENTICATED_INFERENCE=1
```

Only the literal string `1` opts in — `true`, `0`, `yes`, etc. all keep the
fail-closed behavior. If `SERVICE_TOKEN_SECRETS` is set, this flag is
ignored (bearer verification always runs when a secret is provisioned).

Never set this in production. There is no fallback path to permanent
unauthenticated inference.

## Bumping the Modal-facing URL after first deploy

`VllmQr.serve` is exposed at `https://<workspace>--qr-sampler-entropic-vllm-qr-serve.modal.run`
(or whatever Modal assigns). Record the URL after the first deploy and set
it as Replit Secrets on the entropic.science deployment:

- `OPENAI_API_BASE_URL=https://<workspace>--qr-sampler-entropic-vllm-qr-serve.modal.run/v1`
- `MODAL_BASE_URL=https://<workspace>--qr-sampler-entropic-vllm-qr-serve.modal.run` (no `/v1` suffix; used by `POST /api/inference/warm`)
- `OPENAI_API_KEY=<first entry of SERVICE_TOKEN_SECRETS>`

Then restart the api-server + open-webui artifacts on Replit.
