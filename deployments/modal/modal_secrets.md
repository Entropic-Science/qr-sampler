# Modal secrets — `qr-sampler-prod` and `hf-token`

Two Modal Secrets back the deployment. Provision them once per Modal
workspace; keys can be rotated independently afterwards.

## `qr-sampler-prod`

Holds runtime configuration for `VllmQr`, `owui_edge`, and `owui`. Created
once and mounted into all three containers (see `app.py`).

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
  VLLM_API_KEY=<random 32-byte hex> \
  ENTROPIC_API_BASE_URL=https://entropic.science/api \
  SERVICE_TOKEN_SECRETS=<random 32-byte hex> \
  WEBUI_AUTH=true \
  ENABLE_SIGNUP=false \
  WEBUI_AUTH_TRUSTED_EMAIL_HEADER=X-Trusted-Email \
  WEBUI_AUTH_TRUSTED_NAME_HEADER=X-Trusted-Display-Name \
  OPENAI_API_BASE_URL=https://<account>--qr-sampler-entropic-vllmqr-serve.modal.run/v1 \
  OPENAI_API_KEY=<same value as VLLM_API_KEY> \
  OWUI_UPSTREAM_URL=https://<account>--qr-sampler-entropic-owui.modal.run
```

Generate random values:

```bash
openssl rand -hex 32   # VLLM_API_KEY, SERVICE_TOKEN_SECRETS
```

## `hf-token`

Held in a separate Secret because it's only needed by the one-shot
`download_weights` function — neither `VllmQr` nor the OWUI containers
should ever have HF credentials at request time.

```bash
modal secret create hf-token HF_TOKEN=<huggingface token>
```

## Rolling-secret rotation for `SERVICE_TOKEN_SECRETS`

`SERVICE_TOKEN_SECRETS` is a **comma-separated vector** (Pre-flight §11.4).
The signer (qr-sampler's OWUI filter and Pipe) uses the FIRST entry; the
verifier (entropic.science `lib/serviceToken.ts`) accepts a match against
ANY entry. This removes the lockstep-redeploy pain from secret rotation.

Procedure when rotating:

1. **Prepend** the new secret to both sides:
   - Modal: `modal secret update qr-sampler-prod SERVICE_TOKEN_SECRETS=<new>,<old>`
   - entropic.science api-server env: same value.
2. **Redeploy at leisure** — either side first is fine. While both old and
   new are live, requests signed under either secret are accepted.
3. After both deploys have settled and you have verified the new secret is
   actually being used by traffic:
   - Modal: `modal secret update qr-sampler-prod SERVICE_TOKEN_SECRETS=<new>`
   - entropic.science: same.
   - Redeploy both. Old secret is now removed.

No automated rotation in v1. Re-run this procedure once per quarter or
after any suspected leak.

## Bumping the OWUI integration values after first deploy

`OPENAI_API_BASE_URL` and `OWUI_UPSTREAM_URL` cannot be known before the
first `modal deploy` — they are derived from Modal's auto-assigned
`*.modal.run` URLs for each function. After the first deploy:

1. `modal app stats qr-sampler-entropic` to read the URLs.
2. `modal secret update qr-sampler-prod OPENAI_API_BASE_URL=... OWUI_UPSTREAM_URL=...`
3. `modal deploy deployments/modal/app.py` (second deploy picks up the URLs).

Alternative: bind the custom domain `chat.entropic.science` first
(`modal domain create chat.entropic.science --function owui_edge`), then
use the stable hostname.
