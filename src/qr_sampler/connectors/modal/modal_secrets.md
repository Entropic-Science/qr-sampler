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
  QR_FALLBACK_MODE=system \
  QR_SAMPLE_COUNT=12800 \
  QR_GRPC_SERVER_ADDRESS=127.0.0.1:50051 \
  QR_GRPC_MODE=unary \
  QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes \
  QR_GRPC_STREAM_METHOD_PATH= \
  QR_GRPC_API_KEY=<QRNG api-key> \
  QR_GRPC_API_KEY_HEADER=api-key \
  QR_GRPC_TIMEOUT_MS=5000 \
  QRNG_TUNNEL_HOSTNAME=qbert-grpc.cipherstone.co \
  CF_ACCESS_CLIENT_ID=<Cloudflare Access Service Token client id> \
  CF_ACCESS_CLIENT_SECRET=<Cloudflare Access Service Token client secret> \
  VLLM_MODELS=gemma-4-31b-reasoning,qwen3.5-9b-reasoning \
  VLLM_DEFAULT_MODEL=gemma-4-31b-reasoning \
  VLLM_MAX_MODEL_LEN=65536 \
  VLLM_GPU_MEMORY_UTILIZATION_PER_ENGINE=0.45 \
  SERVICE_TOKEN_SECRETS=<random 32-byte base64>
```

### QRNG via Cloudflare Access

The QRNG gRPC service (`qbert-grpc.cipherstone.co`) is published behind a
Cloudflare Zero Trust Access TCP application. Each `VllmQr*` container
runs a `cloudflared access tcp` sidecar (managed by
`qr_sampler.connectors.modal.cloudflared_sidecar`) that opens a loopback
listener at `127.0.0.1:50051` and forwards every byte through Cloudflare's
edge to the tunnel origin. The qr-sampler `QuantumGrpcSource` dials the
loopback address — it does not see Cloudflare or the QRNG public hostname
directly, which keeps auth and transport concerns isolated to the sidecar.

Three env vars drive the sidecar:

| Var | Required | Provisioned by | Notes |
|---|---|---|---|
| `QRNG_TUNNEL_HOSTNAME` | yes | QRNG admin | Hostname of the Cloudflare Access TCP app. Default in production is `qbert-grpc.cipherstone.co`. |
| `CF_ACCESS_CLIENT_ID` | yes | QRNG admin (Cloudflare Zero Trust → Access → Service Auth → Service Tokens) | Service Token client id. Handed over out-of-band. |
| `CF_ACCESS_CLIENT_SECRET` | yes | QRNG admin (same screen) | Service Token client secret. Handed over out-of-band. |

Optional tuning:

| Var | Default | When to override |
|---|---|---|
| `QRNG_TUNNEL_BIND_HOST` | `127.0.0.1` | Almost never — the gRPC client must use the same value via `QR_GRPC_SERVER_ADDRESS`. |
| `QRNG_TUNNEL_BIND_PORT` | `50051` | Only if another process in the container already owns 50051. |
| `QRNG_TUNNEL_STARTUP_TIMEOUT_S` | `15.0` | Raise if cloudflared startup is slow on a degraded edge POP. |

The QRNG service contract (sourced from the operator-supplied
`artifacts/qrng.proto` / `artifacts/README.md`):

* Wire format: `qrng.proto`, package `qrng`, service `QuantumRNG`, method
  `GetRandomBytes(RandomRequest) returns (RandomResponse)`. Request encodes
  `num_bytes` as protobuf field 1 (varint); response returns the random
  bytes as protobuf field 1 (length-delimited). Compatible with the
  qr-sampler protocol-agnostic gRPC client without code-generated stubs.
* Auth: the API key is sent as gRPC metadata under the literal header
  `api-key` (lowercase). Wire path: `QR_GRPC_API_KEY` env var →
  `QuantumGrpcSource._metadata` tuple → gRPC `metadata=` kwarg.
* Streaming: the QRNG proto defines unary only; `QR_GRPC_STREAM_METHOD_PATH`
  MUST be the empty string so the streaming code paths stay disabled.
* Rate limits (per the QRNG team's handoff):

  | Limit | Default |
  |---|---|
  | Per request | 35,200 bytes |
  | Per minute  | 500 requests |
  | Per day     | 500 MB |

  The default `QR_SAMPLE_COUNT=12800` stays comfortably under the 13,000-byte
  current cap on the production API key. Requests exceeding the limits return
  gRPC `RESOURCE_EXHAUSTED`; the qr-sampler client surfaces this as
  `EntropyUnavailableError` and the FallbackEntropySource degrades to
  `os.urandom` until the next minute / day budget refills.

### Fallback to `os.urandom`

`QR_FALLBACK_MODE=system` wires `QuantumGrpcSource` -> `FallbackEntropySource`
-> `SystemEntropySource`. On every primary failure (timeout, circuit-breaker
open, RESOURCE_EXHAUSTED) the wrapper logs a `entropy.degraded` warning with
the structured fields `event`, `primary`, `fallback`, `error`, and a
rate-limited `entropy.degraded.alert` once per minute so the degradation is
loud in `modal app logs` without flooding it on a sustained outage. The
alternative is `QR_FALLBACK_MODE=error`, which raises `EntropyUnavailableError`
and surfaces a 503 to the OWUI side; pick that for experiments where
quantum-vs-classical comparability cannot be silently broken.

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

## Full secret install (single command)

Use this when re-provisioning `qr-sampler-prod` from scratch or when an
investigation has surfaced multiple drifted keys at once. `--force`
replaces the secret atomically, so a partial run cannot leave stale keys
alongside the new ones.

> ⚠️ **Fetch `SERVICE_TOKEN_SECRETS` first.** `--force` replaces every
> key. Losing the rolling-secret vector breaks OWUI → vLLM bearer auth
> instantly.
>
> Modal's CLI has no `secret get` subcommand — secret values are not
> CLI-surfacable for security. To retrieve `SERVICE_TOKEN_SECRETS`,
> use the helper in the qr-llm-chat repo:
>
> ```powershell
> $env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"
> modal run scripts/dump_modal_secret.py::dump --key SERVICE_TOKEN_SECRETS
> ```
>
> (Run from `C:\Code\Entropic-Science\qr-llm-chat`. The helper mounts
> the `qr-sampler-prod` Secret in a tiny throwaway Modal function and
> prints the value once. Close the terminal afterwards so the value
> does not stay in scrollback.)
>
> Equivalent alternative: read `VLLM_API_KEY` from the
> `qr-llm-chat-prod` Modal Secret (same value as the first entry of
> `SERVICE_TOKEN_SECRETS`, see qr-llm-chat/infra/modal_secrets.md §6).

```bash
modal secret create qr-sampler-prod --force \
  SERVICE_TOKEN_SECRETS=<paste current value from `modal secret get`> \
  QR_ENTROPY_SOURCE_TYPE=quantum_grpc \
  QR_PREINIT_ENTROPY_SOURCES=quantum_grpc,system \
  QR_FALLBACK_MODE=system \
  QR_SAMPLE_COUNT=12800 \
  QR_GRPC_SERVER_ADDRESS=127.0.0.1:50051 \
  QR_GRPC_MODE=unary \
  QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes \
  QR_GRPC_STREAM_METHOD_PATH= \
  QR_GRPC_API_KEY=<QRNG api-key> \
  QR_GRPC_API_KEY_HEADER=api-key \
  QR_GRPC_TIMEOUT_MS=5000 \
  QR_MAX_MODEL_LEN=65536 \
  QR_GPU_MEMORY_UTILIZATION=0.90 \
  QRNG_TUNNEL_HOSTNAME=qbert-grpc.cipherstone.co \
  QRNG_API_KEY=<QRNG api-key, same as QR_GRPC_API_KEY> \
  CF_ACCESS_CLIENT_ID=<Cloudflare Access Service Token client id> \
  CF_ACCESS_CLIENT_SECRET=<Cloudflare Access Service Token client secret>
```

Two-name window: `QR_MAX_MODEL_LEN` / `QR_GPU_MEMORY_UTILIZATION` are the
canonical names. The legacy `VLLM_*` spellings are still read defensively
by `build_dispatcher_for`, but they trigger an `Unknown vLLM environment
variable` warning from vLLM's own envs.py scan, so the `vllm_serve.py`
loader pops them from `os.environ` before any vLLM import. Use the
`QR_*` names in new deployments.

Verification on the next cold-start: grep `modal app logs` for
`modal.secret_diag.summary` — the JSON line shows the resolved value of
every non-secret env var (e.g. `"QR_ENTROPY_SOURCE_TYPE": "set: quantum_grpc"`),
so the K-2 / K-3 misconfigurations no longer hide behind `set (N chars)`.

### Operator diagnostic events worth grepping

After a deploy, grep `modal app logs` for:

| Event | What it tells you |
|---|---|
| `modal.secret_diag.summary` | Every group's missing keys + (for allow-listed names) the resolved value |
| `modal.env.promoted` | One emit per legacy `VLLM_*` → `QR_*` rename; verifies the sweep ran |
| `modal.vllm.tunables_resolved` | Which env spelling supplied `max_model_len` / `gpu_memory_utilization` |
| `modal.grpc_address.non_loopback` | WARNING — `QR_GRPC_SERVER_ADDRESS` is not loopback; the cloudflared sidecar binds to 127.0.0.1 only so gRPC fetches will silently fall back to system entropy |
| `qrng.tcp_preprobe.failed` | The 500 ms TCP pre-probe rejected — sidecar is not listening at `127.0.0.1:50051` |
| `entropy.request.routed` | Per-request: what the client asked for vs the pipeline that actually answered |
| `entropy.request.completed` | Per-request boundary with token count + dominant entropy source |

The last two land the K-5 investigation: if `entropy.request.completed`
shows `tokens_generated=0` with `dominant_source=quantum_grpc`, the
gRPC backend dropped every fetch; if `dominant_source=system` for a
request that asked for `quantum_grpc`, the fallback wrapper engaged
(check `entropy.degraded` for the per-fallback record).

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
