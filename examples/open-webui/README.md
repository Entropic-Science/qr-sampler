# Open WebUI Integration

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-style
web interface for chatting with models served by vLLM. Every qr-sampler
deployment profile includes it as an optional Docker Compose service.

This directory contains a **filter function** that lets you control qr-sampler
parameters (temperature, top-k, top-p, sample count, etc.) directly from the
Open WebUI admin panel — no API calls or environment variable changes needed.

## Starting Open WebUI

From any deployment profile directory, add `--profile ui`:

```bash
cd deployments/urandom          # or firefly-1, _template, your-profile
cp .env.example .env
docker compose --profile ui up --build
```

Open http://localhost:3000. Without `--profile ui`, Open WebUI does not start
and the deployment behaves exactly as before.

## Installing the filter function

The filter function ships as two files:

| File | Purpose |
|------|---------|
| `qr_sampler_filter.py` | Human-readable source code |
| `qr_sampler_filter.json` | Open WebUI importable JSON |

### Import steps

1. Open http://localhost:3000 and log in (first user becomes admin).
2. Go to **Admin Panel > Functions** (or **Workspace > Functions**).
3. Click **Import** (the upload icon).
4. Select `qr_sampler_filter.json` from this directory.
5. Toggle the imported function to **Global** so it applies to all models.

The filter is now active. Every chat message will include qr-sampler parameters
in requests sent to vLLM.

### Alternative: paste the source

If you prefer not to use the JSON import:

1. Go to **Admin Panel > Functions** and click **Create a new function**.
2. Set the type to **Filter**.
3. Copy the contents of `qr_sampler_filter.py` into the code editor.
4. Save and toggle to **Global**.

## Configuring parameters (Valves)

After importing the filter, click the **gear icon** next to it to open the
Valves panel. Each Valve maps to a qr-sampler per-request parameter:

### Filter control

| Valve | Default | Description |
|-------|---------|-------------|
| `priority` | `0` | Filter execution order (lower runs first). |
| `enable_qr_sampling` | `true` | Master switch. Set to `false` to pass requests through unmodified. |

### Token selection

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `top_k` | `50` | `qr_top_k` | Keep only the k most probable tokens (0 disables). |
| `top_p` | `0.9` | `qr_top_p` | Nucleus sampling threshold (1.0 disables). |

### Temperature

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `temperature_strategy` | `fixed` | `qr_temperature_strategy` | `fixed` or `edt` (entropy-dependent). |
| `fixed_temperature` | `0.7` | `qr_fixed_temperature` | Constant temperature (fixed strategy). |
| `edt_base_temp` | `0.8` | `qr_edt_base_temp` | Base coefficient for EDT. |
| `edt_exponent` | `0.5` | `qr_edt_exponent` | Power-law exponent for EDT. |
| `edt_min_temp` | `0.1` | `qr_edt_min_temp` | EDT temperature floor. |
| `edt_max_temp` | `2.0` | `qr_edt_max_temp` | EDT temperature ceiling. |

### Signal amplification

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `signal_amplifier_type` | `zscore_mean` | `qr_signal_amplifier_type` | Amplification algorithm. |
| `sample_count` | `20480` | `qr_sample_count` | Entropy bytes fetched per token. |
| `population_mean` | `127.5` | `qr_population_mean` | Null-hypothesis mean for byte values. |
| `population_std` | `73.612...` | `qr_population_std` | Population std for uniform [0, 255]. |
| `uniform_clamp_epsilon` | `1e-10` | `qr_uniform_clamp_epsilon` | Clamp u to avoid degenerate CDF. |

### Logging

| Valve | Default | Maps to | Description |
|-------|---------|---------|-------------|
| `log_level` | `summary` | `qr_log_level` | `none`, `summary`, or `full`. |
| `diagnostic_mode` | `false` | `qr_diagnostic_mode` | Store all token records in memory. |

## How it works

```
User types message in Open WebUI
  |
  +-> Open WebUI sends request to vLLM (/v1/chat/completions)
  |
  +-> Filter inlet() runs BEFORE the request reaches vLLM:
  |     - Reads current Valve values
  |     - Adds qr_top_k, qr_top_p, qr_temperature_strategy, etc.
  |       as top-level keys in the request body
  |
  +-> vLLM receives the request:
  |     - Unknown top-level keys become SamplingParams.extra_args
  |     - qr-sampler's resolve_config() reads qr_* from extra_args
  |     - Token sampling uses the parameters from the Valves
  |
  +-> Response streams back through Open WebUI to the user
```

Infrastructure settings (gRPC server address, fallback mode, etc.) are
**not** exposed as Valves — they cannot change per-request and are controlled
by environment variables on the vLLM container.

## What is NOT controlled by the filter

The filter only manages per-request sampling parameters. These settings are
configured via environment variables in your `.env` file and apply to all
requests:

- Entropy source type and gRPC server address
- gRPC transport mode, timeout, and retry count
- Fallback mode
- Circuit breaker thresholds
- API key authentication

See the [configuration reference](../../README.md#configuration-reference) in
the main README for the full list.

## Disabling the filter

To stop injecting qr-sampler parameters without removing the filter:

1. Open the Valves panel (gear icon).
2. Set `enable_qr_sampling` to `false`.

Requests will pass through to vLLM unmodified, and qr-sampler will use its
default configuration from environment variables.

## entropic.science integration

A second deployment of this stack lives at
[chat.entropic.science](https://chat.entropic.science), hosted on Modal with
a B200 GPU. That deployment adds two responsibilities on top of the local
profile above:

1. **Daily allowance metering** — the entropic.science site issues 128k
   weighted tokens (input + 3× output) per account per day. The qr-sampler
   OWUI **filter** (`qr_sampler_filter.py`) calls `entropic.science/api`
   on every chat:
   - *Inlet*: `POST /allowance/preflight` — rejects with a refill-time +
     waitlist CTA markdown if the user is below the per-request minimum
     reserved cost.
   - *Outlet*: `POST /allowance/debit` and `POST /conversations/upsert` —
     debits the weighted token cost and records the chat in the user's
     cross-device history index.
2. **Comparison mode** — the new manifold **pipe**
   (`qr_comparison_pipe.py`) registers one pseudo-model per real base model
   (e.g. `gemma-4-31b-reasoning--qr-vs-prng`). When selected, it issues two
   parallel streaming completions against the same vLLM endpoint with
   different `qr_entropy_source_type` values (`quantum_grpc` vs `system`)
   and renders the dual-column markdown live. Preflight + debit run with
   `comparisonMode=true` so the allowance gate uses ~2× cost.

### What's different from the local-dev install

| | Local dev | entropic.science / Modal |
|---|---|---|
| OWUI auth | OWUI's own user table | Trusted-header from `owui_edge` proxy |
| Filter Valves | All optional | `service_token_secret` + `api_base_url` **required** |
| Pipe registered? | No (filter only) | Yes — both filter and pipe |
| Per-request entropy source | Container env-only | Per-request override via filter / pipe |

### Wiring on the Modal deployment

The Modal profile at [`deployments/modal/`](../../deployments/modal/) sets
two env vars on the OWUI container that the filter and pipe both read:

| Env var | Value |
|---|---|
| `ENTROPIC_API_BASE_URL` | `https://entropic.science/api` |
| `SERVICE_TOKEN_SECRETS` | rolling-secret vector (see `deployments/modal/modal_secrets.md`) |

The filter and pipe read `SERVICE_TOKEN_SECRETS` via OWUI's Valves
(`service_token_secret`). The Valve's default is a `default_factory` that
picks up the env var, so an operator can override either via the env or
the Valves UI. The signer always uses the **first** entry of the
comma-separated vector; the entropic.science API verifier accepts a match
against **any** entry (Pre-flight §11.4).

### Installing the pipe on Modal's OWUI

After `modal deploy` and your first `chat.entropic.science` sign-in:

1. **Admin Panel → Functions → Import** → upload
   `examples/open-webui/qr_sampler_filter.json`. Toggle to **Global**.
2. **Admin Panel → Functions → Import** → upload
   `examples/open-webui/qr_comparison_pipe.json`. The two pseudo-models
   appear in OWUI's model selector after a refresh.

The filter Valves don't need editing — the defaults read the right env
vars. The pipe Valves expose `base_models` (defaults to both reasoning
models) and timeout knobs.

### Source-of-truth design

- [`../../deployments/modal/`](../../deployments/modal/) — Modal deployment
  profile, with full deploy walkthrough.
- [`../../CROSS-REPO-INTEGRATION.md`](../../CROSS-REPO-INTEGRATION.md) —
  cross-repo handshake with the entropic.science side.
- `entropic.science/.zenflow/tasks/qr-sampler-integration-fad6/spec.md`
  §5.3 (filter), §5.4 (pipe), §5.5–§5.7 (Modal deployment) — the
  authoritative design lives there.

## Customizing the UI port

Set `OPEN_WEBUI_PORT` in your `.env` file:

```
OPEN_WEBUI_PORT=8080
```

Then access Open WebUI at http://localhost:8080.

## Authentication

By default, Open WebUI runs without authentication (`OPEN_WEBUI_AUTH=false`).
This is convenient for local development. For shared or public servers, enable
authentication:

```
OPEN_WEBUI_AUTH=true
```

The first user to sign up becomes the admin.
