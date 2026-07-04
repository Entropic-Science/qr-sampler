# qr-sampler

**Plug any randomness source into LLM token sampling.**

qr-sampler is an engine-agnostic framework that replaces standard pseudorandom token sampling with entropy from external sources — quantum random number generators (QRNGs), processor timing jitter, hardware noise, or any source you connect via gRPC or Python plugin. The core sampling pipeline has zero inference-engine dependencies; thin engine adapters integrate it with [vLLM](https://github.com/vllm-project/vllm), [vLLM-Metal](https://github.com/vllm-project/vllm-metal), or any engine that supports logits processing.

```bash
git clone https://github.com/Entropic-Science/qr-sampler.git
cd qr-sampler
pip install -e ".[cli]"
```

---

## Why qr-sampler?

Standard LLM inference uses pseudorandom number generators (PRNGs) for token sampling. PRNGs are deterministic — given the same seed, they produce the same output every time. qr-sampler replaces this with *true* randomness from physical processes:

- **Quantum RNGs** — photon detectors, vacuum fluctuation devices, or any hardware QRNG over gRPC
- **Hardware noise** — 63 thermal, timing, microarch, and GPU noise sources via [OpenEntropy](https://github.com/amenti-labs/openentropy)
- **Processor timing jitter** — CPU clock variations as an entropy source (experimental)
- **Your own source** — implement the `EntropySource` ABC or connect any hardware via the gRPC protocol
- **OS entropy** — `os.urandom()` as a fallback or baseline

### Research context: entropy purity and integrity verification

qr-sampler provides infrastructure for weak-signal integration experiments: studying whether tiny statistical biases in physical entropy sources produce measurable effects on LLM token selection. The signal amplification system converts thousands of random bytes into a single token choice, designed so that even a small shift in byte means produces a detectable shift in which token gets selected. All entropy is generated **just-in-time** — the physical measurement happens *after* logits are computed, never before. In server-draw mode (see below), integration happens server-side and each draw arrives with purity and integrity labels attached.

This is a research tool. It makes no claims beyond statistics — it provides the infrastructure to run rigorous experiments. The research narrative behind the QPI (quantum purity and integrity) layer lives in the [Qbert0G](https://github.com/Entropic-Science/Qbert0G) README.

---

## Architecture

```
                          qr-sampler
  ┌──────────────────────────────────────────────────────┐
  │                                                      │
  │  ┌─────────────────────────────────────────────┐     │
  │  │           core/ (engine-agnostic)            │     │
  │  │  SamplingPipeline: numpy-only, no torch     │     │
  │  │                                              │     │
  │  │  entropy/ ──► amplification/ ──► selection/  │     │
  │  │      │            │                  │       │     │
  │  │  get_random   amplify(bytes)    CDF search  │     │
  │  │  _bytes(n)    → u ∈ (0,1)      → token_id  │     │
  │  │                                              │     │
  │  │  temperature/ ─── compute_temperature()      │     │
  │  │  logging/ ─────── per-token diagnostics      │     │
  │  └──────────────────────┬──────────────────────┘     │
  │                         │                            │
  │  ┌──────────────────────┴──────────────────────┐     │
  │  │         engines/ (thin adapters)             │     │
  │  │  VLLMAdapter: torch ↔ numpy, one-hot force  │     │
  │  └──────────────────────┬──────────────────────┘     │
  │                         │                            │
  │  qthought.py ── QthoughtRoller: the same entropy     │
  │                 stack driving discrete choices        │
  │                 (consumed by qr-llm-qthought)         │
  │  contract.py ── the only import surface for           │
  │                 downstream repos                      │
  │                                                      │
  │  profiles/ ─── declarative YAML metadata             │
  │  cli/ ──────── validate, build, list, info           │
  │  templates/ ── Jinja2 for Docker Compose generation  │
  └──────────────────────────────────────────────────────┘
```

### Per-token sampling pipeline

```
Engine adapter calls pipeline.sample_token(logits_1d)
  │
  ├─ Temperature strategy ─────── Compute per-token temperature
  │   (fixed / edt / hvh_drift)       from the logit distribution
  │
  ├─ Entropy source ───────────── Fetch fresh random bytes
  │   (gRPC / system / timing /       just-in-time, after logits exist
  │    openentropy / custom)          (10,000 bytes per token by default)
  │
  ├─ Signal amplification ─────── Convert the bytes → one float u ∈ (0,1)
  │   (z-score or ECDF)               via statistical aggregation
  │
  ├─ Token selector ───────────── top-k → softmax → min-p → top-p → CDF
  │   (CDF binary search with u)      → select token
  │
  └─ Force one-hot logits ─────── Set selected token to 0.0, all others to -inf
      (engine picks exactly              (adapter converts to the engine's
       this token)                        native tensor)
```

The core pipeline is importable and functional without vLLM, torch, or any engine package. Engine adapters convert between engine-native tensors and numpy, delegate to `SamplingPipeline.sample_token()`, and write the result back.

---

## Quick start

### 1. Validate your stack (optional)

The CLI checks compatibility of engines, models, entropy sources, and amplifiers before you deploy:

```bash
# Check a specific combination
qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --entropy quantum_grpc

# Exit codes: 0 = all known-working, 1 = untested, 2 = incompatible / missing deps
```

### 2. Generate deployment files

```bash
# Generate Docker Compose for vLLM + quantum gRPC entropy
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy

# Preview without writing files
qr-sampler build --engine vllm --entropy system --dry-run
```

This renders a `docker-compose.yml` and a `.env` from built-in Jinja2 templates:

- an `inference` service (vLLM), and
- an `entropy-server` service (built from `examples/docker/Dockerfile.entropy-server`) when the chosen entropy source is gRPC-based; local sources (`system`, `timing_noise`, ...) need no extra service.

### 3. Launch

```bash
cd deploy
# Edit .env — set HF_TOKEN if using a gated model
docker compose up --build
```

### 4. Send a request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of randomness is",
    "max_tokens": 100
  }'
```

### Bare-metal install (without Docker)

```bash
pip install -e .   # from a clone, on the machine running vLLM

# Start vLLM — qr-sampler registers automatically via entry points
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 --gpu-memory-utilization 0.80
```

Configure the entropy source via environment variables:

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 --gpu-memory-utilization 0.80
```

Look for this line in the server logs to confirm the plugin is active:

```
VLLMAdapter initialized: vocab_size=..., default_entropy_source=..., preinit_sources=..., amplifier=..., temperature=...
```

### Apple Silicon (macOS)

qr-sampler works on Apple Silicon via [vllm-metal](https://github.com/vllm-project/vllm-metal), a community-maintained vLLM plugin under the official `vllm-project` GitHub org. It uses MLX under the hood but exposes the same vLLM API and plugin system — same entry points, same endpoints, same `curl` commands. Use MLX-format models from the [mlx-community](https://huggingface.co/mlx-community) collection.

> **Prerequisite:** vllm-metal currently does not load custom logits processors registered via entry points. [PR #124](https://github.com/vllm-project/vllm-metal/pull/124) fixes this; until it is merged, apply the patch manually or install from the PR branch. Without it, qr-sampler's plugin is silently skipped.

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
pip install -e /path/to/qr-sampler
vllm serve mlx-community/Qwen3-0.6B-4bit
```

All configuration works identically to the NVIDIA setup. Docker deployment profiles are **not** compatible with Apple Silicon (no Metal GPU passthrough in Docker's Linux VM) — vllm-metal must run natively.

---

## CLI reference

The CLI requires the `[cli]` extra: `pip install -e ".[cli]"`

```bash
qr-sampler list engines              # Engine profiles (vllm, vllm_metal)
qr-sampler list models --engine vllm # Known-working models for an engine
qr-sampler list entropy-sources      # All entropy source profiles
qr-sampler list amplifiers           # Signal amplification algorithms
qr-sampler list samplers             # Temperature strategies
qr-sampler list presets              # Preset bundles (see Presets below)

qr-sampler info engine vllm
qr-sampler info entropy quantum_grpc
qr-sampler info amplifier zscore_mean
qr-sampler info sampler edt
qr-sampler info preset creative_sampling

qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct
qr-sampler validate --config stack.yaml

qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy
qr-sampler build --engine vllm --entropy system --dry-run
qr-sampler build --engine vllm --entropy timing_noise --force --output ./deploy
```

`validate` exit codes: `0` = all known-working, `1` = untested combinations (warnings), `2` = incompatible or missing dependencies.

---

## Entropy sources

### Built-in sources

| Source | Identifier | Transport | Description |
|---|---|---|---|
| **System** | `system` | Local | `os.urandom()` — OS cryptographic RNG. Available everywhere. Default. |
| **Quantum gRPC** | `quantum_grpc` | gRPC | Remote entropy server via gRPC. Supports unary, server streaming, and bidi streaming. |
| **OpenEntropy** | `openentropy` | Local | 63 hardware noise sources (thermal, timing, microarch, GPU). No network needed. |
| **Timing noise** | `timing_noise` | Local | CPU timing jitter (experimental). |
| **Mock uniform** | `mock_uniform` | Local | Configurable test source with seed/bias. For testing only. |

### Quantum gRPC

Connect any entropy server that speaks gRPC. The production QRNG server is
[Qbert0G](https://github.com/Entropic-Science/Qbert0G) (>= 1.0), which serves this
package's native `qr_entropy.EntropyService` protocol directly — the defaults
below just work against it, including `sequence_id` echo verification and
`bidi_streaming`. qr-sampler also ships example servers and an annotated template:

```bash
# Run the built-in urandom-over-gRPC example server
python examples/servers/simple_urandom_server.py --address 0.0.0.0:50051

# Point qr-sampler at it
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
```

Three transport modes:

| Mode | `QR_GRPC_MODE` | Latency | Best for |
|---|---|---|---|
| **Unary** | `unary` | ~1-2ms | Simplicity, debugging. Default. |
| **Server streaming** | `server_streaming` | ~0.5-1ms | Middle ground |
| **Bidirectional** | `bidi_streaming` | ~50-100us (same machine) | Production, lowest latency |

For co-located hardware, use Unix domain sockets:

```bash
python my_qrng_server.py --address unix:///var/run/qrng.sock
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
export QR_GRPC_MODE=bidi_streaming
```

The gRPC client uses configurable method paths, so it can talk to third-party QRNG protos as long as field 1 carries the byte count (request) and the random bytes (response). Configure custom protos via `QR_GRPC_METHOD_PATH` and `QR_GRPC_STREAM_METHOD_PATH`. The wire codec lives in one place (`proto/wire.py` + the hand-written pb2 stubs) and follows proto3 semantics.

#### Pipelined commit-then-fetch prefetch

By default (`QR_ENTROPY_PREFETCH=1`), the gRPC request for token *N+1* fires the instant token *N* is selected, so the network round trip overlaps the engine's next forward pass. The post-selection causal contract is preserved by a 63-bit commitment nonce (`SHA-256(salt ‖ step ‖ prev_token_id)`, truncated) carried in the proto's `sequence_id` field; a server that echoes it binds its entropy to a request that could only exist *after* the previous selection. Disable with `QR_ENTROPY_PREFETCH=0` (or per-request `qr_entropy_prefetch: false`) for strictly-serial fetch-after-logits timing.

#### Circuit breaker

The gRPC client includes an adaptive circuit breaker:

- Tracks rolling P99 latency over the last 100 requests; timed-out fetches also feed the window, so the ceiling re-learns upward when the backend genuinely slows
- Sets timeout to `max(5ms, P99 * 1.5)` (configurable via `QR_CB_*` env vars; raise `QR_CB_MIN_TIMEOUT_MS` for tunnelled/remote backends)
- Opens after 3 consecutive failures; the half-open recovery wait starts at `QR_CB_RECOVERY_WINDOW_S` (10 s) and **backs off exponentially** (doubling per open without an intervening success) up to `QR_CB_RECOVERY_WINDOW_MAX_S` (60 s), so a sustained outage settles at ~1 probe/min instead of a connect storm
- The half-open attempt resets the channel first, so a stale connection can't waste the recovery cycle
- Falls back to `QR_FALLBACK_MODE` while the circuit is open

#### Server-side quotas

Per-provider quota guards are config fields (defaults match the reference QRNG provider): `QR_QRNG_MAX_BYTES_PER_REQUEST` (35,200), `QR_QRNG_MAX_REQUESTS_PER_MINUTE` (500), `QR_QRNG_MAX_BYTES_PER_DAY` (500 MiB).

### OpenEntropy

[OpenEntropy](https://github.com/amenti-labs/openentropy) harvests entropy from 63 hardware noise sources on the local machine. No network, no API keys, no gRPC server needed.

```bash
pip install openentropy
export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_CONDITIONING=raw   # raw (research default) | vonneumann | sha256
```

`QR_OE_CONDITIONING` is an infrastructure setting read once at source construction — it cannot be overridden per-request (sending `qr_oe_conditioning` fails with a validation error).

### Fallback behavior

The `FallbackEntropySource` wraps a primary source with an automatic fallback:

- Only catches `EntropyUnavailableError` — other exceptions propagate
- Logs a rate-limited warning while degraded and exposes `last_source_used` / `fallback_count`
- Optionally publishes degraded/recovered state to a cross-process status file (`QR_ENTROPY_STATUS_FILE`, default `<tempdir>/qr_entropy_status.json`; empty string disables) so a separate API-server process can report entropy health without touching the QRNG

Configure with `QR_FALLBACK_MODE`: `system` (default), `mock_uniform`, or `error` (raise immediately, no fallback).

### Per-request entropy-source switching

The vLLM adapter pre-initialises one pipeline per source listed in `QR_PREINIT_ENTROPY_SOURCES` (default `"quantum_grpc,system"`). A request can then switch source with `extra_args: {"qr_entropy_source_type": "system"}` — enabling quantum-vs-PRNG comparison runs on a single GPU. Unknown or un-preinitialised values are rejected cleanly.

### Third-party entropy sources

Any Python package can register entropy sources via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "my_package:LavaLampEntropySource"
```

The source is discovered lazily on first use. See [Setting up your own entropy source](#setting-up-your-own-entropy-source) below.

---

## Signal amplification

The signal amplification system converts raw entropy bytes into a single uniform float `u` in `(0, 1)` that drives token selection from the CDF.

### Z-score mean (`zscore_mean`) — default

1. Interprets raw bytes as uint8 values
2. Computes the sample mean M
3. Derives SEM = `population_std / sqrt(N)` (never stored — always computed)
4. Computes z-score: `z = (M - population_mean) / SEM`
5. Maps to uniform via normal CDF: `u = 0.5 * (1 + erf(z / sqrt(2)))`
6. Clamps to `(epsilon, 1-epsilon)`

Under the null hypothesis (no bias), `u` is uniformly distributed on (0, 1). A small per-byte bias accumulates over thousands of samples, producing a detectable shift:

```
10,000 bytes with +0.06 mean bias per byte:
  M ≈ 127.56, SEM ≈ 0.736, z ≈ 0.08, u ≈ 0.53
```

### ECDF (`ecdf`)

Empirical CDF amplifier with online calibration (`QR_ECDF_CALIBRATION_SAMPLES`, default 2000). Maps raw bytes to uniform via a calibrated empirical distribution function without assuming a specific input distribution.

### Thought-level aggregate (`zscore_thought`)

Used by the qthought decode lane: per-decision draws stay plain `zscore_mean`, while a per-thought aggregate bias statistic rides alongside. See `qr-sampler info amplifier zscore_thought`.

### Server-side integrated draw (`server`) — server-draw mode

With `qr_signal_amplifier_type: "server"`, the client does not fetch raw bytes at all. Instead it calls the `qr_purity.PurityService` gRPC protocol (served by [Qbert0G](https://github.com/Entropic-Science/Qbert0G) ≥ the QPI release, on the same socket as `EntropyService`): the *server* reads a block of device entropy (2 MiB by default), integrates it against a frozen device fingerprint, and returns one externally supplied uniform draw `u` plus metadata — the integration z-score, a purity label (`origin/integrity/processing[/...]` taxonomy), the block-coherence statistic `(coherence_r, coherence_z, coherence_valid)`, the integrator name, and the number of integrated bytes. The pipeline records all of it as a `DrawMeta` on the `SamplingResult` and the token record.

Properties:

- **Same causal contract** — the draw request carries the commit-then-fetch nonce in `sequence_id`, echo-verified exactly like the byte path; prefetch works identically.
- **Fail-safe degradation** — if the server does not implement `PurityService` (or the draw fails), the pipeline falls back to fetching `qr_sample_count` bytes and amplifying locally with `zscore_mean`; the record shows `entropy_is_fallback: true` and no `DrawMeta`. An EntropyService-only server keeps working unmodified.
- **Configuration** — `qr_draw_source_id` (empty = the API key's bound source) and `qr_draw_block_bytes` (`0` = server default) select what the server integrates.

### Coherence gate (`coherence_gate` temperature strategy)

Not an amplifier, but the consumer of the draw metadata: see Temperature strategies below.

---

## Temperature strategies

### Fixed temperature (`fixed`) — default

Returns a constant temperature for every token. Set via `QR_FIXED_TEMPERATURE` (default: 1.0).

### Entropy-dependent temperature (`edt`)

Dynamically adjusts temperature based on the Shannon entropy of the logit distribution:

```
H_norm = H / ln(vocab_size)         # Normalized entropy [0, 1]
T = base_temp * H_norm^exponent     # Power-law scaling
T = clamp(T, min_temp, max_temp)    # Bounds enforcement
```

High-entropy (uncertain) distributions get higher temperatures; low-entropy (confident) distributions get lower temperatures.

### HVH drift (`hvh_drift`)

Stateful per-request strategy tracking EMAs of Shannon entropy (H) and varentropy (VH); the drift between current and EMA values drives temperature and dynamic min-p adjustments token-by-token. Shipped via the `creative_sampling` preset.

### Coherence gate (`coherence_gate`)

A wrapper strategy for server-draw mode: it composes an inner strategy (`qr_coherence_inner_strategy`, default `fixed`) and adds a temperature boost when the *previous* token's draw reported significant device coherence:

```
b   = coherence_t_boost_max * max(0, coherence_r)   if coherence_valid and coherence_z >= coherence_threshold, else 0
b̄   ← ema_alpha * b + (1 - ema_alpha) * b̄            # EMA smoothing
T   = inner_strategy(T_base + b̄)                     # boost applied to the inner strategy's base temperature
```

Key properties:

- **Lag-by-one**: token *N*'s temperature reacts to token *N-1*'s draw metadata (the draw for token *N* happens after logits — and after the temperature — are computed).
- **Fail-safe**: no draw metadata, first token, malformed metadata, `coherence_valid=false`, or any inner-strategy hiccup under boost ⇒ exactly the unboosted base temperature. The gate can only ever *add* temperature, never corrupt sampling.
- **Labelling**: diagnostics gain `gate_open`, `gate_boost`, `coherence_z`, `coherence_valid`, which flow into the token record (and the cross-process status file) — so every sampled token is labelled with whether the gate was open.
- Knobs: `qr_coherence_threshold` (default 3.5), `qr_coherence_t_boost_max` (0.5), `qr_coherence_ema_alpha` (0.3), `qr_coherence_inner_strategy` (`fixed`).

---

## Presets

A preset is a named bundle of `qr_*` overrides that callers opt into via `qr_preset` (per-request) or `QR_PRESET` (env var, process-wide default). Six ship built in:

| Preset | What it does | Status |
|---|---|---|
| `creative_sampling` | `hvh_drift` strategy with the V6_HVD_R01_01 winner hyperparameters + dynamic min-p | **Experimental** |
| `normal_t1` | Vanilla `fixed` strategy at T=1, no top-k / top-p truncation | Stable baseline |
| `qthought` | Entropy profile for the `QthoughtRoller` decode lane (quantum source + `zscore_thought`) | Frozen research lineage |
| `qthought_think` | REFLECT sampling lane for qr-llm-qthought (hotter HVH-drift, 6,000-byte fetch) | Frozen research lineage |
| `qthought_voice` | SPEAK sampling lane for qr-llm-qthought (EDT + nucleus/top-k, 10,000-byte fetch) | Frozen research lineage |
| `qthought_purity` | Server-draw mode: `server` amplifier + `coherence_gate` over `fixed` at T=1 (PurityService required; degrades fail-safe) | QPI default composition |

The three historical `qthought*` presets are scientific lineage consumed by the sibling [qr-llm-qthought](https://github.com/Entropic-Science/LiveLM-backend) service — their values are pinned by contract tests on both sides. Do not tune them casually.

```bash
# Process-wide
QR_PRESET=creative_sampling vllm serve Qwen/Qwen2.5-1.5B-Instruct

# Per-request
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt": "The nature of randomness is",
  "max_tokens": 100,
  "extra_args": {"qr_preset": "creative_sampling"}
}'
```

Per-request `qr_preset` beats the `QR_PRESET` env var; per-request `qr_*` keys override individual values inside the active preset:

```json
{"extra_args": {"qr_preset": "creative_sampling", "qr_hvh_t_base": 1.2}}
```

> **Heads up:** `creative_sampling` is research-grade. Pin to `normal_t1` (or omit `qr_preset` entirely) for reproducible baselines.

---

## Per-request parameter overrides

Override sampling parameters on individual requests via `extra_args`:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of randomness is",
    "max_tokens": 100,
    "extra_args": {
      "qr_temperature_strategy": "edt",
      "qr_top_k": 100,
      "qr_top_p": 0.95,
      "qr_sample_count": 40960,
      "qr_diagnostic_mode": true
    }
  }'
```

Infrastructure fields (gRPC address/mode, fallback mode, quotas, OpenEntropy conditioning, ...) cannot be overridden per-request — they are set at server startup, and attempts are rejected with a clean validation error. The per-request set is derived from field metadata in `config/model.py`, so the tables below are always in sync with the code.

---

## Configuration reference

All configuration is done via environment variables with the `QR_` prefix (a `.env` file works too — pydantic-settings loads it automatically). Per-request overrides use the `qr_` prefix in `extra_args`.

### Infrastructure fields (NOT per-request overridable)

| Environment variable | Default | Description |
|---|---|---|
| `QR_GRPC_SERVER_ADDRESS` | `localhost:50051` | gRPC entropy server address (`host:port` or `unix:///path`) |
| `QR_GRPC_TIMEOUT_MS` | `5000` | gRPC call timeout in milliseconds |
| `QR_GRPC_RETRY_COUNT` | `2` | Retry attempts after gRPC failure |
| `QR_GRPC_MODE` | `unary` | Transport mode: `unary`, `server_streaming`, `bidi_streaming` |
| `QR_GRPC_METHOD_PATH` | `/qr_entropy.EntropyService/GetEntropy` | gRPC method path for unary RPC |
| `QR_GRPC_STREAM_METHOD_PATH` | `/qr_entropy.EntropyService/StreamEntropy` | Streaming method path (empty disables streaming) |
| `QR_GRPC_DRAW_METHOD_PATH` | `/qr_purity.PurityService/GetDraw` | Unary server-integrated draw method path (empty disables the draw handle) |
| `QR_GRPC_DRAW_STREAM_METHOD_PATH` | `/qr_purity.PurityService/StreamDraws` | Bidi-streaming draw method path (empty disables the draw stream handle) |
| `QR_GRPC_API_KEY` | *(empty)* | API key sent via gRPC metadata (empty = no auth; never logged) |
| `QR_GRPC_API_KEY_HEADER` | `api-key` | gRPC metadata header name for the API key |
| `QR_FALLBACK_MODE` | `system` | Fallback when primary fails: `error`, `system`, `mock_uniform` |
| `QR_QRNG_MAX_BYTES_PER_REQUEST` | `35200` | Provider quota: max bytes per gRPC request |
| `QR_QRNG_MAX_REQUESTS_PER_MINUTE` | `500` | Provider quota: request rate ceiling |
| `QR_QRNG_MAX_BYTES_PER_DAY` | `524288000` | Provider quota: daily byte budget (500 MiB) |
| `QR_CB_WINDOW_SIZE` | `100` | Rolling latency window size for P99 computation |
| `QR_CB_MIN_TIMEOUT_MS` | `5.0` | Minimum adaptive timeout in milliseconds |
| `QR_CB_TIMEOUT_MULTIPLIER` | `1.5` | Multiplier applied to P99 latency for adaptive timeout |
| `QR_CB_RECOVERY_WINDOW_S` | `10.0` | BASE seconds before half-open retry after circuit opens |
| `QR_CB_RECOVERY_WINDOW_MAX_S` | `60.0` | Exponential-backoff cap for the recovery window |
| `QR_CB_MAX_CONSECUTIVE_FAILURES` | `3` | Consecutive failures before the circuit opens |
| `QR_ECDF_CALIBRATION_SAMPLES` | `2000` | ECDF amplifier calibration window |
| `QR_OE_CONDITIONING` | `raw` | OpenEntropy conditioning: `raw`, `vonneumann`, `sha256` |
| `QR_OE_SOURCES` | *(empty)* | OpenEntropy source allow-list (comma-separated; empty = all) |
| `QR_OE_PARALLEL` | `true` | OpenEntropy parallel harvesting |
| `QR_OE_TIMEOUT` | `5.0` | OpenEntropy harvest timeout (s) |
| `QR_PRESET` | *(unset)* | Process-wide default preset |

Environment-only settings (read outside the config model):

| Environment variable | Default | Description |
|---|---|---|
| `QR_ENTROPY_STATUS_FILE` | `<tempdir>/qr_entropy_status.json` | Cross-process entropy-status file (`telemetry/status_file.py`); empty string disables |
| `QR_PREINIT_ENTROPY_SOURCES` | `quantum_grpc,system` | Sources the vLLM adapter pre-initialises pipelines for |

### Sampling parameters (per-request overridable)

| Environment variable | extra_args key | Default | Description |
|---|---|---|---|
| `QR_ENTROPY_SOURCE_TYPE` | `qr_entropy_source_type` | `system` | Entropy source for this request (must be pre-initialised) |
| `QR_ENTROPY_PREFETCH` | `qr_entropy_prefetch` | `true` | Pipelined commit-then-fetch prefetch |
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | `zscore_mean` | Signal amplification algorithm (`server` = server-draw mode) |
| `QR_DRAW_SOURCE_ID` | `qr_draw_source_id` | *(empty)* | Source id for server-integrated draws (empty = server's API-key binding) |
| `QR_DRAW_BLOCK_BYTES` | `qr_draw_block_bytes` | `0` | Raw block size for server-integrated draws (`0` = server default) |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | `10000` | Entropy bytes fetched per token |
| `QR_POPULATION_MEAN` | `qr_population_mean` | `127.5` | Null-hypothesis mean for byte values |
| `QR_POPULATION_STD` | `qr_population_std` | `73.612...` | Population std for uniform [0, 255] |
| `QR_UNIFORM_CLAMP_EPSILON` | `qr_uniform_clamp_epsilon` | `1e-10` | Clamp u to avoid degenerate CDF |
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | `fixed` | Strategy: `fixed`, `edt`, `hvh_drift`, `coherence_gate` |
| `QR_COHERENCE_THRESHOLD` | `qr_coherence_threshold` | `3.5` | Minimum `coherence_z` for the gate to open |
| `QR_COHERENCE_T_BOOST_MAX` | `qr_coherence_t_boost_max` | `0.5` | Max temperature boost at `coherence_r = 1` |
| `QR_COHERENCE_EMA_ALPHA` | `qr_coherence_ema_alpha` | `0.3` | EMA smoothing for the gate boost |
| `QR_COHERENCE_INNER_STRATEGY` | `qr_coherence_inner_strategy` | `fixed` | Inner strategy the coherence gate composes over |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | `1.0` | Constant temperature (fixed strategy) |
| `QR_EDT_BASE_TEMP` | `qr_edt_base_temp` | `0.8` | Base coefficient for EDT |
| `QR_EDT_EXPONENT` | `qr_edt_exponent` | `0.5` | Power-law exponent for EDT |
| `QR_EDT_MIN_TEMP` | `qr_edt_min_temp` | `0.1` | EDT temperature floor |
| `QR_EDT_MAX_TEMP` | `qr_edt_max_temp` | `2.0` | EDT temperature ceiling |
| `QR_HVH_T_BASE` (+ other `QR_HVH_*`) | `qr_hvh_*` | see `qr-sampler info preset creative_sampling` | HVH-drift family knobs |
| `QR_TOP_K` | `qr_top_k` | `0` | Top-k filtering (`<=0` disables) |
| `QR_TOP_P` | `qr_top_p` | `1.0` | Nucleus sampling threshold (`1.0` disables) |
| `QR_MIN_P_BASE` | `qr_min_p_base` | `0.0` | Static min-p floor (`0.0` = strict no-op) |
| `QR_LOG_LEVEL` | `qr_log_level` | `summary` | Logging: `none`, `summary`, `full` |
| `QR_DIAGNOSTIC_MODE` | `qr_diagnostic_mode` | `false` | Store all token records in memory |

---

## Setting up your own entropy source

qr-sampler is designed to connect *any* randomness source to LLM token sampling. There are two approaches.

### Approach A: gRPC server (recommended)

Implement a gRPC server. You can use the built-in `qr_entropy.EntropyService` protocol (example servers provided; [Qbert0G](https://github.com/Entropic-Science/Qbert0G) is the production reference implementation), or your own proto as long as field 1 carries the byte count (request) and random bytes (response).

#### 5-minute walkthrough

1. **Copy the template:**

```bash
cp examples/servers/qrng_template_server.py my_qrng_server.py
```

2. **Implement three methods** in the `QRNGHardware` class:

```python
class QRNGHardware:
    def __init__(self, device_path="/dev/qrng0"):
        self._device = open(device_path, "rb")

    def generate(self, n_bytes: int) -> bytes:
        # CRITICAL: Generate entropy NOW, not from a buffer.
        # The quantum measurement must happen during this call.
        return self._device.read(n_bytes)

    def close(self):
        self._device.close()
```

3. **Run it:**

```bash
python my_qrng_server.py --port 50051
```

4. **Point qr-sampler at it:**

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
```

The template handles all gRPC boilerplate (unary + bidirectional streaming, health checks, graceful shutdown). You only write the hardware-specific code.

#### The gRPC protocol

```protobuf
service EntropyService {
  rpc GetEntropy (EntropyRequest) returns (EntropyResponse);
  rpc StreamEntropy (stream EntropyRequest) returns (stream EntropyResponse);
}

message EntropyRequest {
  int32 bytes_needed = 1;
  int64 sequence_id = 2;
}

message EntropyResponse {
  bytes data = 1;
  int64 sequence_id = 2;
  int64 generation_timestamp_ns = 3;
  string device_id = 4;
}
```

Any language that supports gRPC can implement this server. Servers that echo `sequence_id` additionally get verifiable post-selection ordering under prefetch (see the commit-then-fetch section above).

#### Just-in-time constraint

The entropy must be generated **after** the client sends the request, not from a pre-generated pool:

- No buffering or caching of previously generated bytes
- The physical measurement happens during the `generate()` call
- `generation_timestamp_ns` in the response proves freshness

#### Deployment options

**systemd (Linux):**

```bash
sudo cp examples/systemd/qr-entropy-server.service /etc/systemd/system/
sudo cp examples/systemd/qr-entropy-server.env /etc/default/qr-entropy-server
sudo systemctl enable --now qr-entropy-server
```

**Unix domain sockets** (lowest latency for co-located hardware): see the Quantum gRPC section above. Ready-made per-host compose profiles live under `deployments/`.

### Approach B: Python plugin (in-process)

For entropy sources that don't need a separate server, implement the `EntropySource` ABC directly:

```python
from qr_sampler.entropy.base import EntropySource
from qr_sampler.entropy.registry import register_entropy_source

@register_entropy_source("my_source")
class MyEntropySource(EntropySource):
    @property
    def name(self) -> str:
        return "my_source"

    @property
    def is_available(self) -> bool:
        return True

    def get_random_bytes(self, n: int) -> bytes:
        return my_hardware.read(n)

    def close(self) -> None:
        my_hardware.disconnect()
```

Register via entry points in your package's `pyproject.toml`:

```toml
[project.entry-points."qr_sampler.entropy_sources"]
my_source = "my_package.entropy:MyEntropySource"
```

Then set `QR_ENTROPY_SOURCE_TYPE=my_source`.

### Validation

Test your entropy source:

```python
from qr_sampler.config import QRSamplerConfig
from qr_sampler.entropy.qgrpc import QuantumGrpcSource

config = QRSamplerConfig(
    entropy_source_type="quantum_grpc",
    grpc_server_address="localhost:50051",
)
source = QuantumGrpcSource(config)

data = source.get_random_bytes(1024)
assert len(data) == 1024

print(source.health_check())  # {'source': 'quantum_grpc', 'healthy': True, ...}
source.close()
```

For statistical validation, check that your source produces uniform byte distributions:

```python
import numpy as np
from scipy import stats

data = source.get_random_bytes(100_000)
samples = np.frombuffer(data, dtype=np.uint8)

stat, p_value = stats.kstest(samples / 255.0, 'uniform')
print(f"KS statistic: {stat:.6f}, p-value: {p_value:.6f}")
# p-value should be > 0.05 for a good entropy source
```

---

## Downstream consumers and the contract module

qr-sampler has a second, non-vLLM consumer: [qr-llm-qthought](https://github.com/Entropic-Science/LiveLM-backend), a proto-thought chatbot engine that drives its case-frame grammar with the `QthoughtRoller` — the same entropy stack, applied to discrete choices instead of token logits.

Downstream repos import **only** through `qr_sampler.contract` (roller + provenance types, config + presets, entropy primitives, exceptions, `CONTRACT_VERSION`). Everything else is internal and free to move. Contract tests on both sides guard the seam; see `AGENTS.md` for the rules.

---

## Project structure

```
src/qr_sampler/
├── __init__.py                    # Version + re-exports (import is side-effect-free)
├── __main__.py                    # CLI entry: python -m qr_sampler
├── exceptions.py                  # Exception hierarchy
├── contract.py                    # Cross-repo import surface (CONTRACT_VERSION)
├── qthought.py                    # QthoughtRoller: entropy-driven discrete choices
├── config/
│   ├── model.py                   # QRSamplerConfig (pydantic-settings; per-request set from field metadata)
│   ├── presets.py                 # BUILTIN_PRESETS + PRESET_* constants + expansion
│   └── resolve.py                 # resolve_config() + validate_extra_args()
├── core/                          # Engine-agnostic pipeline (NO torch)
│   ├── pipeline.py                # SamplingPipeline + factories + derive_commit_nonce
│   └── types.py                   # SamplingResult
├── engines/
│   ├── base.py / registry.py      # EngineAdapter ABC + registry
│   └── vllm/                      # adapter.py (LogitsProcessor) + telemetry.py
├── entropy/
│   ├── base.py / registry.py      # EntropySource ABC + lazy-builtin registry
│   ├── system.py / timing.py / mock.py / openentropy.py / fallback.py
│   └── qgrpc/                     # Quantum gRPC: source, transport, channel, breaker, preprobe
├── amplification/                 # zscore_mean, zscore_thought, ecdf + registry
├── temperature/                   # fixed, edt, hvh_drift + registry
├── selection/                     # TokenSelector (top-k → softmax → min-p → top-p → CDF)
├── logging/                       # TokenSamplingRecord + SamplingLogger
├── telemetry/                     # status_file.py (cross-process entropy status)
├── proto/                         # wire.py + entropy_service proto/pb2 stubs
├── profiles/                      # Declarative YAML metadata for the CLI
├── cli/                           # list / info / validate / build
└── templates/                     # Jinja2 for qr-sampler build

scripts/check.py                   # One-command verification (lint/format/types/security/tests)
examples/                          # servers/, docker/, systemd/
deployments/                       # Per-host compose profiles
```

---

## Statistical analysis

qr-sampler includes statistical tests (in `tests/test_statistical_properties.py`, requires `scipy`) that validate the mathematical properties of the sampling pipeline:

- **KS-test for u-value uniformity**: under the null hypothesis (no bias), amplified `u` values are uniform on (0, 1).
- **Bias detection**: a small per-byte mean shift produces a statistically detectable shift in the `u` distribution — the amplification system is sensitive enough for weak-signal integration experiments.
- **EDT monotonicity**: higher-entropy logit distributions get higher temperatures.

```bash
pytest tests/test_statistical_properties.py -v
```

---

## Development

```bash
git clone https://github.com/Entropic-Science/qr-sampler.git
cd qr-sampler
pip install -e ".[dev]"

# Everything at once (lint, format, types, security, tests) — CI runs this too
python scripts/check.py

# Subsets
python scripts/check.py --only lint,format
python scripts/check.py --only tests

# Pre-commit hooks (invoke the same checks)
pre-commit install
pre-commit run --all-files
```

See `AGENTS.md` for the architecture invariants and extension recipes, and `CONTRIBUTING.md` for the contribution workflow.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
