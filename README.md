# qr-sampler

**Plug any randomness source into LLM token sampling.**

qr-sampler is an engine-agnostic framework that replaces standard pseudorandom token sampling with entropy from external sources — quantum random number generators (QRNGs), processor timing jitter, hardware noise, or any source you connect via gRPC or Python plugin. The core sampling pipeline has zero inference-engine dependencies; thin engine adapters integrate it with [vLLM](https://github.com/vllm-project/vllm), [vLLM-Metal](https://github.com/vllm-project/vllm-metal), or any engine that supports logits processing.

```
pip install qr-sampler
```

---

## Why qr-sampler?

Standard LLM inference uses pseudorandom number generators (PRNGs) for token sampling. PRNGs are deterministic — given the same seed, they produce the same output every time. qr-sampler replaces this with *true* randomness from physical processes:

- **Quantum RNGs** — photon detectors, vacuum fluctuation devices, or any hardware QRNG over gRPC
- **Hardware noise** — 63 thermal, timing, microarch, and GPU noise sources via [OpenEntropy](https://github.com/amenti-labs/openentropy)
- **Processor timing jitter** — CPU clock variations as an entropy source (experimental)
- **Your own source** — implement the `EntropySource` ABC or connect any hardware via the gRPC protocol
- **OS entropy** — `os.urandom()` as a fallback or baseline

### Consciousness-research context

qr-sampler provides infrastructure for studying whether conscious intent can influence quantum-random processes in LLM token selection. The signal amplification system converts thousands of random bytes into a single token choice, designed so that even a tiny statistical bias (e.g., 0.1% shift in byte means) produces a measurable effect on which token gets selected. All entropy is generated **just-in-time** — the quantum measurement happens *after* logits are computed, never before.

This is a research tool. It makes no claims about consciousness or quantum mechanics — it provides the infrastructure to run rigorous experiments.

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
  │  │  (future: MLXAdapter, TRTLLMAdapter, ...)   │     │
  │  └─────────────────────────────────────────────┘     │
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
  │   (fixed or entropy-dependent)    from the logit distribution
  │
  ├─ Entropy source ───────────── Fetch fresh random bytes
  │   (gRPC / system / timing /       just-in-time, after logits exist
  │    openentropy / custom)
  │
  ├─ Signal amplification ─────── Convert 20,480 bytes → one float u ∈ (0,1)
  │   (z-score or ECDF)               via statistical aggregation
  │
  ├─ Token selector ───────────── top-k → softmax → top-p → CDF → select
  │   (CDF binary search with u)      token from probability distribution
  │
  └─ Force one-hot logits ─────── Set selected token to 0.0, all others to -inf
      (engine picks exactly              (returned as numpy; adapter converts
       this token)                        to engine-native tensor)
```

The core pipeline is importable and functional without vLLM, torch, or any engine package. Engine adapters convert between engine-native tensors and numpy, delegate to `SamplingPipeline.sample_token()`, and write the result back.

---

## Quick start

### 1. Validate your stack (optional)

The CLI checks compatibility of engines, models, entropy sources, and amplifiers before you deploy:

```bash
pip install qr-sampler[cli]

# Check a specific combination
qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct --entropy quantum_grpc

# Exit codes: 0 = all known-working, 1 = untested, 2 = incompatible
```

### 2. Generate deployment files

```bash
# Generate Docker Compose for vLLM + quantum gRPC entropy
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy

# Preview without writing files
qr-sampler build --engine vllm --entropy system --dry-run
```

This renders a `docker-compose.yml` and `.env` from built-in Jinja2 templates, configured for your chosen stack.

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
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

### Bare-metal install (without Docker)

```bash
pip install qr-sampler

# Start vLLM — qr-sampler registers automatically via entry points
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 --gpu-memory-utilization 0.80
```

Configure the entropy source via environment variables:

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 --gpu-memory-utilization 0.80
```

### Apple Silicon (macOS)

qr-sampler works on Apple Silicon via [vllm-metal](https://github.com/vllm-project/vllm-metal), a community-maintained vLLM plugin under the official `vllm-project` GitHub org. It uses MLX under the hood but exposes the same vLLM API and plugin system — same entry points, same endpoints, same `curl` commands.

vllm-metal works with MLX-format models from the [mlx-community](https://huggingface.co/mlx-community) collection on Hugging Face. These are pre-converted and quantized for Apple Silicon — pick one that fits your available memory.

> **Prerequisite:** vllm-metal currently does not load custom logits processors registered via entry points — it creates an empty `LogitsProcessors()` instead of calling `build_logitsprocs()`. [PR #124](https://github.com/vllm-project/vllm-metal/pull/124) fixes this with a 9-line patch that mirrors `GPUModelRunner`'s pattern. Until it is merged, you will need to apply the patch manually or install from the PR branch. Without it, qr-sampler's plugin will be silently skipped.

#### 1. Install vllm-metal

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

This creates a virtual environment at `~/.venv-vllm-metal` with vLLM and all dependencies. Requires Python 3.12+.

#### 2. Install qr-sampler

```bash
source ~/.venv-vllm-metal/bin/activate
pip install qr-sampler
```

#### 3. Start the server

```bash
source ~/.venv-vllm-metal/bin/activate
vllm serve mlx-community/Qwen3-0.6B-4bit
```

qr-sampler registers automatically via the same `vllm.logits_processors` entry point. Look for this line in the server logs to confirm the plugin is active:

```
QRSamplerLogitsProcessor initialized: vocab_size=..., entropy_source=system+system, amplifier=zscore_mean, temperature=fixed
```

#### 4. Send a request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3-0.6B-4bit",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100
  }'
```

All configuration (entropy sources, temperature strategies, per-request overrides) works identically to the NVIDIA setup. The only difference is how vLLM itself is installed.

> **Note:** Docker deployment profiles are not compatible with Apple Silicon — Docker on macOS runs a Linux VM with no Metal GPU passthrough. vllm-metal must run natively.

---

## CLI reference

The CLI requires the `[cli]` extra: `pip install qr-sampler[cli]`

### `qr-sampler list`

Browse available components:

```bash
qr-sampler list engines              # Engine profiles (vllm, vllm_metal)
qr-sampler list models --engine vllm # Known-working models for an engine
qr-sampler list entropy-sources      # All entropy source profiles
qr-sampler list amplifiers           # Signal amplification algorithms
qr-sampler list samplers             # Temperature strategies
qr-sampler list presets              # Preset bundles (creative_sampling, normal_t1)
```

### `qr-sampler info`

Detailed information about a specific component:

```bash
qr-sampler info engine vllm
qr-sampler info entropy quantum_grpc
qr-sampler info amplifier zscore_mean
qr-sampler info sampler edt
qr-sampler info preset creative_sampling
```

### `qr-sampler validate`

Check stack compatibility before deploying:

```bash
# Check specific combination
qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct

# With a config file
qr-sampler validate --config stack.yaml
```

Exit codes: `0` = all known-working, `1` = untested combinations (warnings), `2` = incompatible or missing.

### `qr-sampler build`

Generate Docker Compose deployment files:

```bash
# Generate files
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy

# Preview without writing
qr-sampler build --engine vllm --entropy system --dry-run

# Bypass compatibility warnings
qr-sampler build --engine vllm --entropy timing_noise --force --output ./deploy

# From a config file
qr-sampler build --config stack.yaml --output ./deploy
```

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

Connect any entropy server that speaks gRPC. qr-sampler ships example servers and an annotated template:

```bash
# Run the built-in urandom-over-gRPC example server
pip install qr-sampler
python examples/servers/simple_urandom_server.py --address 0.0.0.0:50051

# Point qr-sampler at it
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
```

Three transport modes:

| Mode | `QR_GRPC_MODE` | Latency | Best for |
|---|---|---|---|
| **Unary** | `unary` | ~1-2ms | Simplicity, debugging |
| **Server streaming** | `server_streaming` | ~0.5-1ms | Middle ground |
| **Bidirectional** | `bidi_streaming` | ~50-100us (same machine) | Production, lowest latency |

For co-located hardware, use Unix domain sockets:

```bash
python my_qrng_server.py --address unix:///var/run/qrng.sock
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
export QR_GRPC_MODE=bidi_streaming
```

The gRPC client is **protocol-agnostic**. It uses configurable method paths and generic protobuf wire-format encoding. The only requirement is that your proto puts the byte count as field 1 in the request and the random bytes as field 1 in the response. Configure custom protos via `QR_GRPC_METHOD_PATH` and `QR_GRPC_STREAM_METHOD_PATH`.

#### Circuit breaker

The gRPC client includes an adaptive circuit breaker:

- Tracks rolling P99 latency over the last 100 requests; timed-out
  fetches also feed the window, so the ceiling re-learns upward when
  the backend genuinely slows
- Sets timeout to `max(5ms, P99 * 1.5)` (configurable via `QR_CB_*`
  env vars; raise `QR_CB_MIN_TIMEOUT_MS` for tunnelled/remote backends)
- Opens after 3 consecutive failures, enters half-open state after 10s;
  the half-open attempt resets the channel first so a stale connection
  can't waste the recovery cycle
- Falls back to `QR_FALLBACK_MODE` when the circuit is open

### OpenEntropy

[OpenEntropy](https://github.com/amenti-labs/openentropy) harvests entropy from 63 hardware noise sources on the local machine — thermal sensors, CPU timing jitter, memory timing, GPU scheduling, and more. No network, no API keys, no gRPC server needed.

```bash
pip install openentropy
export QR_ENTROPY_SOURCE_TYPE=openentropy
export QR_OE_CONDITIONING=raw   # raw (research default) | vonneumann | sha256
```

List available sources on your machine:

```python
from openentropy import detect_available_sources
print([s["name"] for s in detect_available_sources()])
```

### Fallback behavior

The `FallbackEntropySource` wraps a primary source with an automatic fallback:

- Only catches `EntropyUnavailableError` — other exceptions propagate
- Logs a warning when fallback is used
- Exposes `last_source_used` for diagnostics

Configure with `QR_FALLBACK_MODE`:
- `system` — fall back to `os.urandom()` (default)
- `mock_uniform` — fall back to the mock source
- `error` — raise immediately, no fallback

### Third-party entropy sources

Any Python package can register entropy sources via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."qr_sampler.entropy_sources"]
lava_lamp = "my_package:LavaLampEntropySource"
```

The source is auto-discovered when qr-sampler starts. See [Setting up your own entropy source](#setting-up-your-own-entropy-source) below.

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
20,480 bytes with +0.003 mean bias per byte:
  M ~ 127.56, SEM ~ 0.514, z ~ 0.12, u ~ 0.548
```

### ECDF (`ecdf`)

Empirical CDF amplifier with online calibration. Maps raw bytes to uniform via a calibrated empirical distribution function. Does not assume a specific input distribution — it calibrates from observed data.

---

## Temperature strategies

### Fixed temperature (`fixed`)

Returns a constant temperature for every token. Set via `QR_FIXED_TEMPERATURE` (default: 0.7).

### Entropy-dependent temperature (`edt`)

Dynamically adjusts temperature based on the Shannon entropy of the logit distribution:

```
H_norm = H / ln(vocab_size)         # Normalized entropy [0, 1]
T = base_temp * H_norm^exponent     # Power-law scaling
T = clamp(T, min_temp, max_temp)    # Bounds enforcement
```

High-entropy (uncertain) distributions get higher temperatures; low-entropy (confident) distributions get lower temperatures. This creates a feedback loop where the model's own uncertainty calibrates the randomness of selection.

---

## Creative Sampling (experimental)

`creative_sampling` is an **experimental** preset bundling the `hvh_drift` temperature strategy plus a dynamic per-token `min_p` mask. The default hyperparameters reproduce the winning configuration (`V6_HVD_R01_01`) from the v6 phase of the [createmp-evalsuite](https://github.com/alchemystack/createmp-evalsuite) sampler-evaluation testbed. The strategy tracks Shannon entropy (`H`) and varentropy (`VH`) of each step's distribution, maintains per-request EMAs of both, and uses the drift between the current and EMA values to drive temperature and min-p adjustments token-by-token.

Two presets ship out of the box:

| Preset | What it does | Status |
|---|---|---|
| `creative_sampling` | `hvh_drift` strategy with V6 winner hyperparameters + dynamic min-p | **Experimental** |
| `normal_t1` | Vanilla `fixed` strategy at T=1, no top-k or top-p filtering | Stable baseline |

### Process-wide via env var

```bash
QR_PRESET=creative_sampling python -m vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

### Per-request via `extra_args`

```python
client.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompt="The nature of consciousness is",
    max_tokens=100,
    extra_body={"extra_args": {"qr_preset": "creative_sampling"}},
)
```

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
    "max_tokens": 100,
    "extra_args": {"qr_preset": "creative_sampling"}
  }'
```

### Precedence

Per-request `qr_preset` beats the `QR_PRESET` env var. Per-request `qr_*` keys override individual values inside whichever preset is active, so you can pin a single hyperparameter without abandoning the preset:

```json
{"extra_args": {"qr_preset": "creative_sampling", "qr_hvh_t_base": 1.2}}
```

### Open WebUI toggle

The bundled [Open WebUI filter](examples/open-webui/) exposes a per-user `UserValves.preset` dropdown so end users can flip between `creative_sampling` (experimental) and `normal_t1` from the chat UI without touching admin settings. See the filter README for installation steps.

### Inspecting a preset

```bash
qr-sampler list presets             # Both presets, with experimental flag and origin
qr-sampler info preset creative_sampling
```

> **Heads up:** `creative_sampling` is research-grade. It changes token distributions in non-obvious ways and carries per-request state. Pin to `normal_t1` (or omit `qr_preset` entirely) for reproducible baselines.

---

## Per-request parameter overrides

Override sampling parameters on individual requests via `extra_args`:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "prompt": "The nature of consciousness is",
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

Infrastructure fields (`QR_GRPC_SERVER_ADDRESS`, `QR_FALLBACK_MODE`, etc.) cannot be overridden per-request — they are set at server startup.

---

## Web UI

qr-sampler works with [Open WebUI](https://github.com/open-webui/open-webui), a self-hosted ChatGPT-style interface that connects to vLLM's OpenAI-compatible API.

**Docker (NVIDIA / Linux):** The `qr-sampler build` command includes Open WebUI in the generated Compose file:

```bash
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy
cd deploy
docker compose up --build
# Open http://localhost:3000
```

**Apple Silicon:** Run Open WebUI standalone in Docker, pointing at your vllm-metal server:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=not-needed \
  -e WEBUI_AUTH=false \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### Controlling qr-sampler from the UI

A pre-built [filter function](examples/open-webui/) injects qr-sampler per-request parameters into every chat message via the Open WebUI Valves system. This lets you adjust temperature, top-k, top-p, sample count, and other sampling parameters from the admin panel.

To set it up:

1. Go to **Admin Panel > Functions** in Open WebUI.
2. Click **Import** and select [`examples/open-webui/qr_sampler_filter.json`](examples/open-webui/qr_sampler_filter.json).
3. Toggle the function to **Global**.
4. Click the **gear icon** to adjust parameters.

See [`examples/open-webui/README.md`](examples/open-webui/README.md) for the full guide.

> Open WebUI is entirely optional. qr-sampler works the same way with direct API calls, `curl`, Python clients, or any OpenAI-compatible tool.

---

## Configuration reference

All configuration is done via environment variables with the `QR_` prefix. Per-request overrides use the `qr_` prefix in `extra_args`.

### Infrastructure fields (NOT per-request overridable)

| Environment variable | Default | Description |
|---|---|---|
| `QR_ENTROPY_SOURCE_TYPE` | `system` | Primary entropy source identifier |
| `QR_GRPC_SERVER_ADDRESS` | `localhost:50051` | gRPC entropy server address (`host:port` or `unix:///path`) |
| `QR_GRPC_TIMEOUT_MS` | `5000` | gRPC call timeout in milliseconds |
| `QR_GRPC_RETRY_COUNT` | `2` | Retry attempts after gRPC failure |
| `QR_GRPC_MODE` | `unary` | Transport mode: `unary`, `server_streaming`, `bidi_streaming` |
| `QR_GRPC_METHOD_PATH` | `/qr_entropy.EntropyService/GetEntropy` | gRPC method path for unary RPC |
| `QR_GRPC_STREAM_METHOD_PATH` | `/qr_entropy.EntropyService/StreamEntropy` | gRPC method path for streaming RPC (empty disables streaming) |
| `QR_GRPC_API_KEY` | *(empty)* | API key sent via gRPC metadata (empty = no auth) |
| `QR_GRPC_API_KEY_HEADER` | `api-key` | gRPC metadata header name for the API key |
| `QR_FALLBACK_MODE` | `system` | Fallback when primary fails: `error`, `system`, `mock_uniform` |
| `QR_CB_WINDOW_SIZE` | `100` | Rolling latency window size for P99 computation |
| `QR_CB_MIN_TIMEOUT_MS` | `5.0` | Minimum adaptive timeout in milliseconds |
| `QR_CB_TIMEOUT_MULTIPLIER` | `1.5` | Multiplier applied to P99 latency for adaptive timeout |
| `QR_CB_RECOVERY_WINDOW_S` | `10.0` | Seconds before half-open retry after circuit opens |
| `QR_CB_MAX_CONSECUTIVE_FAILURES` | `3` | Consecutive failures before circuit breaker opens |
| `QR_ENTROPY_STATUS_FILE` | `<tempdir>/qr_entropy_status.json` | Cross-process entropy-status file (powers `/health/entropy` in split-process engines like vLLM); empty string disables |

### Sampling parameters (per-request overridable)

| Environment variable | extra_args key | Default | Description |
|---|---|---|---|
| `QR_SIGNAL_AMPLIFIER_TYPE` | `qr_signal_amplifier_type` | `zscore_mean` | Signal amplification algorithm |
| `QR_SAMPLE_COUNT` | `qr_sample_count` | `10000` | Entropy bytes fetched per token |
| `QR_POPULATION_MEAN` | `qr_population_mean` | `127.5` | Null-hypothesis mean for byte values |
| `QR_POPULATION_STD` | `qr_population_std` | `73.612...` | Population std for uniform [0, 255] |
| `QR_UNIFORM_CLAMP_EPSILON` | `qr_uniform_clamp_epsilon` | `1e-10` | Clamp u to avoid degenerate CDF |
| `QR_TEMPERATURE_STRATEGY` | `qr_temperature_strategy` | `fixed` | Strategy: `fixed` or `edt` |
| `QR_FIXED_TEMPERATURE` | `qr_fixed_temperature` | `0.7` | Constant temperature (fixed strategy) |
| `QR_EDT_BASE_TEMP` | `qr_edt_base_temp` | `0.8` | Base coefficient for EDT |
| `QR_EDT_EXPONENT` | `qr_edt_exponent` | `0.5` | Power-law exponent for EDT |
| `QR_EDT_MIN_TEMP` | `qr_edt_min_temp` | `0.1` | EDT temperature floor |
| `QR_EDT_MAX_TEMP` | `qr_edt_max_temp` | `2.0` | EDT temperature ceiling |
| `QR_TOP_K` | `qr_top_k` | `50` | Top-k filtering (`<=0` disables) |
| `QR_TOP_P` | `qr_top_p` | `0.9` | Nucleus sampling threshold (`1.0` disables) |
| `QR_LOG_LEVEL` | `qr_log_level` | `summary` | Logging: `none`, `summary`, `full` |
| `QR_DIAGNOSTIC_MODE` | `qr_diagnostic_mode` | `false` | Store all token records in memory |

You can also use a `.env` file — pydantic-settings loads it automatically.

---

## Setting up your own entropy source

qr-sampler is designed to connect *any* randomness source to LLM token sampling. There are two approaches.

### Approach A: gRPC server (recommended)

Implement a gRPC server. You can use the built-in `qr_entropy.EntropyService` protocol (example servers provided), or your own proto as long as field 1 carries the byte count (request) and random bytes (response).

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
pip install qr-sampler
python my_qrng_server.py --port 50051
```

4. **Deploy with Docker:**

```bash
# Generate deployment files
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy

# Configure the gRPC address
cd deploy
# Edit .env: QR_GRPC_SERVER_ADDRESS=<your-server>:50051
docker compose up --build
```

Or configure directly via environment variables (bare-metal):

```bash
export QR_ENTROPY_SOURCE_TYPE=quantum_grpc
export QR_GRPC_SERVER_ADDRESS=localhost:50051
vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 8192 --gpu-memory-utilization 0.80
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

Any language that supports gRPC can implement this server — Python, C++, Rust, Go, etc.

#### Just-in-time constraint

The entropy must be generated **after** the client sends the request, not from a pre-generated pool:

- No buffering or caching of previously generated bytes
- The physical measurement happens during the `generate()` call
- `generation_timestamp_ns` in the response proves freshness

This is critical for consciousness-research applications where the timing relationship between logit computation and entropy generation matters.

#### Deployment options

**systemd (Linux):**

```bash
sudo cp examples/systemd/qr-entropy-server.service /etc/systemd/system/
sudo cp examples/systemd/qr-entropy-server.env /etc/default/qr-entropy-server
sudo systemctl enable --now qr-entropy-server
```

**Unix domain sockets** (lowest latency for co-located hardware):

```bash
python my_qrng_server.py --address unix:///var/run/qrng.sock
export QR_GRPC_SERVER_ADDRESS=unix:///var/run/qrng.sock
```

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
from qr_sampler.entropy.quantum import QuantumGrpcSource
from qr_sampler.config import QRSamplerConfig

config = QRSamplerConfig(
    entropy_source_type="quantum_grpc",
    grpc_server_address="localhost:50051",
)
source = QuantumGrpcSource(config)

# Basic connectivity
data = source.get_random_bytes(1024)
assert len(data) == 1024

# Health check
status = source.health_check()
print(status)  # {'source': 'quantum_grpc', 'healthy': True, ...}

source.close()
```

For statistical validation, check that your source produces uniform byte distributions:

```python
import numpy as np
from scipy import stats

data = source.get_random_bytes(100_000)
samples = np.frombuffer(data, dtype=np.uint8)

# KS test against uniform distribution
stat, p_value = stats.kstest(samples / 255.0, 'uniform')
print(f"KS statistic: {stat:.6f}, p-value: {p_value:.6f}")
# p-value should be > 0.05 for a good entropy source
```

---

## Plugin architecture

qr-sampler uses a registry + entry-points pattern for extensibility:

```
qr_sampler.entropy_sources          Third-party entropy sources
qr_sampler.engine_adapters          Third-party engine adapters
vllm.logits_processors              vLLM plugin registration
```

Each subsystem (entropy, amplification, temperature, engines) has its own registry with decorator-based registration for built-in implementations and entry-point discovery for third-party extensions. The pipeline never instantiates strategy classes directly — it always goes through the registry.

### Adding new components

**New engine adapter:** Subclass `EngineAdapter`, implement `get_pipeline()`. Register with `@EngineAdapterRegistry.register("name")`. Add entry point under `qr_sampler.engine_adapters`.

**New entropy source:** Subclass `EntropySource`, implement `name`, `is_available`, `get_random_bytes()`, `close()`. Register with `@register_entropy_source("name")`.

**New signal amplifier:** Subclass `SignalAmplifier`, implement `amplify(raw_bytes) -> AmplificationResult`. Register with `@AmplifierRegistry.register("name")`.

**New temperature strategy:** Subclass `TemperatureStrategy`, implement `compute_temperature(logits, config) -> TemperatureResult`. Always compute Shannon entropy. Register with `@TemperatureStrategyRegistry.register("name")`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development instructions.

---

## Project structure

```
src/qr_sampler/
├── __init__.py                    # Package version, re-exports
├── __main__.py                    # CLI entry: python -m qr_sampler
├── config.py                      # Pydantic-settings configuration
├── exceptions.py                  # Exception hierarchy
├── presets.py                     # BUILTIN_PRESETS + extra_args expansion (creative_sampling, normal_t1)
├── processor.py                   # Backward compat: re-exports VLLMAdapter
├── py.typed                       # PEP 561 type hint marker
├── core/                          # Engine-agnostic pipeline (NO torch)
│   ├── pipeline.py                # SamplingPipeline + factory functions
│   └── types.py                   # SamplingResult frozen dataclass
├── engines/                       # Engine adapter layer
│   ├── base.py                    # EngineAdapter ABC
│   ├── registry.py                # EngineAdapterRegistry
│   └── vllm.py                    # VLLMAdapter: vLLM V1 LogitsProcessor
├── profiles/                      # Declarative YAML metadata
│   ├── schema.py                  # Pydantic validation models
│   ├── loader.py                  # Profile discovery + loading
│   ├── compatibility.py           # Tri-state compatibility checker
│   ├── engines/                   # Engine profiles (vllm, vllm_metal)
│   ├── entropy/                   # Entropy profiles (system, quantum_grpc, ...)
│   ├── amplifiers/                # Amplifier profiles (zscore_mean, ecdf)
│   ├── samplers/                  # Sampler profiles (fixed, edt)
│   └── presets/                   # Preset profiles (creative_sampling, normal_t1)
├── cli/                           # CLI commands (requires [cli] extra)
│   ├── main.py                    # Click group
│   ├── validate_cmd.py            # qr-sampler validate
│   ├── build_cmd.py               # qr-sampler build
│   ├── list_cmd.py                # qr-sampler list
│   └── info_cmd.py                # qr-sampler info
├── templates/                     # Jinja2 templates for qr-sampler build
│   ├── docker-compose.yml.j2
│   └── env.j2
├── amplification/
│   ├── base.py                    # SignalAmplifier ABC, AmplificationResult
│   ├── registry.py                # AmplifierRegistry
│   ├── zscore.py                  # Z-score mean amplifier
│   └── ecdf.py                    # ECDF amplifier
├── entropy/
│   ├── base.py                    # EntropySource ABC
│   ├── registry.py                # Auto-discovery registry + entry points
│   ├── quantum.py                 # gRPC QRNG (3 transport modes, circuit breaker)
│   ├── system.py                  # os.urandom()
│   ├── timing.py                  # CPU timing jitter (experimental)
│   ├── mock.py                    # Configurable test source
│   ├── fallback.py                # Fallback composition wrapper
│   └── openentropy.py             # OpenEntropy integration
├── logging/
│   ├── types.py                   # TokenSamplingRecord (frozen, __slots__)
│   └── logger.py                  # SamplingLogger (none/summary/full)
├── proto/
│   ├── entropy_service.proto      # gRPC protocol definition
│   ├── entropy_service_pb2.py     # Hand-written protobuf stubs
│   └── entropy_service_pb2_grpc.py
├── selection/
│   ├── types.py                   # SelectionResult dataclass
│   └── selector.py                # CDF-based token selector
└── temperature/
    ├── base.py                    # TemperatureStrategy ABC, Shannon entropy
    ├── registry.py                # TemperatureStrategyRegistry
    ├── fixed.py                   # Fixed temperature
    ├── edt.py                     # Entropy-dependent temperature
    └── hvh_drift.py               # Stateful per-request EMA-driven temperature + dynamic min-p (creative_sampling)

examples/
├── servers/                       # Example entropy servers
│   ├── simple_urandom_server.py   # Minimal reference (~50 lines)
│   ├── timing_noise_server.py     # CPU timing jitter
│   └── qrng_template_server.py    # Annotated template for custom QRNGs
├── open-webui/                    # Open WebUI filter function
├── docker/                        # Dockerfiles
└── systemd/                       # systemd unit files
```

---

## Statistical analysis

qr-sampler includes statistical tests (in `tests/test_statistical_properties.py`, requires `scipy`) that validate the mathematical properties of the sampling pipeline:

- **KS-test for u-value uniformity**: Under the null hypothesis (no bias), amplified `u` values should be uniformly distributed on (0, 1).
- **Bias detection**: Verifies that a small per-byte mean shift produces a statistically detectable shift in the `u` distribution — confirming the amplification system is sensitive enough for consciousness-research experiments.
- **EDT monotonicity**: Validates that the entropy-dependent temperature strategy produces higher temperatures for higher-entropy logit distributions.

```bash
pytest tests/test_statistical_properties.py -v
```

---

## Development

```bash
git clone https://github.com/alchemystack/qr-sampler.git
cd qr-sampler
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy --strict src/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development guide.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
