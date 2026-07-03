# OpenEntropy Profile

Runs vLLM with qr-sampler using **OpenEntropy** — a local hardware entropy
source that collects noise from 63 hardware sources on Apple Silicon (thermal,
timing, microarchitecture, GPU, etc.). This is a **native-only profile** — no
Docker, no network dependency.

## Why not Docker?

Docker containers cannot access Metal GPU or native hardware entropy sources on
macOS. Apple's Virtualization.framework has no GPU passthrough, and hardware
noise sources (thermal sensors, CPU timing, GPU state) are not exposed to
containerized processes. OpenEntropy requires native execution.

## Quick start

1. Install OpenEntropy and qr-sampler:

   ```bash
   pip install openentropy
   pip install -e /path/to/qr-sampler
   ```

2. Configure your environment:

   ```bash
   cd deployments/openentropy
   cp .env.example .env
   ```

   Edit `.env` if needed — set `HF_TOKEN` if using a gated model.

3. Start vLLM:

   ```bash
   source .env
   vllm serve $HF_MODEL \
     --port $VLLM_PORT \
     --logits-processors qr_sampler
   ```

## Available entropy sources

OpenEntropy provides 63 entropy sources across 13 categories. For the full
catalog with physics explanations, see the
[OpenEntropy Source Catalog](https://github.com/amenti-labs/openentropy/blob/master/docs/SOURCES.md).

List all available sources on your hardware:

```bash
python -c "from openentropy import detect_available_sources; print([s['name'] for s in detect_available_sources()])"
```

Sources span thermal, timing, microarchitecture, GPU, IPC, scheduling, and more.
Some notable ones for research:

| Source | Category | Physical mechanism |
|--------|----------|-------------------|
| `counter_beat` | Thermal | CPU counter vs audio PLL crystal beat frequency |
| `dual_clock_domain` | Microarch | 24 MHz x 41 MHz independent oscillator beat |
| `gpu_divergence` | GPU | Shader warp execution order divergence |
| `dvfs_race` | Microarch | Cross-core DVFS frequency race |
| `clock_jitter` | Timing | Timing jitter between readout paths |
| `dram_row_buffer` | Timing | DRAM row buffer hit/miss timing |

To sample from a specific source, set `QR_OE_SOURCES`:

```bash
export QR_OE_SOURCES=counter_beat
```

## Conditioning modes

OpenEntropy supports three conditioning strategies:

| Mode | Use case | Properties |
|------|----------|-----------|
| `raw` | Research (default) | Preserves hardware noise signal; minimal processing |
| `vonneumann` | Debiased entropy | Von Neumann debiasing; slower, more uniform |
| `sha256` | Cryptographic | SHA-256 hashing; suitable for security-critical applications |

Set `QR_OE_CONDITIONING` in `.env` or override per-request:

```python
# Per-request override
extra_args = {"qr_oe_conditioning": "sha256"}
```

## Parallel collection

By default, `QR_OE_PARALLEL=true` collects from multiple sources simultaneously,
increasing entropy throughput. Set to `false` for sequential collection (slower,
lower memory overhead).

## When to use this profile

- **Consciousness research**: Study whether intent influences quantum-random
  processes using native hardware entropy.
- **Local experiments**: No network latency, no external dependencies.
- **Apple Silicon development**: Leverage Metal GPU and native hardware sensors.
- **Research baseline**: Compare hardware entropy against system entropy
  (`/dev/urandom`).

## Web UI (optional)

This profile includes [Open WebUI](https://github.com/open-webui/open-webui), a
ChatGPT-style web interface. To use it, you'll need to run it separately (not
included in this native profile):

```bash
docker run -d -p 3000:3000 --name open-webui ghcr.io/open-webui/open-webui:latest
```

Then point it at your vLLM instance running on `localhost:8000`.

Adjust qr-sampler parameters with per-request `extra_args` (`qr_*` keys) or
`QR_*` environment variables — see the configuration reference in the
top-level README.

## Next steps

Once this profile works, you can:
1. Adjust `QR_OE_SOURCES` to use specific entropy sources.
2. Experiment with different conditioning modes (`raw`, `vonneumann`, `sha256`).
3. Compare results against the `urandom` profile (gRPC-based) or `system` profile
   (fallback).
4. Browse the full [OpenEntropy Source Catalog](https://github.com/amenti-labs/openentropy/blob/master/docs/SOURCES.md)
   for detailed physics explanations of each entropy source.
