# urandom Profile

Runs a co-located gRPC entropy server that generates random bytes from
`os.urandom()`. This is the recommended starting point for testing the full
gRPC entropy pipeline before connecting to real quantum random number generator
hardware.

## Quick start

1. Configure your environment:

   ```bash
   cd deployments/urandom
   cp .env.example .env
   ```

   Edit `.env` if needed — set `HF_TOKEN` if using a gated model.

2. Start:

   ```bash
   docker compose up --build
   ```

3. Send a request:

   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen2.5-1.5B-Instruct",
       "prompt": "The nature of randomness is",
       "max_tokens": 50
     }'
   ```

## What this runs

- **entropy-server** -- Lightweight gRPC server container running
  `simple_urandom_server.py`. Listens on port 50051.
- **vllm** -- vLLM inference server with qr-sampler plugin. Fetches entropy
  from `entropy-server:50051` via Docker networking.

Both containers start together. The entropy server starts first; vLLM waits
for it via `depends_on`.

> **Preset default & fail-safe:** the compose file defaults to
> `QR_PRESET=qthought_purity` (server-draw mode). The urandom example server
> only implements `EntropyService`, so the draw path degrades fail-safe to
> local byte amplification (`entropy_is_fallback: true` in the records) —
> everything keeps working. To opt out, set `QR_PRESET` to another preset
> name (e.g. `qthought_think`) in `.env` — Compose's `${QR_PRESET:-…}`
> default treats an *empty* value the same as unset, so `QR_PRESET=` does
> NOT opt out. To run with no preset at all, remove the `QR_PRESET` line
> from the compose `environment:` block.

## Switching to bidirectional streaming

For lower latency, change `QR_GRPC_MODE` in your `.env` file:

```
QR_GRPC_MODE=bidi_streaming
```

The urandom server supports all three transport modes (unary, server streaming,
bidirectional streaming). Bidirectional streaming keeps a persistent gRPC
connection open, reducing per-token overhead to ~50-100us on the same machine.

## When to use this profile

- **Getting started**: Validate that the full entropy pipeline (gRPC fetch,
  signal amplification, token selection) works end-to-end.
- **Baseline experiments**: Compare system-entropy sampling against
  gRPC-sourced sampling to measure the overhead.
- **Development**: Test changes to the gRPC client without needing real
  hardware.

## Web UI (optional)

This profile includes [Open WebUI](https://github.com/open-webui/open-webui), a
ChatGPT-style web interface. It is not started by default — enable it with the
`ui` Docker Compose profile:

```bash
docker compose --profile ui up --build
```

Open http://localhost:3000 to start chatting. Open WebUI connects to vLLM
automatically — no configuration needed.

### Controlling qr-sampler parameters

Use per-request `extra_args` (`qr_*` keys) or `QR_*` environment
variables — see the configuration reference in the top-level README.

### Customizing the UI

| Setting | `.env` variable | Default |
|---------|----------------|---------|
| Port | `OPEN_WEBUI_PORT` | `3000` |
| Authentication | `OPEN_WEBUI_AUTH` | `false` |

Set `OPEN_WEBUI_AUTH=true` if the server is accessible by others.

## Next steps

Once this profile works, you can:
1. Copy `_template/` to create a profile for your own entropy server.
2. Point `QR_GRPC_SERVER_ADDRESS` at real QRNG hardware.
3. Set up API key authentication if your server requires it.
