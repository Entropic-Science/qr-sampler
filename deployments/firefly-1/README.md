# firefly-1 Profile

Connects to an external quantum random number generator (firefly-1 device).
No additional containers are needed — the QRNG server is external. vLLM
connects to it directly over the network.

## Quick start

1. Configure your environment:

   ```bash
   cd deployments/firefly-1
   cp .env.example .env
   ```

   Edit `.env`:
   - Set `QR_GRPC_API_KEY` to your actual API key.
   - Set `HF_TOKEN` if using a gated model.

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

## Server details

| Field | Value |
|-------|-------|
| Address | `10.0.0.115:50051` |
| Protocol | `qrng.QuantumRNG` (gRPC, unary only) |
| Device | firefly-1 |
| Authentication | API key via `api-key` metadata header |

## Rate limits

| Limit | Value |
|-------|-------|
| Requests per minute | 500 |
| Daily data transfer | 500 MB |
| Max bytes per request | 13 KB (13,312 bytes) |

## Important notes

- **Sample count**: `QR_SAMPLE_COUNT` is set to 13,312 (13 KB) to stay within
  the server's max-bytes-per-request limit. The default of 10,000 fits, but the
  larger values used in some experiments would be rejected.
- **Unary only**: `QR_GRPC_STREAM_METHOD_PATH` is empty because this server
  only supports unary RPC. Do not set `QR_GRPC_MODE` to `server_streaming` or
  `bidi_streaming`.
- **API key**: The `.env` file contains a real API key. If you fork this repo
  publicly, add `firefly-1/` to `deployments/.gitignore`.
- **Preset default & fail-safe**: the compose file defaults to
  `QR_PRESET=qthought_purity` (server-draw mode). The firefly-1 server speaks
  `qrng.QuantumRNG` only (no `qr_purity.PurityService`), so the draw path
  degrades fail-safe to local byte amplification — sampling keeps working.
  To opt out, set `QR_PRESET` to another preset name (e.g. `qthought_think`)
  in `.env` — Compose's `${QR_PRESET:-…}` default treats an *empty* value the
  same as unset, so `QR_PRESET=` does NOT opt out. To run with no preset at
  all, remove the `QR_PRESET` line from the compose `environment:` block.

## Testing the connection

Verify the server is reachable before starting vLLM:

```bash
grpcurl -plaintext \
  -H 'api-key: YOUR_API_KEY_HERE' \
  -d '{"num_bytes": 100}' \
  10.0.0.115:50051 qrng.QuantumRNG/GetRandomBytes
```

## Web UI (optional)

This profile includes [Open WebUI](https://github.com/open-webui/open-webui), a
ChatGPT-style web interface. It is not started by default — enable it with the
`ui` Docker Compose profile:

```bash
docker compose --profile ui up --build
```

Open http://localhost:3000 to start chatting. Open WebUI connects to vLLM
automatically.

### Controlling qr-sampler parameters

Use per-request `extra_args` (`qr_*` keys) or `QR_*` environment
variables — see the configuration reference in the top-level README.

### Customizing the UI

| Setting | `.env` variable | Default |
|---------|----------------|---------|
| Port | `OPEN_WEBUI_PORT` | `3000` |
| Authentication | `OPEN_WEBUI_AUTH` | `false` |

Set `OPEN_WEBUI_AUTH=true` if the server is accessible by others.
