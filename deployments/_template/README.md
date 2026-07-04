# Deployment Profile Template

Starting point for creating a new deployment profile. Copy this folder and
customize it for your entropy source.

## Quick start

1. Copy this template:

   ```bash
   cp -r deployments/_template deployments/my-server
   cd deployments/my-server
   ```

2. Configure your environment:

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   - Set `QR_GRPC_SERVER_ADDRESS` to your server's address.
   - Set `QR_GRPC_METHOD_PATH` to match your server's proto service/method.
   - Set `QR_GRPC_STREAM_METHOD_PATH` (or leave empty to disable streaming).
   - Add `QR_GRPC_API_KEY` if your server requires authentication.

3. Start:

   ```bash
   docker compose up --build
   ```

## Adding a co-located entropy server

If your entropy server should run as a Docker container alongside vLLM,
uncomment the `entropy-server` service block in `docker-compose.yml` and
configure it with your server's Dockerfile or image. See
[`../urandom/docker-compose.yml`](../urandom/docker-compose.yml) for a
working example.

When using a co-located server, set the gRPC address to the Docker service
name (e.g., `entropy-server:50051`) instead of `localhost`.

## Finding the right gRPC method path

The method path format is `/<package>.<Service>/<Method>` from your `.proto`
file. For example:

```protobuf
package qr_entropy;            // package = "qr_entropy"
service EntropyService {        // service = "EntropyService"
  rpc GetEntropy (...) ...;     // method  = "GetEntropy"
}
```

Produces the path: `/qr_entropy.EntropyService/GetEntropy`

The only requirement is that:
- The **request** has the byte count as protobuf **field 1** (varint).
- The **response** has the random bytes as protobuf **field 1** (length-delimited bytes).

Any proto definition following this convention works without code changes.

## Preset default & fail-safe

The compose file defaults to `QR_PRESET=qthought_purity` (server-draw mode via
`qr_purity.PurityService`). Against a server that only implements an
entropy-bytes protocol, the draw path degrades fail-safe to local byte
amplification (`entropy_is_fallback: true` in the token records) — sampling
keeps working. To opt out, set `QR_PRESET` to another preset name (e.g.
`qthought_think`) in `.env` — note that Compose's `${QR_PRESET:-…}` default
treats an *empty* value the same as unset, so `QR_PRESET=` does NOT opt out.
To run with no preset at all, remove the `QR_PRESET` line from the compose
`environment:` block.

## Web UI (optional)

This template includes [Open WebUI](https://github.com/open-webui/open-webui), a
ChatGPT-style web interface. It is not started by default — enable it with the
`ui` Docker Compose profile:

```bash
docker compose --profile ui up --build
```

Open http://localhost:3000 to start chatting. Open WebUI connects to vLLM
automatically.

Adjust qr-sampler parameters with per-request `extra_args` (`qr_*` keys) or
`QR_*` environment variables — see the configuration reference in the
top-level README.

### Customizing the UI

| Setting | `.env` variable | Default |
|---------|----------------|---------|
| Port | `OPEN_WEBUI_PORT` | `3000` |
| Authentication | `OPEN_WEBUI_AUTH` | `false` |

Set `OPEN_WEBUI_AUTH=true` if the server is accessible by others.
