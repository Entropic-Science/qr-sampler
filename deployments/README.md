# Deployment Profiles

Each deployment profile is a self-contained directory that configures qr-sampler
to connect to a specific entropy source. Every profile contains:

- **`docker-compose.yml`** -- Complete Docker Compose file. Runs vLLM with
  qr-sampler and any co-located entropy servers. No external files needed.
- **`.env.example`** -- Annotated environment variable template. Copy to `.env`
  and customize.
- **`README.md`** -- Setup guide specific to this entropy source.

## Available profiles

| Profile | Entropy source | What it runs |
|---------|---------------|--------------|
| [`urandom/`](urandom/) | `os.urandom()` via gRPC | vLLM + co-located gRPC entropy server |
| [`firefly-1/`](firefly-1/) | Quantum RNG via gRPC | vLLM only (QRNG server is external) |
| [`_template/`](_template/) | Your custom source | Starting point for new profiles |

## Quick start (any profile)

Every profile follows the same three steps:

```bash
cd deployments/<profile>
cp .env.example .env      # then edit .env with your settings
docker compose up --build
```

For example, to run with the urandom entropy server:

```bash
cd deployments/urandom
cp .env.example .env
docker compose up --build
```

To also start [Open WebUI](https://github.com/open-webui/open-webui) (a
ChatGPT-style web interface), add `--profile ui`:

```bash
docker compose --profile ui up --build
```

Then open http://localhost:3000. See each profile's README for UI setup
details.

## Creating your own profile

1. Copy the template:

   ```bash
   cp -r deployments/_template deployments/my-server
   ```

2. Edit `deployments/my-server/.env.example` with your server's address, gRPC
   method paths, and authentication settings. Then:

   ```bash
   cd deployments/my-server
   cp .env.example .env
   ```

3. If your entropy server runs as a container alongside vLLM, uncomment the
   `entropy-server` service in `docker-compose.yml`.

4. Start:

   ```bash
   docker compose up --build
   ```

See [`_template/README.md`](_template/README.md) for full details.

## Security notes

- Profile `.env` files may contain API keys or other credentials.
- The `_template/` and `urandom/` profiles contain **no secrets** and are
  always safe to commit.
- If your profile contains credentials you do not want in version control,
  add its folder name to `deployments/.gitignore`.
- qr-sampler never logs the `QR_GRPC_API_KEY` value. Health checks report
  only whether authentication is enabled (`"authenticated": true/false`).
