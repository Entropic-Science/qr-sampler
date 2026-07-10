# LEARNINGS

Cross-cutting notes about architectural pivots, observed runtime quirks,
and decisions whose rationale is too long for a commit message.
Companion to `AGENTS.md` (structure + invariants) and `README.md`
(end-user documentation).

---

## Live lessons (constrain current code)

### QRNG wire contract

The quantum gRPC client (`entropy/qgrpc/`) is protocol-agnostic via
configurable method paths, but field numbers MUST stay at 1 for both the
request byte count (varint) and the response `data` (length-delimited
bytes). Other proto fields can be added freely; field 1 changes are
breaking. Decoding follows proto3/pb2 semantics since the 2026-07
refactor: the LAST field-1 occurrence wins, and an explicitly empty
payload raises `EntropyUnavailableError` in the transport (test-pinned in
`tests/test_wire_format.py`).

- Third-party QRNG protos are mapped via `QR_GRPC_METHOD_PATH` (and
  `QR_GRPC_STREAM_METHOD_PATH`, empty when the proto defines no
  streaming RPC).
- Auth is a single gRPC metadata entry, header `QR_GRPC_API_KEY_HEADER`
  (default `api-key`), value `QR_GRPC_API_KEY`. The client never logs
  the key — `health_check()` redacts it.
- Some providers define no Healthcheck RPC: liveness is observed
  implicitly via per-token fetches; the circuit breaker + fallback
  wrapper own degraded conditions. The one deliberate liveness check is
  `QuantumGrpcSource.warmup()` at process start.
- Known field collision (qbert-style servers): response field 2 may be a
  server timestamp in µs rather than a `sequence_id` echo, so
  `echo_verified` stays `False` by construction there. The note lives on
  the decode site (`entropy/qgrpc/transport.py`). As of Qbert0G 1.0 the
  production server ALSO serves the native `qr_entropy.EntropyService`
  (echo + `generation_timestamp_ns` + bidi), so the collision path only
  applies when `grpc_method_path` is pointed at the legacy
  `/qrng.QuantumRNG/GetRandomBytes` method.

### Lazy channel creation is load-bearing

`QuantumGrpcSource` creates its gRPC channel lazily, never in
`__init__`. This started as a Modal-snapshot requirement (see History),
but it remains the right shape for any freeze/restore or fork-based
deployment, and it keeps `import qr_sampler` side-effect-free
(test-pinned by the import-time socket guard). Do not "optimise" channel
construction into the constructor.

### QRNG adaptive-timeout ratchet (iter-53)

Only successful fetches originally fed the adaptive-timeout P99 window,
so once P99 converged fast (~96 ms via a tunnel → ~145 ms ceiling) every
tail fetch was cut off and discarded — the ceiling could never re-learn
upward and the source flapped timeout↔fallback indefinitely. Fixes that
remain in `entropy/qgrpc/breaker.py`:

- timeout-shaped failures (≥0.8× the budget) count as latency samples;
- the half-open attempt resets the gRPC channel first (the dominant
  open cause was a stale channel — testing recovery on the suspect
  channel wasted whole cycles);
- `QR_CB_MIN_TIMEOUT_MS` defaults to 5 ms, a **localhost assumption** —
  floor it at ~300 ms for tunnelled/remote backends.

### Circuit-breaker recovery backs off exponentially (iter-57)

A short fixed recovery window is right for the stale-channel case but
hammers a genuinely-down server (each half-open rebuilds the channel and
fires fresh connects). `QR_CB_RECOVERY_WINDOW_S` is therefore the BASE
only: consecutive opens without an intervening success double the wait
(`base × 2^opens`) up to `QR_CB_RECOVERY_WINDOW_MAX_S` (60 s), reset on
first success. Pair with `QR_GRPC_RETRY_COUNT=0` in deploys where the
QRNG can be down for long stretches — each per-token retry is another
connect against a dead server; the breaker + fallback are the correct
resilience layer. Degraded logging is throttled (first-of-window + at
most once/min); the running `fallback_count` rides every record.

### Entropy health reporting is PASSIVE (iter-57 lesson)

vLLM runs the API server and EngineCore as separate processes; module
globals don't cross. The only correct channel for "is entropy degraded?"
is the cross-process status file (`telemetry/status_file.py`, written by
`FallbackEntropySource` on transitions + throttled count refreshes;
publishing is opt-in via `enable_status_publishing()` and enabled by the
vLLM adapter on the DEFAULT pipeline only, so a preinit `system` wrapper
can't clobber the quantum lane's file). **Do not add a live gRPC probe
or recurring poll to any health path** — a status consumer needs
last-known state, not a fresh measurement; the 2026-07 refactor removed
the HTTP reader middleware, and any future reader should stay passive.

### Per-request entropy-source override (`qr_entropy_source_type`)

Enables comparison mode (quantum vs system) on a single GPU. Contract:

- `entropy_source_type` carries per-request metadata in
  `config/model.py`;
- the vLLM adapter pre-initialises one pipeline per source in
  `QR_PREINIT_ENTROPY_SOURCES` (default `"quantum_grpc,system"`) at
  construction and rejects un-preinit'd values cleanly;
- `apply()` looks up the per-request pipeline by source key. The
  just-in-time invariant is preserved: entropy fetched after logits.

### vLLM reasoning field naming

vLLM 0.17+ with `--reasoning-parser` extracts `<think>...</think>` into
a separate field on chat-completion deltas — but names it
`delta.reasoning`, while the spec name is `delta.reasoning_content`.
Consumers should read both (the qthought client's
`_extract_content_delta` does).

### vLLM CLI flag churn

vLLM removes CLI flags between minor releases (e.g. PR #21739 dropped
`--disable-log-requests` in v0.10/v0.11). If you script `vllm serve`
argv anywhere, validate flags against
`vllm.entrypoints.openai.cli_args:make_arg_parser` at startup rather
than discovering renames in production tracebacks.

### V6 temperature-strategy tranche parity check (2026-07)

Recorded while porting the V6 families `tt_exchange` / `evdt_tt` from
createmp-evalsuite (research spec §7.1/§7.3). Documented check only — no
behavior changes:

- **`hvh_drift` matches V6 §8.3 exactly**: temperature and min-p
  formulas, EMA update `(1−λ)·ema + λ·x`, first-token seeding (Δ = 0,
  no cold-start branch), drift-after-update ordering, and the guardrail
  box `T ∈ [0.3, 2.2]`, `min_p ∈ [0, 0.15]`. Defaults deliberately pin
  the **V6_HVD_R01_01 BO winner**, not the §8.4 *predicted* defaults —
  that divergence is intentional and documented on the config fields.
- **`edt` is NOT the createmp/V5 EDT formula.** qr-sampler's `edt`
  computes `T = base · (H/ln V)^exponent` (power-law of normalised
  entropy); createmp's `EDTProcessor` computes `T = T0 · N^(θ/H)`
  (the original entropy-dependent-temperature paper form). Both are
  entropy-monotone but numerically different. Re-scoping to align them
  would be a behavior change to a shipped strategy — deferred until a
  study actually needs createmp-parity EDT (if so, port it as a NEW
  strategy id, e.g. `edt_v5`, rather than mutating `edt`).
- The remaining V6 families are deferred: DeCoupT-Smooth (LL-15 R00
  gate failures), Mixture-of-Temperatures / Nonlinear remap /
  Ring-buffer (need a probability-transform seam — they reshape the
  distribution, not just `(T, min_p)`). The seam design is recorded in
  the qr-llm-research area docs, not here. (LL-14's `T_hot = 1.45`
  correction applies to the deferred Mixture-of-Temperatures family,
  so it is not exercised by this tranche.)
- `qr_truncate_first` (EVDT-TT's truncate-before-temperature order) is
  the one pinned exception to selector invariant 15 — see AGENTS.md.

---

## History (Modal / Open WebUI / Cipherstone era — surface removed 2026-07)

> The Modal deployment connectors, Open WebUI integration, and the
> Cipherstone-specific deploy glue were deleted in the 2026-07 refactor
> (see `CHANGELOG.md`). These notes are kept for lineage and for anyone
> resurrecting a snapshot/serverless deploy from git history. Path
> references below are to deleted files.

- **Production QRNG (Cipherstone)**: proto package `qrng`, service
  `QuantumRNG`, method `GetRandomBytes` — mapped via
  `QR_GRPC_METHOD_PATH=/qrng.QuantumRNG/GetRandomBytes` (unary; no
  streaming RPC). Transport was a `cloudflared access tcp` sidecar
  binding loopback `127.0.0.1:50051`; the insecure channel was safe
  ONLY because the sidecar was in-container. Quota ceilings from that
  provider survive as the `qrng_max_*` config defaults.
- **Modal snapshot integrity**: no live gRPC channel captured in the
  snapshot (hence lazy channel creation, still live above); no
  process-relative state; secrets mounted after restore, so config had
  to be constructed inside `@modal.enter(snap=True)`, never at import.
- **Modal `@modal.web_server` needs `--host 0.0.0.0`**: inbound traffic
  proxies via the external interface, not loopback; vllm serve binding
  loopback made every external request hang after TLS accept.
- **`add_python` + Dockerfile pip layering trap**: Modal's auto-injected
  Python does not see Dockerfile-side pip installs; deps needed
  `.pip_install(...)` even when already in the image.
- **FP8 + CRIU snapshot ceiling**: bf16 27B (~54 GiB resident) exceeded
  cuda-checkpoint's 180 s enumeration timeout on every cold-cold
  restore; the FP8 build (~27 GiB) made warm restores reliable (~2 s)
  while cold-cold stayed intermittent. Warmup requests were baked into
  the snapshot; `--enforce-eager` proved NOT load-bearing.
- **Post-wake CUDA-graph recapture (iter-55)**: sleep→snapshot→restore
  silently dropped captured CUDA graphs (~14× slower tokens); `_wake`
  re-ran `compile_or_warm_up_model` via `/collective_rpc` and A/B'd
  decode speed per boot.
- **`/health/entropy` cost lesson**: every external GET reset Modal's
  idle-scaledown timer — an always-on OWUI status chip polling the
  endpoint kept an idle H100 warm 24/7, and the endpoint's live 8-byte
  gRPC probe poked the QRNG just to tint a chip. Root of the "passive
  health" rule above.
- **Region pinning**: vLLM classes stayed in `["us-east", "us-west"]`
  because the QRNG endpoint was east-US and every token issued an RPC;
  single-zone pinning starved GPU scheduling.
- **Service tokens (OWUI filter/pipe)**: `X-Service-Token:
  <unix_ts>.<hmac>` with HMAC-SHA256 over `ts + path`; signer used the
  FIRST secret of a comma-separated list, verifier accepted ANY —
  rolling rotation without lockstep redeploys. 60 s window.
- **Secret split**: QRNG vars lived in the `qr-sampler-prod` Modal
  Secret, mounted only on the vLLM classes; the chat-side secret never
  carried them (leak-surface minimisation).
- The iter-by-iter Modal deploy narrative (iter-01..iter-57) lives in
  git history and the sibling qr-llm-chat repo's records.
