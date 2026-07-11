# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — sampling hot-path performance tranche (2026-07)

Standard optimizations only; the sampled distribution, selector order
(AGENTS.md invariant 15), just-in-time entropy contract, and all
diagnostics are unchanged (equivalence is test-pinned in
`tests/test_selection/test_compact_topk.py` and
`tests/test_temperature/test_entropy_fastpath.py`). Measured at a 152k
vocabulary against the PRNG/system entropy lane, the sampler-side ceiling
on concurrent throughput rose from ~39–127 tok/s to ~400–530 tok/s:

- **Shannon entropy via one BLAS dot** (`temperature/base.py`):
  `H = ln Z - dot(exp(s), s) / Z` replaces the full-vocab `log` pass plus
  two boolean fancy-index copies (~42% of the per-token budget). New shared
  `compute_entropy_varentropy` gives the drift strategies (`hvh_drift`,
  `evdt_tt`) the same treatment via a centered second moment;
  `tt_exchange`'s kept-support measurement now works on the exp values
  directly. Degenerate (-inf/NaN) inputs keep their historical outputs.
- **Compacted top-k selection** (`selection/selector.py`): a truncating
  `top_k` now gathers the k surviving logits once (O(vocab) argpartition)
  and runs softmax/min-p/top-p/CDF on the k-element support — the
  historical flow ran every stage (including a full-vocab argsort in
  top-p) over the whole vocabulary. 13x on the `top_k=50, top_p=0.9`
  shape the qthought presets use. The full-vocab path skips the
  temperature divide when `T == 1.0`, counts `probs > 0` once when min-p
  and top-p are disabled, and softmax runs in place on its own scratch
  array.
- **Batched engine-tensor edges** (`engines/vllm/adapter.py`): apply()
  now converts the whole logits batch GPU→CPU with ONE device sync
  (staged through a lazily-grown pinned buffer when vLLM enables
  `is_pin_memory`), and forces all one-hot rows with one `fill_` + one
  `scatter_` instead of a template copy + scalar host→device write per
  row.
- **Parallel per-row sampling** (`engines/vllm/adapter.py` + new
  infrastructure config `apply_parallel_rows` / `QR_APPLY_PARALLEL_ROWS`):
  concurrent requests in a batch no longer serialize behind one another —
  rows are sampled on a worker pool (default cap: CPU count; `1` restores
  the historical serial loop). Thread-safety hardening for the shared hot
  path: per-thread `last_source_used` on `FallbackEntropySource`, a lock
  inside `AdaptiveCircuitBreaker`, locked prefetch counters on
  `QuantumGrpcSource`, a locked RNG in `MockUniformSource`, and a lock on
  the pipeline's gate-status publisher. The `EntropySource` ABC documents
  the concurrency expectations for third-party sources.

### Added — named entropy-source instances (qr-llm-research enabler)

- **`entropy_source_instances` infrastructure config field**
  (`QR_ENTROPY_SOURCE_INSTANCES`, JSON): declare named instances of a
  registered source type with per-instance transport overrides
  (allowlist: `grpc_server_address`, `grpc_api_key`, `grpc_mode`,
  `grpc_timeout_ms`, `grpc_retry_count`). Validated loudly at
  config-construction time (`ConfigValidationError` on unknown override
  keys, names shadowing source types, or unknown/missing `type`). NOT
  per-request overridable.
- **Adapter preinit union**: the vLLM adapter pre-initialises one pipeline
  per declared instance (union with `QR_PREINIT_ENTROPY_SOURCES`); a
  per-request `qr_entropy_source_type` may name an instance, and
  un-preinitialised names keep the existing clean rejection. With no
  declared instances behavior is byte-identical to before (test-pinned).
- **`InstanceNamedSource`** (`entropy/named.py`): rename wrapper applied to
  the primary source, so `TokenSamplingRecord.entropy_source_used`, the
  degraded/recovered log legs, and the status file's `primary_name` all
  carry the instance name end-to-end — PRNG comparison lanes served through
  a `quantum_grpc`-shaped transport are loudly labelled as PRNG.
- No `contract.py` change; `CONTRACT_VERSION` unchanged.

### Removed (2026-07 refactor) — BREAKING, no compatibility kept

- **The entire Modal deployment surface**: `src/qr_sampler/connectors/`
  (Modal app, cloudflared sidecar, vllm_serve patches, health-entropy
  middleware), `deployments/modal/`, and the Modal extra / pytest marker /
  mypy overrides. The status-file *write* side survives in
  `telemetry/status_file.py` + `entropy/fallback.py` so a future reader can
  be reintroduced deliberately.
- **The Open WebUI integration**: `examples/open-webui/` and its tests.
- **`processor.py` and the `QRSamplerLogitsProcessor` alias.** The
  `vllm.logits_processors` and `qr_sampler.engine_adapters` entry points now
  target `qr_sampler.engines.vllm:VLLMAdapter` directly. `import qr_sampler`
  and `import qr_sampler.engines.vllm` are 100% side-effect-free — no
  monkey-patches (the mm-probe patch chain was already a silent no-op), no
  sockets, no file writes (test-pinned).
- **The `contseq` roller and preset** (`contseq.py`, `BUILTIN_PRESETS["contseq"]`,
  its YAML profile and tests). `qthought.py` is the only roller.
- **`deployments/entropic-science/`** — the compose profile existed to
  smoke-test the removed Open WebUI filter/Pipe path against a local vLLM; with
  `examples/open-webui/` gone it was dead. (Recorded divergence: the refactor
  spec said non-Modal `deployments/*` profiles are kept; delete-over-deprecate
  won once its only consumer was removed.)
- **The hand-rolled protobuf codec** in the quantum source. One wire format
  remains: `proto/wire.py` primitives + the pb2 stubs.
- **Root import paths**: `qr_sampler.processor`, `qr_sampler.presets`,
  `qr_sampler.contseq`, `qr_sampler.entropy.quantum`,
  `qr_sampler.entropy.status_file` are gone (see Changed below for the new
  homes). Downstream consumers import through `qr_sampler.contract` only.

### Changed (2026-07 refactor) — BREAKING

- `config.py` + `presets.py` -> `config/` package (`model.py`, `presets.py`,
  `resolve.py`); the config<->presets import cycle is dead. The per-request
  field set is derived from `Field(json_schema_extra={"per_request": True})`
  metadata instead of a hand-maintained frozenset.
- `entropy/quantum.py` (1257 LOC) -> `entropy/qgrpc/` package (`source.py`,
  `transport.py`, `channel.py`, `breaker.py`, `preprobe.py`); prefetch
  ordering, echo verification, and API-key redaction preserved bit-for-bit.
  Cipherstone quota constants became config fields
  (`qrng_max_bytes_per_request`, `qrng_max_requests_per_minute`,
  `qrng_max_bytes_per_day`).
- `engines/vllm.py` -> `engines/vllm/` package (`adapter.py`,
  `telemetry.py`); entry-point strings unchanged.
- `entropy/status_file.py` -> `telemetry/status_file.py` (it is cross-process
  IPC, not an entropy source).
- All four registries (entropy, amplification, temperature, engines) use
  explicit lazy `_BUILTINS` tables resolved on first `get()` instead of
  package-`__init__` import side effects; the builtin table takes precedence
  over entry points.

### Added (2026-07 refactor)

- `qr_sampler.contract` — the cross-repo seam (`CONTRACT_VERSION`, roller +
  provenance types, config + preset names, entropy primitives, exceptions),
  drift-guarded by `tests/test_contract.py` and the consumer-side
  `test_sampler_contract.py` in qr-llm-qthought.
- `QthoughtRoller.draw_u()` / `draw_index(k)` — buffer-free single draws with
  provenance returned directly (replaces the downstream `coin(0.0)` probe
  hack), and the `QthoughtRoller(config, *, entropy_source=...)` injection
  seam.
- `scripts/check.py` — the one-command verification runner (CI and
  pre-commit invoke it too).

### Behavior-change ledger (2026-07 refactor, final state)

Every intended behavior change of the refactor; anything not listed here is
preserved behavior. Items 2 and 3 land in the qr-llm-qthought repo and are
mirrored in its `PRD.md` divergence addendum.

| # | Change | Justification |
|---|---|---|
| 1 | `qr_oe_conditioning` rejected as a per-request override (`ConfigValidationError`) | was a silent no-op (read only at source construction) that falsified `config_hash` provenance |
| 2 | `QTHOUGHT_ENTROPY_DEGRADED/RECOVERED` fire exactly once per transition, from the broker (qthought) | duplicate per-engine state machine deleted |
| 3 | no warmth GET before each completion; `probe_warmth` = `GET /health`, 200 = warm (qthought) | Modal-era wake logic deleted |
| 4 | `import qr_sampler` no longer monkey-patches vLLM | shim + Modal surface removed (mm-probe patch was already a no-op) |
| 5 | `contseq` preset/roller gone | no consumer |
| 6 | proto decode: LAST field-1 occurrence wins; explicitly empty payload raises `EntropyUnavailableError` | pb2/proto3 semantics; byte-identical for every real server (test-asserted) |

### Fixed

- QRNG quota-log throttle silently swallowed the first event on young hosts.
- Duplicate `OPENAI_API_BASE_URL` key in the entropic-science compose profile
  (the whole profile was subsequently removed — see Removed above).

### Changed (iter-57) — stop hammering the QRNG gRPC

- **`/health/entropy` is now PASSIVE**: the middleware reports last-known
  entropy health from the cross-process status file (written by EngineCore
  during real token-sampling) and never opens a gRPC channel on poll. The
  per-poll live 8-byte probe, the lazily-built APIServer `QuantumGrpcSource`,
  the TCP pre-probe, and the `_combine_rpc_ok`/`_live_probe_sync` machinery
  are removed. `rpc_ok = not currently_degraded` when state is known,
  `null`+503 otherwise. `tcp_ok` is always `null`; the `probe` block is gone.
  Rationale: every external `GET` resets Modal's idle-scaledown timer (an
  always-on OWUI chip polling it kept an idle H100 warm), and the live probe
  poked the QRNG just to tint a status chip. The one-time gRPC verification
  stays in `QuantumGrpcSource.warmup()` at container start.
- **Circuit-breaker recovery window backs off exponentially**: new
  `cb_recovery_window_max_s` (default 60 s, env `QR_CB_RECOVERY_WINDOW_MAX_S`).
  `QR_CB_RECOVERY_WINDOW_S=3` is now the BASE; consecutive opens without an
  intervening success double the wait (`base × 2^opens`, capped at the max),
  reset on first success. A sustained QRNG outage settles at ~1 half-open
  reconnect/min instead of a channel-reset storm every 3 s.
- **No per-token gRPC retries**: `QR_GRPC_RETRY_COUNT` default set to `0` for
  both vLLM classes. When the QRNG is down each retry is another connect
  against a dead server; the circuit breaker + system-PRNG fallback are the
  correct resilience layer.
- **Throttled degraded logging**: the structured `entropy.degraded` WARNING is
  now rate-limited together with the `entropy.degraded.alert` ERROR (first
  fallback of a window + at most once/min), instead of one WARNING per
  generated token. The running `fallback_count` rides every record and the
  status file carries the exact live count, so nothing is lost.

### Fixed (iter-55)

- **Post-wake CUDA-graph recapture**: the Modal sleep → CRIU snapshot → restore →
  wake cycle silently dropped the engine's captured CUDA graphs, leaving every
  serving container ~14x slower than its pre-sleep self (~30 ms/token warmup vs
  ~360-430 ms/token post-wake, iter-54 boot-log evidence). `_wake` now measures
  decode speed, re-runs the worker's `compile_or_warm_up_model` via the dev
  `/collective_rpc` endpoint, and measures again — events `vllm.wake.perf` /
  `vllm.wake.recapture_ok` carry the per-boot A/B. Soft-fail; kill switch
  `QR_WAKE_RECAPTURE=0`

### Changed (iter-55)

- `TokenSelector._cdf_select` fast path: O(n) `argpartition` of the top-512
  head + O(K log K) head sort replaces the unconditional full `argsort` over
  the entire vocabulary (~152k) per token; escalates to the exact full-sort
  path whenever `u` is not strictly covered by the head's nonzero cumulative
  mass, so selections are identical (equivalence-tested across distribution
  shapes and u-draws)
- `_stable_softmax` avoids the full-vocab boolean-mask copy on the hot path
  (max over all logits equals max over finite logits whenever any finite
  value exists)

### Added (iter-55)

- Per-stage sampling telemetry: `temperature_ms` / `amplify_ms` / `select_ms`
  on `TokenSamplingRecord`; rolling-window aggregator in the vLLM adapter
  publishing stage means/p95 + prefetch hit / echo-verified ratios through a
  perf status file (surfaced as `/health/entropy`'s `"perf"` block) and a
  rate-limited `qr.sampling.stats` WARNING log line (INFO from the qr_sampler
  logger is invisible in the production EngineCore log stream)

### Added

- **Pipelined commit-then-fetch entropy** (iter-54): the gRPC request for token *N+1*
  is fired the instant token *N* is selected, so the network round trip overlaps the
  engine's next forward pass instead of serializing behind it (the first token's fetch
  fires at request-add time and overlaps the entire prefill). Enabled by default;
  `QR_ENTROPY_PREFETCH=0` (or per-request `qr_entropy_prefetch: false`) restores the
  strictly-serial fetch-after-logits timing. Affects timing only — the sampled
  distribution and the post-selection causal contract are unchanged
  - `EntropySource.prefetch(n, nonce)` / `get_random_bytes_with_ticket(n, ticket)`
    optional hooks (safe no-op defaults; `SystemEntropySource` and the PRNG comparison
    lane are untouched)
  - Verifiable post-selection ordering with **zero server changes**: each pipelined
    request carries a 63-bit commitment nonce — `SHA-256(salt ‖ step ‖ prev_token_id)`
    truncated — in the proto's existing `sequence_id` field; a server that echoes
    `sequence_id` binds its entropy response to a request that could only exist after
    the previous token's selection (`derive_commit_nonce` in `core/pipeline.py`)
  - Per-token verification diagnostics on `TokenSamplingRecord`:
    `entropy_prefetch_hit`, `entropy_nonce`, `entropy_echo_verified`,
    `entropy_server_timestamp_ns`; prefetch hit/miss counters in
    `QuantumGrpcSource.health_check()`
- `_BidiSession`: the bidi-streaming transport now uses a dedicated reader task with
  `sequence_id`-echo correlation (FIFO fallback for non-echoing servers), making
  concurrent in-flight fetches safe — the previous write-then-read pattern interleaved
  incorrectly with more than one fetch outstanding

### Changed

- TCP pre-probe is suppressed while fetches are succeeding (a successful fetch within
  the last 30 s is strictly stronger reachability evidence than a fresh `connect()`),
  removing one connect/close syscall pair from every steady-state token; the probe
  re-engages automatically when no fetch has succeeded recently
- `SamplingPipeline.sample_token()` accepts `build_onehot=False`; the vLLM adapter
  passes it, eliminating a dead vocab-size numpy allocation + fill (~600 KB at 150k
  vocab) per token (the adapter always forced the one-hot directly on the engine tensor)
- gRPC fetch latency samples are now measured inside the transport coroutine (true
  network + server time) rather than around the blocking wait

### Fixed

- `EDTTemperatureStrategy`: `diagnostics["pre_clamp_temp"]` now correctly reports the
  power-law result *before* clamping rather than re-computing the same expression after
  the clamp was already applied (was a silent diagnostic bug when `edt_min_temp` or
  `edt_max_temp` was active)

### Removed

- Dead function `_encode_svarint()` from `entropy_service_pb2.py` — ZigZag encoding is
  only needed for `sint32`/`sint64` proto field types, none of which appear in the
  entropy service proto

### Changed

- `import inspect` in `processor.py` promoted from inside `_accepts_config()` to the
  module-level imports block

### Added

- vLLM V1 LogitsProcessor plugin (`QRSamplerLogitsProcessor`) with batch-level processing
- Pydantic-settings configuration system with `QR_` env prefix and per-request overrides
- Entropy source subsystem with ABC, auto-discovery registry, and entry-point support
  - `QuantumGrpcSource`: gRPC client with unary, server-streaming, and bidirectional transport modes
  - `SystemEntropySource`: `os.urandom()` wrapper
  - `TimingNoiseSource`: CPU timing jitter entropy (experimental)
  - `MockUniformSource`: configurable test source with seed and bias control
  - `FallbackEntropySource`: automatic failover wrapper
- Adaptive circuit breaker for gRPC source (rolling P99, half-open recovery)
- Z-score mean signal amplifier (`zscore_mean`) for bias-preserving entropy-to-uniform mapping
- Temperature strategies: fixed and entropy-dependent (EDT) with Shannon entropy computation
- CDF-based token selector with top-k, top-p (nucleus) filtering
- Diagnostic logging subsystem with three verbosity levels and in-memory record storage
- gRPC proto definition and hand-written stubs for `EntropyService`
- Reference entropy servers: `simple_urandom_server.py`, `timing_noise_server.py`, `qrng_template_server.py`
- Docker and docker-compose deployment templates
- systemd service unit and environment file
- Apache 2.0 license
- Pre-commit configuration with ruff, mypy, bandit, and standard hooks
- Comprehensive test suite with statistical validation tests
