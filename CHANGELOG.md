# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
