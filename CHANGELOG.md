# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
