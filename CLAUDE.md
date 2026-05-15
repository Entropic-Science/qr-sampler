# CLAUDE.md -- Codebase Guide for Coding Agents

## What this project is

`qr-sampler` is an engine-agnostic framework that replaces standard LLM token sampling with external-entropy-driven selection. It fetches random bytes from any entropy source (QRNGs via gRPC, OS randomness, CPU timing jitter, OpenEntropy), amplifies the signal into a uniform float via z-score or ECDF statistics, and uses that float to select a token from a probability-ordered CDF.

The core sampling pipeline (`qr_sampler.core`) has **zero inference-engine dependencies** -- it operates on numpy arrays and knows nothing about torch, vLLM, or any specific engine. Engine-specific integration is handled by thin adapter classes (`qr_sampler.engines`). A vLLM V1 adapter ships out of the box; other engines (e.g., vLLM-Metal for Apple Silicon) are supported via the `EngineAdapter` plugin system and declarative YAML profiles.

The primary use case is consciousness-research: studying whether conscious intent can influence quantum-random processes in LLM token selection.

## Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v
pytest tests/test_processor.py -v
pytest tests/test_statistical_properties.py -v
pytest tests/test_core/ -v
pytest tests/test_engines/ -v
pytest tests/test_profiles/ -v
pytest tests/test_cli/ -v

# Run with coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing

# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with CLI (click + jinja2)
pip install -e ".[cli]"

# Lint and format
ruff check src/ tests/
ruff format --check src/ tests/

# Type check
mypy --strict src/

# CLI commands (requires [cli] extra)
qr-sampler list engines              # List available engine profiles
qr-sampler list models --engine vllm # List known-working models for an engine
qr-sampler list entropy-sources      # List entropy source profiles
qr-sampler list amplifiers           # List amplifier profiles
qr-sampler list samplers             # List adaptive sampler profiles
qr-sampler info engine vllm          # Detailed info for a component
qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct  # Check compatibility
qr-sampler build --engine vllm --entropy quantum_grpc --output ./deploy # Generate Docker Compose
```

## File map

```
src/qr_sampler/
+-- __init__.py                    # Package version (setuptools-scm), re-exports
+-- __main__.py                    # CLI entry: `python -m qr_sampler` -> cli/main.py
+-- config.py                      # QRSamplerConfig (pydantic BaseSettings), resolve_config(), validate_extra_args()
+-- exceptions.py                  # QRSamplerError -> {EntropyUnavailableError, ConfigValidationError, SignalAmplificationError, TokenSelectionError}
+-- processor.py                   # Re-export: VLLMAdapter as QRSamplerLogitsProcessor (backward compat)
+-- py.typed                       # PEP 561 marker
+-- presets.py                     # BUILTIN_PRESETS, resolve_preset(), expand_extra_args() -- preset resolution layer over extra_args
+-- core/                          # Engine-agnostic sampling pipeline (NO torch/vLLM imports)
|   +-- __init__.py                # Re-exports SamplingPipeline, SamplingResult, build_pipeline, etc.
|   +-- pipeline.py                # SamplingPipeline class + factory functions (build_pipeline, build_entropy_source, config_hash, accepts_config)
|   +-- types.py                   # SamplingResult frozen dataclass (token_id, one_hot, record)
+-- engines/                       # Engine adapter layer
|   +-- __init__.py                # Re-exports EngineAdapter, EngineAdapterRegistry
|   +-- base.py                    # EngineAdapter ABC (get_pipeline, close)
|   +-- registry.py                # EngineAdapterRegistry (decorator + qr_sampler.engine_adapters entry-point discovery)
|   +-- vllm.py                    # VLLMAdapter: vLLM V1 LogitsProcessor, delegates sampling to SamplingPipeline
+-- profiles/                      # Declarative YAML profile system (read-only metadata for CLI)
|   +-- __init__.py                # Re-exports ProfileLoader, CompatibilityChecker, CompatibilityReport
|   +-- schema.py                  # Pydantic models: EngineProfile, EntropySourceProfile, AmplifierProfile, SamplerProfile, etc.
|   +-- loader.py                  # ProfileLoader: discovers built-in + user override profiles, lazy loading with cache
|   +-- compatibility.py           # CompatibilityChecker: tri-state logic (known_working/untested/known_incompatible)
|   +-- engines/
|   |   +-- vllm.yaml              # vLLM NVIDIA GPU profile
|   |   +-- vllm_metal.yaml        # vLLM Apple Silicon / Metal profile
|   +-- entropy/
|   |   +-- system.yaml            # os.urandom() entropy
|   |   +-- quantum_grpc.yaml      # gRPC quantum entropy
|   |   +-- timing_noise.yaml      # CPU timing jitter
|   |   +-- mock_uniform.yaml      # Test mock source
|   |   +-- openentropy.yaml       # OpenEntropy service
|   +-- amplifiers/
|   |   +-- zscore_mean.yaml       # Z-score mean amplifier
|   |   +-- ecdf.yaml              # ECDF amplifier
|   +-- samplers/
|   |   +-- fixed.yaml             # Fixed temperature sampler
|   |   +-- edt.yaml               # Entropy-dependent temperature sampler
|   +-- presets/
|       +-- creative_sampling.yaml # V6_HVD_R01_01 hvh_drift preset (experimental)
|       +-- normal_t1.yaml         # Baseline fixed T=1 preset
+-- cli/                           # Command-line interface (requires [cli] extra: click + jinja2)
|   +-- __init__.py
|   +-- main.py                    # Click group with version_option
|   +-- validate_cmd.py            # `qr-sampler validate` -- check stack compatibility, exit 0/1/2
|   +-- build_cmd.py               # `qr-sampler build` -- generate Docker Compose from templates
|   +-- list_cmd.py                # `qr-sampler list` -- list engines/models/entropy-sources/amplifiers/samplers
|   +-- info_cmd.py                # `qr-sampler info` -- detailed component info
+-- templates/                     # Jinja2 templates for `qr-sampler build`
|   +-- docker-compose.yml.j2      # Docker Compose template (conditional entropy server, engine-specific)
|   +-- env.j2                     # .env file template
+-- amplification/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # SignalAmplifier ABC, AmplificationResult frozen dataclass
|   +-- registry.py                # AmplifierRegistry (decorator + build pattern)
|   +-- zscore.py                  # ZScoreMeanAmplifier (z-score -> normal CDF -> uniform)
|   +-- ecdf.py                    # ECDFAmplifier (empirical CDF amplification)
+-- entropy/
|   +-- __init__.py                # Re-exports
|   +-- base.py                    # EntropySource ABC (name, is_available, get_random_bytes, get_random_float64, close, health_check)
|   +-- registry.py                # EntropySourceRegistry with entry-point auto-discovery from qr_sampler.entropy_sources
|   +-- quantum.py                 # QuantumGrpcSource: 3 modes (unary, server_streaming, bidi_streaming), circuit breaker, grpc.aio
|   +-- system.py                  # SystemEntropySource: os.urandom()
|   +-- timing.py                  # TimingNoiseSource: CPU timing jitter (experimental)
|   +-- mock.py                    # MockUniformSource: configurable seed/bias for testing
|   +-- fallback.py                # FallbackEntropySource: composition wrapper, catches only EntropyUnavailableError
|   +-- openentropy.py             # OpenEntropySource: OpenEntropy service integration
+-- logging/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # TokenSamplingRecord frozen dataclass (16 fields, __slots__)
|   +-- logger.py                  # SamplingLogger: none/summary/full log levels, diagnostic_mode
+-- proto/
|   +-- __init__.py
|   +-- entropy_service.proto      # gRPC proto: GetEntropy (unary) + StreamEntropy (bidi)
|   +-- entropy_service_pb2.py     # Hand-written protobuf message stubs
|   +-- entropy_service_pb2_grpc.py # Hand-written gRPC client + server stubs
+-- selection/
|   +-- __init__.py                # Re-exports
|   +-- types.py                   # SelectionResult frozen dataclass
|   +-- selector.py                # TokenSelector: top-k -> softmax -> top-p -> CDF -> searchsorted
+-- temperature/
    +-- __init__.py                # Re-exports
    +-- base.py                    # TemperatureStrategy ABC, TemperatureResult, compute_shannon_entropy()
    +-- registry.py                # TemperatureStrategyRegistry (passes vocab_size if constructor accepts it)
    +-- fixed.py                   # FixedTemperatureStrategy: constant temperature
    +-- edt.py                     # EDTTemperatureStrategy: entropy-dependent, H_norm^exp scaling
    +-- hvh_drift.py               # HVHDriftStrategy: stateful per-request EMA-driven temperature + dynamic min-p (V6 creative_sampling)

tests/
+-- __init__.py
+-- conftest.py                    # Shared fixtures: default_config, sample_logits
+-- test_config.py                 # Config defaults, env vars, per-request resolution, validation
+-- test_processor.py              # Backward compat: QRSamplerLogitsProcessor via re-export
+-- test_statistical_properties.py # KS-test uniformity, bias detection, EDT monotonicity (requires scipy)
+-- test_wire_format.py            # Proto wire format tests
+-- test_core/
|   +-- test_types.py              # SamplingResult immutability, fields
|   +-- test_pipeline.py           # Pipeline with MockUniformSource + numpy: single token, overrides, factory
+-- test_engines/
|   +-- test_vllm_adapter.py       # VLLMAdapter delegates to pipeline, batch processing, one-hot forcing
|   +-- test_registry.py           # EngineAdapterRegistry decorator, entry-point discovery, list_available
|   +-- test_hvh_drift_lifecycle.py # Per-request state isolation for HVHDriftStrategy via _RequestState
+-- test_presets/
|   +-- test_resolution.py         # resolve_preset, expand_extra_args, per-request precedence, env-var fallback
|   +-- test_yaml_sync.py          # Drift guard: YAML overrides must match BUILTIN_PRESETS key-for-key
+-- test_profiles/
|   +-- test_schema.py             # Pydantic validation of profile models, frozen enforcement
|   +-- test_loader.py             # Load built-in profiles, user overrides, missing profile error
|   +-- test_compatibility.py      # Tri-state logic, full stack check, dependency detection
+-- test_cli/
|   +-- test_validate.py           # Click CliRunner: exit codes 0/1/2 for known-working/untested/incompatible
|   +-- test_build.py              # Compose generation, --dry-run, --output, --force
|   +-- test_list.py               # All list subcommands return expected profile data
|   +-- test_list_presets.py       # `qr-sampler list presets` shows both presets, experimental flag
|   +-- test_info.py               # Detailed info output for each component type
|   +-- test_info_preset.py        # `qr-sampler info preset <id>` outputs override map, V6 values
+-- test_amplification/
|   +-- test_zscore.py             # Known values, SEM derivation, edge cases, frozen immutability
|   +-- test_ecdf.py               # ECDF amplifier tests
+-- test_entropy/
|   +-- test_system.py             # Correct byte count, always available
|   +-- test_timing.py             # Correct byte count, non-zero output
|   +-- test_mock.py               # Seeded reproducibility, bias simulation
|   +-- test_fallback.py           # Primary delegation, fallback trigger, error propagation
|   +-- test_registry.py           # Decorator registration, entry-point discovery, lazy loading
|   +-- test_quantum.py            # Mocked gRPC for 3 modes, circuit breaker, error mapping
|   +-- test_openentropy.py        # OpenEntropy source tests
+-- test_logging/
|   +-- test_logger.py             # Record immutability, log levels, diagnostic mode, summary stats
+-- test_selection/
|   +-- test_selector.py           # CDF known values, top-k/top-p, edge cases
|   +-- test_min_p.py              # min_p mask: no-op at 0.0, threshold semantics, ordering vs top_p
+-- test_temperature/
    +-- test_fixed.py              # Constant output, Shannon entropy computation
    +-- test_edt.py                # Monotonicity, clamping, exponent effects
    +-- test_hvh_drift.py          # HVHDriftStrategy: EMA init, drift formula, clamps, V6 reference values

examples/
+-- servers/
|   +-- simple_urandom_server.py   # Minimal reference server (~50 lines of logic)
|   +-- timing_noise_server.py     # CPU timing jitter entropy server
|   +-- qrng_template_server.py    # Annotated template with 3 TODO sections
+-- docker/
|   +-- Dockerfile.entropy-server  # Slim Python image for any example server
|   +-- Dockerfile.vllm            # vLLM inference server image
+-- systemd/
|   +-- qr-entropy-server.service  # systemd unit with restart-on-failure
|   +-- qr-entropy-server.env      # Environment variables
+-- open-webui/
    +-- qr_sampler_filter.py       # Open WebUI filter plugin
    +-- qr_sampler_filter.json     # Filter metadata
    +-- README.md                  # Open WebUI integration guide
```

## Architecture invariants -- DO NOT break these

1. **No hardcoded values.** Every numeric constant traces to a named field in `QRSamplerConfig` (pydantic-settings `BaseSettings` in `config.py`). Mathematical constants like `sqrt(2)` and `0.5 * (1 + erf(...))` are acceptable -- they are math, not configuration.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, `TemperatureStrategy`, or `EngineAdapter` implementations are registered via class method decorators (`@AmplifierRegistry.register("name")`, `@TemperatureStrategyRegistry.register("name")`, `@register_entropy_source("name")`, `@EngineAdapterRegistry.register("name")`). Code never instantiates strategies directly -- it goes through registry `.build()` or `.get()` methods. No if/else chains for strategy selection.

3. **ABCs define contracts.** `EntropySource` (in `entropy/base.py`), `SignalAmplifier` (in `amplification/base.py`), `TemperatureStrategy` (in `temperature/base.py`), and `EngineAdapter` (in `engines/base.py`) are ABCs. All concrete implementations must subclass them. The core pipeline and adapters only reference abstract types.

4. **FallbackEntropySource is a composition wrapper**, not a subclass of a specific source. It takes any `EntropySource` as primary and any as fallback. It only catches `EntropyUnavailableError` -- all other exceptions propagate.

5. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(N)`. It is computed at amplification time from config fields. There is no `sem` config field.

6. **Frozen dataclasses for all result types.** `AmplificationResult`, `TemperatureResult`, `SelectionResult`, `TokenSamplingRecord`, and `SamplingResult` use `@dataclass(frozen=True, slots=True)`. `QRSamplerConfig` is immutable via pydantic BaseSettings. Do not make these mutable.

7. **Per-request config resolution.** `resolve_config(defaults, extra_args)` creates a new config instance via `model_validate()` on a merged dict. It never mutates the default config. Infrastructure fields (`grpc_server_address`, `grpc_timeout_ms`, `grpc_retry_count`, `grpc_mode`, `fallback_mode`, `entropy_source_type`) are NOT overridable per-request. This is enforced by `_PER_REQUEST_FIELDS` frozenset in `config.py`.

8. **Engine adapters force one-hot logits.** After selecting a token, the adapter sets the entire logit row to `-inf` except the selected token (set to `0.0`). This forces the engine's downstream sampler to pick exactly that token. The one-hot array is generated by the core pipeline as numpy; the adapter converts it to the engine's native tensor format.

9. **Logging uses `logging.getLogger("qr_sampler")`.** No `print()` statements anywhere in production code. All per-token logging goes through `SamplingLogger`.

10. **Just-in-time entropy.** Physical entropy generation occurs ONLY when `get_random_bytes()` is called -- after logits are available. No pre-buffering, no caching. The gRPC request is sent only when the pipeline needs bytes for a specific token.

11. **Entry-point auto-discovery for entropy sources and engine adapters.** Third-party packages register sources via the `qr_sampler.entropy_sources` entry-point group and engine adapters via the `qr_sampler.engine_adapters` group. Registries load entry points lazily on first `get()` call. Built-in decorator registrations take precedence over entry points.

12. **Circuit breaker protects gRPC source.** `QuantumGrpcSource` tracks rolling P99 latency (deque, 100 samples), computes adaptive timeout = `max(5ms, P99 * 1.5)`, opens after 3 consecutive failures, enters half-open state after 10s.

13. **Engine adapters are thin wrappers.** Must not contain sampling logic. Convert logits to numpy, call `SamplingPipeline.sample_token()`, convert result back to engine-native tensors. All sampling math lives in the core pipeline (`core/pipeline.py`).

14. **Profiles are declarative data, not code.** Compatibility matrices, model lists, and defaults are expressed in YAML files under `profiles/`. Code reads and validates profiles; it does not encode this information. Profile data is for CLI validation and documentation only.

15. **The core pipeline has zero engine dependencies.** `qr_sampler.core` is importable and functional without vLLM, torch, MLX, or any engine package. Only numpy and runtime deps (pydantic, grpcio, etc.) are used.

16. **Profile loading never affects runtime sampling.** Profiles exist for CLI validation and documentation. The existing registry/config system drives all runtime decisions. Missing or corrupt profile YAML never causes sampling failure.

17. **Stateful temperature strategies are per-request, not per-process.** The `hvh_drift` strategy carries EMA state (`H_ema`, `VH_ema`) on the strategy instance. Each request gets its own strategy instance via `_RequestState.strategy` (managed by the engine adapter). Strategy instances must never be shared across requests — that would leak distributional drift between concurrent users. Built-in stateless strategies (`fixed`, `edt`) are safe to share but are constructed fresh per request anyway for uniformity.

18. **Selector pipeline order is `top-k → softmax → min-p → top-p → CDF`.** The `min_p` step (added for `hvh_drift`) sits between softmax and top-p so the threshold is applied to normalized probabilities relative to the top probability mass. `min_p=0.0` (the default) is a strict no-op and preserves prior behavior byte-for-byte. The threshold semantic is "keep tokens with `prob >= min_p * max_prob`"; if everything would be masked, argmax is reserved.

19. **Presets are a thin resolution layer over `extra_args`, expanded inside `resolve_config` before `validate_extra_args`.** `BUILTIN_PRESETS` in `presets.py` is the runtime source of truth; YAML files under `profiles/presets/` are documentation only. A drift-guard test (`tests/test_presets/test_yaml_sync.py`) keeps the two in sync. `expand_extra_args` consumes a `qr_preset` key (per-request) or the `QR_PRESET` env var (process-wide default) and produces the equivalent `qr_*` extra_args dict, with caller-supplied keys winning over preset values. Engine adapters do not need to know about presets — resolution happens engine-agnostically.

## Coding conventions

- **Python 3.10+** -- use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** -- Google style on every public class and method
- **Imports** -- standard library first, third-party second, local third. No wildcard imports.
- **Line length** -- 100 characters (configured in `pyproject.toml` ruff section)
- **Errors** -- custom exception hierarchy rooted in `QRSamplerError` (in `exceptions.py`). Never catch bare `Exception` (health checks are the sole documented exception with `# noqa` comments).
- **No global mutable state** outside processor/adapter instances. Registries are populated at module import time and are effectively read-only after that.
- **No `print()`** -- use `logging` module with the `"qr_sampler"` logger
- **`QR_` prefix** for environment variables, `qr_` prefix for extra_args keys
- **Pydantic-settings BaseSettings** for configuration (not raw dataclasses)
- **Pydantic models with `ConfigDict(frozen=True)`** for profile schemas (not raw dataclasses)

## Key data flows

### Per-token sampling pipeline (in `core/pipeline.py` `sample_token()`)

```
Engine adapter calls pipeline.sample_token(logits_1d: np.ndarray, config?, ...)
  |
  +-> TemperatureStrategy.compute_temperature(logits, config)
  |     -> TemperatureResult { temperature, shannon_entropy, diagnostics }
  |
  +-> EntropySource.get_random_bytes(config.sample_count)
  |     -> raw bytes (20,480 by default) -- JUST-IN-TIME
  |
  +-> SignalAmplifier.amplify(raw_bytes)
  |     -> AmplificationResult { u, diagnostics }
  |
  +-> TokenSelector.select(logits, temperature, top_k, top_p, u)
  |     -> SelectionResult { token_id, token_rank, token_prob, num_candidates }
  |
  +-> Generate numpy one-hot: array of -inf with 0.0 at token_id
  |
  +-> SamplingLogger.log_token(TokenSamplingRecord)
  |
  +-> return SamplingResult { token_id, one_hot, record }
```

### Engine adapter flow (e.g., VLLMAdapter in `engines/vllm.py`)

```
vLLM calls VLLMAdapter.apply(logits: torch.Tensor)
  -> for each batch row:
       -> convert torch row to numpy (zero-copy if CPU)
       -> pipeline.sample_token(numpy_row, per_request_config, ...)
            -> returns SamplingResult { token_id, one_hot, record }
       -> write numpy one_hot back into torch logit tensor
```

### Config resolution flow

```
Environment variables (QR_*)
  -> QRSamplerConfig() -> pydantic-settings auto-loads from env + .env file

Per-request extra_args (qr_*)
  -> resolve_config(defaults, extra_args) -> new QRSamplerConfig instance
```

### Component construction flow (via `core/pipeline.py` factory functions)

```
build_pipeline(config, vocab_size) -> SamplingPipeline
  -> build_entropy_source(config)
      -> EntropySourceRegistry.get(config.entropy_source_type)
      -> source class, instantiated with config if constructor accepts it
      -> wrapped in FallbackEntropySource if fallback_mode != "error"
  -> AmplifierRegistry.build(config)
      -> ZScoreMeanAmplifier(config) or ECDFAmplifier(config) (from registry by signal_amplifier_type)
      -> calibrate amplifier if it supports calibration
  -> TemperatureStrategyRegistry.build(config, vocab_size)
      -> FixedTemperatureStrategy() or EDTTemperatureStrategy(vocab_size) (from registry)
  -> TokenSelector()
  -> SamplingLogger(config)
  -> return SamplingPipeline(entropy_source, amplifier, strategy, selector, logger, config)
```

### CLI validation flow (in `cli/validate_cmd.py`)

```
qr-sampler validate --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct
  -> ProfileLoader loads YAML profiles
  -> CompatibilityChecker.check_stack(engine, model, entropy_source, amplifier, sampler)
      -> check_engine_model(engine_id, model_id) -> CompatibilityStatus
      -> check_entropy_amplifier(source_id, amplifier_id) -> CompatibilityStatus
      -> check_dependencies(engine_id) -> list of missing deps
  -> CompatibilityReport { checks, warnings, errors, available_samplers, missing_deps }
  -> exit code: 0 (all known-working), 1 (untested pairs), 2 (incompatible/missing)
```

### gRPC transport modes (in `entropy/quantum.py`)

```
Unary (grpc_mode="unary"):
  Client --EntropyRequest--> Server --EntropyResponse--> Client
  (one HTTP/2 stream per call, ~1-2ms overhead)

Server streaming (grpc_mode="server_streaming"):
  Client --EntropyRequest--> Server
  Server --EntropyResponse--> Client
  (short-lived stream, ~0.5-1ms)

Bidirectional (grpc_mode="bidi_streaming"):
  Client <--persistent stream--> Server
  (stream stays open for entire session, ~50-100us same-machine)
```

All modes use `grpc.aio` (asyncio) on a background thread with sync wrappers via `run_coroutine_threadsafe()`.

## How to add new components

### New engine adapter

1. Create a class in `src/qr_sampler/engines/` subclassing `EngineAdapter`
2. Implement `get_pipeline(self) -> SamplingPipeline` (required)
3. Override `close(self)` if cleanup beyond `pipeline.close()` is needed
4. Implement engine-specific hook methods (e.g., `apply()` for vLLM) -- these are NOT on the ABC
5. Use `build_pipeline(config, vocab_size)` from `core/pipeline.py` to construct the pipeline
6. Register: `@EngineAdapterRegistry.register("my_engine")`
7. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.engine_adapters"]`
8. Create a YAML profile in `src/qr_sampler/profiles/engines/my_engine.yaml`
9. Add tests in `tests/test_engines/`

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(self, raw_bytes: bytes) -> AmplificationResult`
3. Constructor takes `config: QRSamplerConfig` as first arg
4. Register: `@AmplifierRegistry.register("my_name")`
5. Use via config: `signal_amplifier_type = "my_name"` or `extra_args={"qr_signal_amplifier_type": "my_name"}`
6. Create a YAML profile in `src/qr_sampler/profiles/amplifiers/my_name.yaml`
7. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(self, logits, config) -> TemperatureResult`
3. Always compute and return `shannon_entropy` even if not used in formula (logging depends on it)
4. Register: `@TemperatureStrategyRegistry.register("my_name")`
5. If the constructor needs `vocab_size`, accept it as first positional arg -- the registry detects this via try/except
6. Create a YAML profile in `src/qr_sampler/profiles/samplers/my_name.yaml`
7. Add tests in `tests/test_temperature/`

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement: `name` (property), `is_available` (property), `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` if the source cannot provide bytes
4. Register: `@register_entropy_source("my_name")` (from `entropy.registry`)
5. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]`
6. Create a YAML profile in `src/qr_sampler/profiles/entropy/my_name.yaml`
7. Add tests in `tests/test_entropy/`

### New config field

1. Add the field to `QRSamplerConfig` in `config.py` with `Field(default=..., description=...)`
2. If per-request overridable, add the field name to `_PER_REQUEST_FIELDS` frozenset
3. The env var `QR_{FIELD_NAME_UPPER}` is automatically supported by pydantic-settings
4. The extra_args key `qr_{field_name}` is automatically supported by `resolve_config()`
5. Add tests in `tests/test_config.py`

### New profile

1. Create a YAML file in the appropriate `src/qr_sampler/profiles/<category>/` directory
2. Follow the schema defined by the corresponding Pydantic model in `profiles/schema.py`
3. Include `id`, `name`, `description`, and all relevant compatibility cross-references
4. Add tests in `tests/test_profiles/` to validate the new profile loads correctly
5. User overrides go in `$QR_PROFILES_DIR` or `~/.qr-sampler/profiles/` (same directory structure)

### New preset

A preset is a named bundle of `qr_*` overrides that callers can opt into via `qr_preset` (per-request) or `QR_PRESET` (env var). The runtime entry lives in `presets.py`; the YAML entry is documentation only.

1. Add a new entry to `BUILTIN_PRESETS` in `src/qr_sampler/presets.py` -- a dict of `qr_*` keys to values (omit the `qr_` prefix for the inner keys; `resolve_preset` adds it)
2. Create the matching YAML file `src/qr_sampler/profiles/presets/<id>.yaml` with fields `id`, `name`, `description`, `experimental`, `origin`, and an `overrides:` map whose keys/values exactly match the `BUILTIN_PRESETS` entry
3. The drift-guard test in `tests/test_presets/test_yaml_sync.py` iterates over `BUILTIN_PRESETS` automatically -- no test change needed if the YAML matches
4. Add CLI integration tests in `tests/test_cli/test_list_presets.py` and `tests/test_cli/test_info_preset.py` if the preset has behaviour worth surfacing (e.g., experimental flag, origin link)
5. If the preset is experimental, mention "experimental" prominently in the YAML `description` so it appears in `qr-sampler info preset <id>` output

## Testing approach

- **No real QRNG server or GPU needed.** Tests use `MockUniformSource` and numpy arrays. The core pipeline tests (`test_core/`) run without torch or any engine dependency.
- **Dependency injection everywhere.** The VLLMAdapter accepts `vllm_config=None` for testing. The SamplingPipeline accepts all components via constructor.
- **Core pipeline tested independently.** `test_core/test_pipeline.py` validates the full sampling loop using only numpy and mock sources -- proves the pipeline works without any engine.
- **Engine adapter tests** in `test_engines/` verify that adapters correctly delegate to the pipeline and handle engine-specific concerns (batch state, tensor conversion).
- **Profile tests** in `test_profiles/` are data-driven: load each YAML file, validate against Pydantic schema, verify cross-references, test compatibility checker tri-state logic.
- **CLI tests** in `test_cli/` use Click's `CliRunner` to capture output and exit codes without subprocesses.
- **Statistical tests** in `test_statistical_properties.py` require `scipy` (dev dependency). They validate mathematical properties: KS-test for u-value uniformity, bias detection, EDT monotonicity.
- **Frozen dataclass tests** verify immutability of all result types (including `SamplingResult`).
- **Edge case coverage** is thorough: empty inputs, single-token vocab, all-identical logits, all-inf-except-one logits, zero temperature.
- **gRPC tests** mock the gRPC channel/stub to test all 3 transport modes and the circuit breaker without a real server.
- **Backward compatibility** is verified: `test_processor.py` imports `QRSamplerLogitsProcessor` via the re-export in `processor.py` and confirms it resolves to `VLLMAdapter`.

## Proto stubs

The files in `src/qr_sampler/proto/` are hand-written minimal stubs (not generated by `protoc`). They define just enough for the gRPC client and example servers to work:

- `entropy_service.proto` -- the canonical protocol definition
- `entropy_service_pb2.py` -- `EntropyRequest` and `EntropyResponse` message classes
- `entropy_service_pb2_grpc.py` -- `EntropyServiceStub` (client) and `EntropyServiceServicer` (server base) + `add_EntropyServiceServicer_to_server()`

If the proto definition changes, these stubs must be updated manually or regenerated with `grpc_tools.protoc`.

## Dependencies

- **Runtime:** `numpy>=2.0.0`, `pydantic>=2.5.0`, `pydantic-settings>=2.5.0`, `grpcio>=1.68.0`, `protobuf>=5.26.0`, `pyyaml>=6.0`
- **CLI extra (`[cli]`):** `click>=8.0`, `jinja2>=3.1`
- **Dev:** `pytest>=8.0`, `pytest-cov>=5.0`, `scipy>=1.12.0`, `ruff>=0.8.0`, `mypy>=1.10.0`, `pre-commit>=4.0`, `bandit>=1.8.0`, `types-PyYAML>=6.0`, `click>=8.0`, `jinja2>=3.1`
- **Optional extras:** `openentropy>=0.10.0` (`[openentropy]`), `[vllm]` and `[vllm-metal]` (marker extras)
- **Implicit:** vLLM V1 (provides `LogitsProcessor` base class, `torch`). Not listed as a dependency since the adapter runs inside vLLM's process. Other engines may have their own implicit deps.
