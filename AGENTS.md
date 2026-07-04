# AGENTS.md — Codebase Guide for Coding Agents

Companion to `LEARNINGS.md` (non-obvious lessons + a labelled history
section) and `README.md` (end-user documentation).

## What this project is

`qr-sampler` is an engine-agnostic framework that replaces standard LLM token
sampling with external-entropy-driven selection. It fetches random bytes from
any entropy source (QRNGs via gRPC, OS randomness, CPU timing jitter,
OpenEntropy), amplifies the signal into a uniform float via z-score or ECDF
statistics, and uses that float to select a token from a probability-ordered
CDF. The primary use case is weak-signal integration research: studying
whether small statistical biases in physical entropy sources are detectable
in LLM token selection. In server-draw mode the integration happens
server-side (Qbert0G's `qr_purity.PurityService`) and each draw arrives with
purity/coherence metadata; the research narrative lives in Qbert0G's README.

Two independent consumers sit on top of the library:

1. **vLLM** loads `VLLMAdapter` as a V1 logits processor via the
   `vllm.logits_processors` entry point (no vLLM source changes).
2. **`qr-llm-qthought`** (sibling checkout at
   `../Entropic-Science/qr-llm-qthought`, relative to this repo) imports the
   `QthoughtRoller` entropy stack — no vLLM, no GPU — through
   `qr_sampler.contract` (see below).

## Verification — one command

```bash
python scripts/check.py            # every oracle: lint, format, types, security, tests
python scripts/check.py --only lint,types
```

The oracle rows (CI and pre-commit invoke this same script):

| Check | Command |
|---|---|
| lint | `ruff check .` |
| format | `ruff format --check .` |
| types | `mypy --strict src/` |
| security | `bandit -c pyproject.toml -r src/ -q` |
| tests | `pytest tests/ -v --cov=src/qr_sampler` (coverage `fail_under=90`) |

A change is NOT done until every row passes. Use these as ground truth — do
not act as a linter yourself.

## Layering (imports only point down)

```
L0 foundation   exceptions.py, config/  (model + presets + resolve)
L1 core         core/, entropy/, amplification/, temperature/, selection/,
                logging/, telemetry/, proto/
L2 roller       qthought.py
L3 adapters     engines/
L4 periphery    cli/, profiles/, templates/, __main__.py, contract.py
```

Rules:

- Nothing in L0–L3 may import from `cli/`, `profiles/`, or `templates/` —
  the periphery is CLI/documentation tooling, never runtime sampling.
- `core/` has **zero engine dependencies**: numpy only, no torch/vLLM.
  Engine-specific code lives exclusively under `engines/`.
- `contract.py` is a pure re-export module (may import from anywhere; nothing
  internal imports it).
- Import side effects are forbidden. `import qr_sampler` and
  `import qr_sampler.engines.vllm` are 100% side-effect-free — no sockets, no
  monkey-patches, no file writes. Pinned by
  `tests/test_engines/test_import_time_socket_guard.py`.

## The cross-repo contract (`contract.py`)

`src/qr_sampler/contract.py` is **the only surface downstream consumers may
import**; `qr-llm-qthought` is the live consumer. It re-exports (grouped):

- roller + provenance: `QthoughtRoller`, `ChoiceProvenance`, `BindSpec`, `IntRange`
- config + presets: `QRSamplerConfig`, `resolve_config`, `resolve_preset`,
  `BUILTIN_PRESETS`, `PRESET_QTHOUGHT`, `PRESET_QTHOUGHT_THINK`, `PRESET_QTHOUGHT_VOICE`
- entropy primitives: `EntropySource`, `MockUniformSource`, `FallbackEntropySource`
- exceptions: `EntropyUnavailableError`, `ConfigValidationError`
- `CONTRACT_VERSION` — bump on ANY breaking change to this surface; qthought
  asserts it at import and fails loudly on mismatch.

Internal module boundaries are free to move as long as `contract.__all__`
keeps re-exporting the same names. `tests/test_contract.py` pins `__all__`,
the three qthought preset dicts (scientific lineage — do not touch their
values), and `inspect.signature` snapshots of the roller surface. Its
counterpart `tests/test_sampler_contract.py` lives in the qthought repo.
Breaking-seam changes must land atomically with the qthought consumer update
(both repos green in the same increment).

## File map

```
src/qr_sampler/
+-- __init__.py                # Version + top-level re-exports (side-effect-free)
+-- __main__.py                # `python -m qr_sampler` -> cli/main.py
+-- exceptions.py              # QRSamplerError -> {EntropyUnavailable, ConfigValidation, SignalAmplification, TokenSelection}Error
+-- contract.py                # Cross-repo seam (see above)
+-- qthought.py                # QthoughtRoller: typed random-choice family over the entropy stack (choose/coin/bind_int/draw_u/draw_index + ChoiceProvenance)
+-- py.typed
+-- config/
|   +-- model.py               # QRSamplerConfig (pydantic BaseSettings); PER_REQUEST_FIELDS derived from Field(json_schema_extra={"per_request": True})
|   +-- presets.py             # BUILTIN_PRESETS, PRESET_* name constants, resolve_preset(), expand_extra_args()
|   +-- resolve.py             # resolve_config(), validate_extra_args() — the single validation point
+-- core/
|   +-- pipeline.py            # SamplingPipeline + factories (build_pipeline, build_entropy_source, config_hash, derive_commit_nonce)
|   +-- types.py               # SamplingResult (frozen)
+-- entropy/
|   +-- base.py                # EntropySource ABC (get_random_bytes, prefetch/ticket hooks, health_check; DrawMeta + get_draw/prefetch_draw server-draw surface)
|   +-- registry.py            # EntropySourceRegistry: lazy _BUILTINS table + entry-point discovery
|   +-- system.py / timing.py / mock.py / openentropy.py
|   +-- fallback.py            # FallbackEntropySource composition wrapper (+ status-file publishing hook)
|   +-- qgrpc/                 # Quantum gRPC source, decomposed:
|       +-- source.py          #   QuantumGrpcSource facade + PrefetchTicket + warmup + health_check
|       +-- transport.py       #   unary/server-streaming/bidi dispatch, _BidiSession, pb2 encode/decode
|       +-- channel.py         #   background asyncio loop + channel lifecycle
|       +-- breaker.py         #   adaptive-P99 circuit breaker (pure class)
|       +-- preprobe.py        #   TCP pre-probe state machine
+-- amplification/             # SignalAmplifier ABC + registry; zscore.py, zscore_thought.py, ecdf.py, server_side.py ("server": server-integrated draws)
+-- temperature/               # TemperatureStrategy ABC + registry; fixed.py, edt.py, hvh_drift.py, coherence_gate.py
+-- selection/                 # TokenSelector: top-k -> softmax -> min-p -> top-p -> CDF
+-- logging/                   # TokenSamplingRecord + SamplingLogger (none/summary/full)
+-- telemetry/
|   +-- status_file.py         # Cross-process entropy-status file IPC (QR_ENTROPY_STATUS_FILE)
+-- proto/
|   +-- wire.py                # THE varint/tag/fixed64 codec (encode_varint, decode_varint, encode_tag, encode_fixed64, decode_fixed64)
|   +-- entropy_service.proto  # Canonical protocol definition
|   +-- entropy_service_pb2.py # Hand-written message stubs (import wire.py; the single wire format)
|   +-- entropy_service_pb2_grpc.py
|   +-- purity_service.proto   # PurityService (byte-identical to Qbert0G's copy; sha256-pinned in tests)
|   +-- purity_service_pb2.py  # Hand-written DrawRequest/DrawResponse stubs (import wire.py)
|   +-- purity_service_pb2_grpc.py
+-- engines/
|   +-- base.py                # EngineAdapter ABC
|   +-- registry.py            # EngineAdapterRegistry (lazy _BUILTINS + entry points)
|   +-- vllm/
|       +-- adapter.py         # VLLMAdapter (vLLM V1 LogitsProcessor), _RequestState
|       +-- telemetry.py       # _PerfAggregator (stage timings, perf status file)
+-- cli/                       # click commands: list / info / validate / build ([cli] extra)
+-- profiles/                  # Declarative YAML metadata for the CLI (never affects runtime sampling)
+-- templates/                 # Jinja2 templates rendered by `qr-sampler build`

scripts/check.py               # The one-command verification runner
tests/                         # Mirrors src/ layout; see "Testing approach"
examples/servers|docker|systemd
deployments/                   # Per-host compose profiles (generic, non-Modal)
```

## Architecture invariants — DO NOT break these

1. **No hardcoded values.** Every numeric constant traces to a named
   `QRSamplerConfig` field (including the QRNG quota limits
   `qrng_max_bytes_per_request` / `qrng_max_requests_per_minute` /
   `qrng_max_bytes_per_day`). Pure math constants are fine.
2. **Registry pattern for all strategies.** Four registries (entropy,
   amplification, temperature, engines), all with the same shape: an explicit
   lazy `_BUILTINS` table (`name -> "module:Class"`, imported on first
   `get()`), a `register()` decorator for runtime/third-party classes, and
   lazy entry-point discovery. Builtin table takes precedence over entry
   points. No import-side-effect registration anywhere.
3. **ABCs define contracts**: `EntropySource`, `SignalAmplifier`,
   `TemperatureStrategy`, `EngineAdapter`. Core code references only the
   abstract types.
4. **FallbackEntropySource is a composition wrapper.** Catches only
   `EntropyUnavailableError`; everything else propagates.
5. **SEM is derived, never stored**: `population_std / sqrt(N)` computed at
   amplification time.
6. **Frozen dataclasses for all result types** (`AmplificationResult`,
   `TemperatureResult`, `SelectionResult`, `TokenSamplingRecord`,
   `SamplingResult`).
7. **Per-request config resolution.** `resolve_config(defaults, extra_args)`
   builds a new validated instance; never mutates defaults. The per-request
   field set is **derived from field metadata**
   (`Field(json_schema_extra={"per_request": True})` in `config/model.py`) —
   never a hand-maintained list. Infrastructure fields (gRPC address/mode,
   fallback mode, quotas, `oe_conditioning`, ...) are rejected per-request
   with `ConfigValidationError`.
8. **Engine adapters force one-hot logits** (selected token 0.0, rest -inf).
9. **Logging uses `logging.getLogger("qr_sampler")`**; no `print()` in
   production code.
10. **Just-in-time entropy.** Bytes are fetched only when
    `get_random_bytes()` is called — after logits exist. The pipelined
    prefetch (commit-then-fetch, `entropy_prefetch`) preserves the
    post-selection causal contract via commitment nonces
    (`derive_commit_nonce`); `tests/test_pipelined_prefetch.py` pins it.
11. **Circuit breaker protects the gRPC source** (rolling P99 window,
    adaptive timeout, exponential half-open backoff). Lives in
    `entropy/qgrpc/breaker.py` as a pure class.
12. **Engine adapters are thin.** Tensor conversion + delegation to
    `SamplingPipeline.sample_token()` only; all sampling math in `core/`.
13. **Profiles are declarative data for the CLI only** — never consulted on
    the runtime sampling path; a corrupt YAML must never break sampling.
14. **Stateful temperature strategies are per-request** (`hvh_drift` EMA
    state lives on a per-request instance via `_RequestState`).
15. **Selector order is `top-k -> softmax -> min-p -> top-p -> CDF`**;
    `min_p=0.0` is a strict no-op. Pinned by `tests/test_selection/`.
16. **Presets are a thin resolution layer over `extra_args`.**
    `BUILTIN_PRESETS` in `config/presets.py` is the runtime source of truth;
    YAML files under `profiles/presets/` are documentation kept in sync by
    `tests/test_presets/test_yaml_sync.py`. The three historical `qthought*`
    presets are scientific lineage — value changes require a
    `CONTRACT_VERSION` bump. The serve-path preset lineage extends to
    `qthought_purity` (published 2026-07): its dict is pinned by
    `tests/test_contract.py` too.
17. **One wire format.** All varint/tag/fixed64 encoding lives in
    `proto/wire.py`; the pb2 stubs and the qgrpc transport share it. Decode
    follows pb2 semantics (last field-1 occurrence wins; empty payload raises
    `EntropyUnavailableError` in the transport). No hand-rolled codecs
    elsewhere. Pinned by `tests/test_wire_format.py`. `purity_service.proto`
    is byte-identical to Qbert0G's copy — both repos pin the same sha256.
18. **The draw path preserves commit-then-fetch.** Server-integrated draws
    (`get_draw`/`prefetch_draw`) carry the same commitment nonce in
    `DrawRequest.sequence_id` and apply the identical echo-verification rule
    as the byte path. Prefetch semantics are shared via `PrefetchTicket`.
19. **The coherence gate is fail-safe by construction.** Every failure branch
    in `temperature/coherence_gate.py` (no draw meta, first token, malformed
    meta, invalid coherence, boosted-inner failure, unknown inner) yields
    exactly the unboosted base temperature and never raises; the boost can
    only add temperature. An unboosted inner failure propagates — that is an
    inner-strategy bug, not gate machinery. Same spirit at the pipeline
    level: a failed draw degrades to local bytes + `zscore_mean` with
    `entropy_is_fallback=True`, so an EntropyService-only server keeps
    working.
20. **Neutral language in this repo.** The research narrative (and its
    vocabulary) lives in Qbert0G; qr-sampler describes itself in terms of
    weak-signal integration and entropy purity verification. Enforced for
    `src/` by `tests/test_language_scrub.py`.

### Config shape: flat, deliberately

`QRSamplerConfig` stays a **flat** field namespace (no nested sub-models).
Rejected alternative, recorded during the 2026-07 refactor: nesting
(e.g. `config.grpc.timeout_ms`) would add a mapping layer for zero consumer
benefit, because flat field names are used end-to-end by preset dicts,
`qr_*` extra-args keys, and `QR_*` env names.

## How to add components

### New entropy source (builtin)

1. Create the class in `src/qr_sampler/entropy/` subclassing `EntropySource`
   (implement `name`, `is_available`, `get_random_bytes(n)`, `close()`;
   raise `EntropyUnavailableError` when bytes cannot be produced).
2. Add it to `EntropySourceRegistry._BUILTINS` in `entropy/registry.py`
   (`"my_name": "qr_sampler.entropy.my_module:MyClass"`).
3. Optionally mirror it in `[project.entry-points."qr_sampler.entropy_sources"]`
   in `pyproject.toml` for external discoverability (the builtin table wins
   on name collisions — test-pinned).
4. Add a YAML profile in `profiles/entropy/` + tests in `tests/test_entropy/`.

Third-party packages skip steps 2–4 and register via their own entry point,
or at runtime with `@register_entropy_source("name")`.

### New amplifier / temperature strategy / engine adapter

Same pattern against the respective ABC + registry `_BUILTINS` table
(`amplification/registry.py`, `temperature/registry.py`,
`engines/registry.py`) + YAML profile + tests. Temperature strategies must
always compute `shannon_entropy`; constructors needing `vocab_size` accept it
as first positional arg. Engine adapters implement `get_pipeline()` and the
engine-specific hook (e.g. `apply()` for vLLM), and are also exposed via the
`qr_sampler.engine_adapters` entry-point group.

### New config field

1. Add to `QRSamplerConfig` in `config/model.py` with
   `Field(default=..., description=...)`; mark per-request-overridable fields
   with `json_schema_extra={"per_request": True}` — that metadata IS the
   per-request set.
2. Env var `QR_<FIELD>` and extra-args key `qr_<field>` work automatically.
3. Tests in `tests/test_config.py` (the derived-set pin will fail loudly on
   metadata mistakes).

### New preset

1. Add the dict to `BUILTIN_PRESETS` in `config/presets.py` (inner keys
   without the `qr_` prefix).
2. Create the matching `profiles/presets/<id>.yaml` (the sync test iterates
   `BUILTIN_PRESETS` automatically).
3. If a downstream repo must reference it by name, add a `PRESET_*` constant
   and export it via `contract.py` (contract-version rules apply).

## Key data flows

### Per-token sampling (in `core/pipeline.py::sample_token`)

```
temperature strategy -> entropy fetch (just-in-time or prefetch-ticket)
  -> amplifier (bytes -> u in (0,1)) -> TokenSelector(logits, T, u)
  -> one-hot -> SamplingLogger -> SamplingResult
```

### vLLM adapter (`engines/vllm/adapter.py`)

```
vLLM calls VLLMAdapter.apply(logits)
  -> per batch row: torch -> numpy -> pipeline.sample_token(...)
  -> next-token prefetch ticket handoff -> write one-hot into the torch tensor
```

The `apply()` sequence (to_numpy -> `sample_token` -> next-ticket handoff ->
one-hot force) is invariant; `tests/test_engines/test_vllm_lp_abi.py` pins
the entry-point string `qr_sampler.engines.vllm:VLLMAdapter`.

### Config resolution

```
env (QR_*) -> QRSamplerConfig()            # process defaults
extra_args (qr_*, incl. qr_preset) -> resolve_config(defaults, extra_args)
  -> expand_extra_args (preset expansion) -> validate_extra_args -> new instance
```

### QthoughtRoller (`qthought.py`)

One decision = one full-size entropy fetch reduced to a uniform `u` — the
same statistical shape as one token-sampling step. Surface: `begin_thought`,
`choose`, `choose_weighted`, `coin`, `bind_int`, `drain`, `status`, plus the
buffer-free `draw_u()` / `draw_index(k)` (fresh draw, provenance returned
directly, no thought scope). Constructor seam:
`QthoughtRoller(config=None, *, entropy_source=None)`.

## Coding conventions

- Python 3.10+ (`X | Y` unions), full type hints, Google-style docstrings on
  public API, line length 100.
- Custom exceptions rooted in `QRSamplerError`; never catch bare `Exception`
  (health checks and plugin loading are the documented exceptions).
- `QR_` env prefix, `qr_` extra-args prefix.
- Conventional commits; commit only on green oracles; never push without
  being asked.

## Testing approach

- No real QRNG server, GPU, or vLLM needed: `MockUniformSource` + numpy
  everywhere; gRPC is tested against mocked channels/stubs
  (`tests/test_entropy/qgrpc_util.py`).
- Frozen gates (behavioral anchors — must pass with at most import-path
  updates): `test_statistical_properties.py` (KS uniformity, bias
  detectability, EDT monotonicity — requires scipy),
  `test_selection/*`, `test_wire_format.py`, `test_pipelined_prefetch.py`,
  `test_qthought_roller.py`, `test_contract.py`.
- Import hygiene: `test_engines/test_import_time_socket_guard.py` asserts no
  sockets AND no monkey-patches at import time.
- CLI via click's `CliRunner`; profiles are data-driven-validated.

## Proto stubs

`src/qr_sampler/proto/` is hand-written (not protoc-generated): just enough
for the client + example servers. The wire primitives live in `proto/wire.py`
and are the only codec in the repo. If the proto changes, update the stubs by
hand and extend `tests/test_wire_format.py`.

## History

The Modal deployment surface (`connectors/`), the Open WebUI integration, the
`processor.py` shim, and the `contseq` roller were removed wholesale in the
2026-07 refactor (no backward compatibility kept; git history serves any
archaeology). `LEARNINGS.md` keeps the still-relevant entropy/vLLM lessons
and moves the Modal/OWUI/Cipherstone material to a labelled history section.
