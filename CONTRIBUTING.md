# Contributing to qr-sampler

Thank you for your interest in contributing to qr-sampler. This document covers everything you need to get started.

## Development setup

### Prerequisites

- Python 3.10 or later
- Git

### Install

```bash
git clone https://github.com/Entropic-Science/qr-sampler.git
cd qr-sampler
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies (pytest, ruff, mypy, scipy, pre-commit, bandit, gRPC, click, jinja2).

### Pre-commit hooks

```bash
pre-commit install
```

This runs ruff (lint + format), mypy, bandit, and standard checks (trailing whitespace, YAML/TOML validation, merge conflict detection) on every commit.

## Running tests

```bash
# All tests
pytest tests/ -v

# Specific subsystem
pytest tests/test_config.py -v
pytest tests/test_amplification/ -v
pytest tests/test_temperature/ -v
pytest tests/test_selection/ -v
pytest tests/test_logging/ -v
pytest tests/test_entropy/ -v
pytest tests/test_qthought_roller.py -v
pytest tests/test_statistical_properties.py -v
pytest tests/test_core/ -v
pytest tests/test_engines/ -v
pytest tests/test_profiles/ -v
pytest tests/test_cli/ -v

# With coverage
pytest tests/ -v --cov=src/qr_sampler --cov-report=term-missing
```

Coverage must stay at or above 90%.

## Code quality checks

```bash
# Lint
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format check
ruff format --check src/ tests/

# Auto-format
ruff format src/ tests/

# Type check
mypy --strict src/

# Security scan
bandit -c pyproject.toml -r src/ -q

# Or run every oracle at once (what CI runs)
python scripts/check.py
```

## Coding conventions

- **Python 3.10+** — use `X | Y` union syntax, not `Union[X, Y]`
- **Type hints** on all function signatures and return types
- **Docstrings** — Google style on every public class and method
- **Imports** — standard library first, third-party second, local third. No wildcard imports.
- **Line length** — 100 characters (configured in `pyproject.toml`)
- **Errors** — custom exception hierarchy rooted in `QRSamplerError`. Never catch bare `Exception` (health checks are the sole exception with `# noqa` comments).
- **No `print()`** — use `logging.getLogger("qr_sampler")` for all output
- **No global mutable state** outside processor/adapter instances and module-load registries
- **Frozen dataclasses** for all result types (`AmplificationResult`, `TemperatureResult`, `SelectionResult`, `TokenSamplingRecord`, `SamplingResult`)
- **`__slots__`** on frozen dataclasses in the hot path
- **Pydantic `ConfigDict(frozen=True)`** for profile schemas and config models

## Architecture invariants

These are fundamental design rules. Do not break them:

1. **No hardcoded values.** Every numeric constant traces to a `QRSamplerConfig` field. Mathematical constants (e.g., `sqrt(2)`) are acceptable.

2. **Registry pattern for all strategies.** New `EntropySource`, `SignalAmplifier`, `TemperatureStrategy`, or `EngineAdapter` implementations are registered via decorators. No if/else chains for strategy selection.

3. **ABCs define contracts.** All concrete implementations subclass the relevant ABC. The pipeline and adapters only reference abstract types.

4. **SEM is derived, never stored.** Standard error of mean = `population_std / sqrt(sample_count)`, computed at amplification time.

5. **Per-request config is immutable.** `resolve_config()` creates new instances; it never mutates defaults. Infrastructure fields cannot be overridden per-request.

6. **Just-in-time entropy.** Physical entropy generation must occur only when `get_random_bytes()` is called — after logits are available. No pre-buffering.

7. **The core pipeline has zero engine dependencies.** `qr_sampler.core` is importable and functional without vLLM, torch, MLX, or any engine package. Only numpy and runtime deps are used.

8. **Engine adapters are thin wrappers.** They convert logits to numpy, call `SamplingPipeline.sample_token()`, convert the result back to engine-native tensors, and force one-hot logits. No sampling math in adapters.

9. **Profiles are declarative data, not code.** Profile YAML files are for CLI validation and documentation only. Missing or corrupt profiles never cause sampling failure.

## Adding new components

### New engine adapter

1. Create a class in `src/qr_sampler/engines/` subclassing `EngineAdapter`
2. Implement `get_pipeline(self) -> SamplingPipeline`
3. Use `build_pipeline(config, vocab_size)` from `core/pipeline.py` to construct the pipeline
4. Implement engine-specific hook methods (e.g., `apply()` for vLLM)
5. Built-in: add it to `EngineAdapterRegistry._BUILTINS` in `engines/registry.py`. Third-party: `@EngineAdapterRegistry.register("my_engine")` or an entry point
6. Add entry point in `pyproject.toml` under `[project.entry-points."qr_sampler.engine_adapters"]`
7. Create a YAML profile in `src/qr_sampler/profiles/engines/my_engine.yaml`
8. Add tests in `tests/test_engines/`

### New entropy source

1. Create a class in `src/qr_sampler/entropy/` subclassing `EntropySource`
2. Implement `name`, `is_available`, `get_random_bytes(n)`, `close()`
3. Raise `EntropyUnavailableError` from `get_random_bytes()` on failure
4. Built-in: add it to `EntropySourceRegistry._BUILTINS` in `entropy/registry.py`. Third-party: register with `@register_entropy_source("my_name")` or an entry point
5. Optionally mirror it in `pyproject.toml` under `[project.entry-points."qr_sampler.entropy_sources"]` (the builtin table wins on collisions)
6. Create a YAML profile in `src/qr_sampler/profiles/entropy/my_name.yaml`
7. Add tests in `tests/test_entropy/`

### New signal amplifier

1. Create a class in `src/qr_sampler/amplification/` subclassing `SignalAmplifier`
2. Implement `amplify(raw_bytes) -> AmplificationResult`
3. Built-in: add it to the `_BUILTINS` table in `amplification/registry.py`. Third-party: `@AmplifierRegistry.register("my_name")`
4. Create a YAML profile in `src/qr_sampler/profiles/amplifiers/my_name.yaml`
5. Add tests in `tests/test_amplification/`

### New temperature strategy

1. Create a class in `src/qr_sampler/temperature/` subclassing `TemperatureStrategy`
2. Implement `compute_temperature(logits, config) -> TemperatureResult`
3. Always compute and return Shannon entropy (the logging subsystem depends on it)
4. Built-in: add it to the `_BUILTINS` table in `temperature/registry.py`. Third-party: `@TemperatureStrategyRegistry.register("my_name")`
5. Create a YAML profile in `src/qr_sampler/profiles/samplers/my_name.yaml`
6. Add tests in `tests/test_temperature/`

### New config field

1. Add the field to `QRSamplerConfig` in `config/model.py` with a default and description
2. If per-request overridable, mark it with `Field(json_schema_extra={"per_request": True})` — the per-request set is derived from that metadata
3. Environment variable `QR_{FIELD_NAME_UPPER}` is auto-supported
4. Extra args key `qr_{field_name}` is auto-supported
5. Add tests in `tests/test_config.py`

### New profile

1. Create a YAML file in the appropriate `src/qr_sampler/profiles/<category>/` directory
2. Follow the schema defined by the corresponding Pydantic model in `profiles/schema.py`
3. Include `id`, `name`, `description`, and all relevant compatibility cross-references
4. Add tests in `tests/test_profiles/` to validate the new profile loads correctly

## Pull request process

1. Fork the repository and create a branch from `main`
2. Make your changes, following the coding conventions above
3. Add or update tests for your changes
4. Run the full quality gate:
   ```bash
   python scripts/check.py
   ```
5. Write a clear PR description explaining *what* and *why*
6. Submit the PR for review

## Reporting bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version, OS, inference engine and version
- Relevant configuration (`QR_*` env vars)

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
