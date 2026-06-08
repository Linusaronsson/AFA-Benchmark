# AGENTS.md - Development Guide for AI Coding Agents

This guide provides essential information for AI coding agents working with the AFA-Benchmark codebase.

## Project Overview

**Name:** afa-benchmark
**Description:** A benchmark of active feature acquisition (AFA) methods
**Python Version:** 3.12.10 (exact version required)
**Package Manager:** uv (use the repository `uv.lock`)
**Main Package:** `afabench/`

## Quick Start Commands

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
pre-commit install

# Run all quality checks (format, lint, type check, tests)
just qa

# Run tests
uv run pytest .
```

## Build, Lint & Test Commands

### Required Final Verification

Run `just qa` before reporting any code change as complete. This is the
project's source of truth for quality because it runs formatting, linting,
type checking, and tests together. Focused checks are useful while iterating,
but they are not a substitute for `just qa` unless the user explicitly asks to
skip it or a concrete blocker prevents running it. If `just qa` fails, fix the
failure before finishing or clearly report the unresolved failure.

### Formatting & Linting

```bash
# Format code with ruff
uv run ruff format .

# Lint code with auto-fix
uv run ruff check . --fix

# Type check with basedpyright
uv run basedpyright --warnings

# Run all pre-commit hooks, including basedpyright
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
uv run pytest .

# Run tests with verbose output
uv run pytest . -v

# Run a single test file
uv run pytest test/path/to/test_file.py

# Run a specific test function
uv run pytest test/path/to/test_file.py::test_function_name

# Run a specific test class
uv run pytest test/path/to/test_file.py::TestClassName

# Run tests with custom options (defined in test/conftest.py)
uv run pytest . --device=cuda           # Use CUDA device
uv run pytest . --cores=4               # Use 4 CPU cores
uv run pytest . -m optional             # Run optional tests
uv run pytest . -m "not optional"       # Skip optional tests (default)
uv run pytest . -m pipeline             # Run pipeline/system tests
uv run pytest . -m "not pipeline"       # Skip pipeline tests (default)
uv run pytest . --no-smoke-test         # Run full tests instead of smoke tests
uv run pytest . --force-rerun           # Force rerun, ignore existing outputs

# Pipeline tests with specific methods and datasets
uv run pytest . -m pipeline --methods jafa ol --datasets cube_without_noise

# Note: Pretrain configs are automatically determined from selected methods
# Example: --methods odin_model_free will only pretrain 'pvae'
#          --methods odin_model_free jafa will pretrain 'pvae' and 'jafa'

# Run coverage analysis
just coverage                           # Generates HTML report in htmlcov/
```

### Dependency Management

```bash
# Add a new dependency
uv add <package>

# Add a development dependency
uv add --dev <package>

# Update lock file
uv lock

# Sync environment with lock file
uv sync
```

## Code Style Guidelines

### General Style

- **Line length:** 79 characters (strictly enforced)
- **Python version:** 3.12.10 exact (use modern Python features)
- **Formatter:** ruff (automatic formatting)
- **Linter:** ruff (ALL rules enabled with specific ignores)
- **Type checker:** basedpyright (recommended mode, relaxed settings)

### Imports

- Import order managed automatically by ruff
- Group imports: standard library, third-party, local
- Implicit namespace packages allowed (no `__init__.py` required everywhere)
- Example:
  ```python
  import os
  from pathlib import Path

  import torch
  from jaxtyping import Float

  from afabench.core.bundle_system.bundle import save_bundle
  from afabench.core.registry import Registry
  ```

### Type Annotations

- Type hints encouraged but not strictly required
- Use jaxtyping for tensor shape specifications
- Use Python 3.12+ type alias syntax: `type Features = Float[Tensor, "batch features"]`
- Return type annotations required except for `__init__` methods
- Example:
  ```python
  from jaxtyping import Float
  from torch import Tensor

  type Features = Float[Tensor, "batch features"]
  type Labels = Float[Tensor, "batch"]

  def predict(features: Features) -> Labels:
      ...
  ```

### Naming Conventions

- Follow PEP 8 with some relaxations:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
  - Type aliases: `PascalCase` (e.g., `Features`, `Labels`)
- N806 (lowercase variables in functions) and N812 (lowercase imports) are ignored where needed

### Error Handling

- Explicit error handling preferred
- Use appropriate exception types
- Logging with f-strings allowed (G004 ignored)
- Example:
  ```python
  import logging

  logger = logging.getLogger(__name__)

  def process_data(data: dict) -> None:
      if "required_field" not in data:
          raise ValueError("Missing required_field in data")
      logger.info(f"Processing data with {len(data)} fields")
  ```

### Documentation

- Docstrings not required for all functions (incrementally adopting)
- Use clear, descriptive function/variable names
- Document complex logic with inline comments
- Key modules should have module-level docstrings

### Testing

- Framework: pytest
- Test files: `test_*.py` in `test/` directory
- Test functions: `test_*`
- Test classes: `Test*`
- Asserts allowed in tests (S101 ignored)
- Use pytest fixtures for setup/teardown
- Mark optional/slow tests with `@pytest.mark.optional`
- Mark pipeline/system tests with `@pytest.mark.pipeline`
- Example:
  ```python
  import pytest

  def test_basic_functionality():
      result = my_function(input_data)
      assert result == expected_output

  @pytest.mark.optional
  def test_expensive_operation():
      # Long-running test
      ...

  @pytest.mark.pipeline
  class TestPipeline:
      # End-to-end system tests
      def test_full_workflow():
          ...
  ```

### Code Organization

- **Registry pattern:** Use `afabench.core.registry` for class lookup and extensible components
- **Bundle system:** Use `afabench.core.bundle_system` for `.bundle/` serialization
- **Configuration:** Put script configs under `extra/conf/scripts/<script_group>/<script_name>/`
- **Type definitions:** Define reusable type aliases at module level
- Example:
  ```python
  from afabench.core.registry import Registry

  my_registry: Registry[MyClass] = Registry()

  @my_registry.register("my_implementation")
  class MyImplementation(MyClass):
      ...
  ```

### Allowed Relaxations (from ruff.toml)

- Print statements allowed (T201, T203) for scripts
- TODO comments allowed (TD002, TD003, TD005)
- Magic values allowed (PLR2004)
- Many arguments allowed in functions (PLR0913)
- Boolean positional arguments allowed (FBT001, FBT003)
- Commented code allowed during development (ERA001)

## Project Structure

```
afabench/                   # Main source package
├── core/                  # Registry, bundle system, shared types, naming, utilities
├── components/            # Active implementation modules used by scripts and configs
│   ├── classifiers/       # Classifier wrappers and dummy classifiers
│   ├── initializers/      # Initial feature-mask initializers
│   ├── methods/           # AFA methods grouped by family
│   │   ├── discriminative/
│   │   ├── dummy/
│   │   ├── generative/
│   │   ├── oracle/
│   │   ├── rl/
│   │   └── static/
│   └── unmaskers/         # Feature unmasking strategies
├── datasets/              # Dataset definitions, aliases, wrappers, and utilities
├── training/              # Shared training helpers, configs, and smoke-test support
├── evaluation/            # Evaluation logic and config dataclasses
├── plotting/              # Plotting config and helpers
└── testing/               # Reusable test helpers

test/                      # Test suite
├── src/                  # Source/component tests
│   ├── afa_discriminative/
│   ├── afa_generative/
│   ├── afa_oracle/
│   ├── afa_rl/
│   └── common/
├── scripts/              # Script integration and smoke tests
├── config/               # Hydra/config schema tests
└── workflow/             # Snakemake workflow config tests

scripts/                   # Executable scripts
├── dataset_generation/   # Dataset generation
├── pretrain_model/       # Pretraining scripts
├── train_classifier/     # Classifier training scripts
├── train_method/         # AFA method training scripts
├── eval/                 # Evaluation scripts
├── plotting/             # Plotting scripts
├── visualizations/       # Dataset/visual inspection scripts
├── misc/                 # Data transformation and maintenance utilities
└── dev/                  # Developer maintenance scripts

extra/                     # Non-source files
├── conf/                 # Hydra configuration files for scripts and globals
├── data/                 # Local/raw dataset and miscellaneous data storage
├── logs/                 # Local logs
├── output/               # Generated pipeline outputs and bundles
└── workflow/             # Snakemake workflows, configs, profiles, and envs
    ├── conf/             # Workflow config sets
    ├── envs/             # Workflow Conda environment files
    ├── profiles/         # Snakemake execution profiles
    ├── snakefiles/       # Orchestration and rule Snakefiles
    └── src/              # Workflow Python support code

docs/                      # User and developer documentation
data/, outputs/, plots/    # Root-level local/generated artifacts; do not rely on these in tests
```

## Development Workflow

1. **Before starting work:**
   ```bash
   uv sync                 # Ensure dependencies are up to date
   ```

2. **While developing:**
   - Write code following style guidelines
   - Add tests for new functionality in `test/` directory
   - Use type hints with jaxtyping for tensors

3. **Before finishing or committing code changes:**
   ```bash
   just qa                 # Required final quality gate
   ```
   Individual commands are only for iteration and debugging, not a replacement
   for the final `just qa` run:
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   uv run basedpyright --warnings
   uv run pytest .
   ```

4. **Pre-commit hooks automatically:**
   - Fix trailing whitespace and EOF issues
   - Format code with ruff
   - Lint and auto-fix with ruff
   - Sync exclude patterns between configs
   - Type check with basedpyright

## Important Notes

- **Excluded files:** `ruff.toml` currently has no project-specific excludes; keep it synchronized with `pyrightconfig.json` through the pre-commit hook when this changes
- **Bundle format:** Serializable objects use `.bundle/` directory format (see docs/bundle_format.md) through `afabench.core.bundle_system`
- **Hydra configs:** Scripts use `@hydra.main()` decorator for configuration management
- **CUDA support:** Optional GPU acceleration via cupy-cuda12x (Linux only)
- **Experiment tracking:** Weights & Biases integration (run `uv run wandb login` if needed)
- **Snakefile documentation:** When modifying Snakefiles in `extra/workflow/snakefiles/orchestration/`, always update the docstring at the top of the file to document configuration arguments, required files, and usage examples

## Reference Files

- `pyproject.toml` - Project metadata and dependencies
- `ruff.toml` - Linting and formatting configuration
- `pyrightconfig.json` - Type checking configuration
- `pytest.ini` - Test configuration
- `.pre-commit-config.yaml` - Pre-commit hooks
- `justfile` - Common development commands
- `docs/` - Additional documentation
