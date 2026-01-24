# AGENTS.md - Development Guide for AI Coding Agents

This guide provides essential information for AI coding agents working with the AFA-Benchmark codebase.

## Project Overview

**Name:** afa-benchmark
**Description:** A benchmark of active feature acquisition (AFA) methods
**Python Version:** 3.12.10 (exact version required)
**Package Manager:** uv (v0.9.25)
**Main Package:** `afabench/`

## Quick Start Commands

```bash
# Install dependencies
uv sync

# Install pre-commit hooks
pre-commit install

# Run all quality checks (format, lint, type check, test)
just qa

# Run tests
uv run pytest .
```

## Build, Lint & Test Commands

### Formatting & Linting

```bash
# Format code with ruff
uv run ruff format .

# Lint code with auto-fix
uv run ruff check . --fix

# Type check with basedpyright
pre-commit run basedpyright --all-files

# Run all pre-commit hooks
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

# Run tests with custom options
uv run pytest . --device=cuda           # Use CUDA device
uv run pytest . --cores=4               # Use 4 CPU cores
uv run pytest . -m optional             # Run optional tests
uv run pytest . -m "not optional"       # Skip optional tests (default)

# Run tests without smoke tests
uv run pytest . --no-smoke-test

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

  from afabench.common.bundle import Bundle
  from afabench.common.registry import Registry
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
  ```

### Code Organization

- **Registry pattern:** Use `afabench.common.registry.Registry` for extensible components
- **Bundle system:** Use `afabench.common.bundle.Bundle` for serialization
- **Configuration:** Use Hydra configs with dataclasses in `config_classes.py`
- **Type definitions:** Define reusable type aliases at module level
- Example:
  ```python
  from afabench.common.registry import Registry

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
├── afa_rl/                # RL-based AFA methods
├── afa_discriminative/    # Discriminative methods
├── afa_generative/        # Generative methods
├── afa_oracle/            # Oracle methods
├── static/                # Static feature selection
├── common/                # Shared utilities
└── eval/                  # Evaluation utilities

test/                      # Test suite
├── afa_rl/               # RL method tests
├── common/               # Common utilities tests
└── scripts/              # Script integration tests

scripts/                   # Executable scripts
├── dataset_generation/   # Dataset generation
├── pretrain/             # Pretraining scripts
├── train/                # Training scripts
├── eval/                 # Evaluation scripts
└── plotting/             # Visualization scripts

extra/                     # Non-source files
├── conf/                 # Hydra configuration files
├── data/                 # Dataset storage (gitignored)
└── workflow/             # Snakemake workflows
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

3. **Before committing:**
   ```bash
   just qa                 # Run all quality checks
   ```
   Or individually:
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   pre-commit run basedpyright --all-files
   uv run pytest .
   ```

4. **Pre-commit hooks automatically:**
   - Fix trailing whitespace and EOF issues
   - Format code with ruff
   - Lint and auto-fix with ruff
   - Sync exclude patterns between configs
   - Type check with basedpyright

## Important Notes

- **Excluded files:** Large number of legacy files excluded from linting (see ruff.toml lines 4-78)
- **Bundle format:** Serializable objects use `.bundle/` directory format (see docs/bundle_format.md)
- **Hydra configs:** Scripts use `@hydra.main()` decorator for configuration management
- **CUDA support:** Optional GPU acceleration via cupy-cuda12x (Linux only)
- **Experiment tracking:** Weights & Biases integration (run `uv run wandb login` if needed)

## Reference Files

- `pyproject.toml` - Project metadata and dependencies
- `ruff.toml` - Linting and formatting configuration
- `pyrightconfig.json` - Type checking configuration
- `pytest.ini` - Test configuration
- `.pre-commit-config.yaml` - Pre-commit hooks
- `justfile` - Common development commands
- `docs/` - Additional documentation
