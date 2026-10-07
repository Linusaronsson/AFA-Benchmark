# Coding standards

Read during review. Formatting, import order, line length (79) and the
allowed lint relaxations are enforced by `just qa` (`ruff.toml`,
`pyrightconfig.json`); this file covers only what no check can.

## Code

- **Tensor shapes**: annotate tensors with jaxtyping and name reusable
  shapes as module-level type aliases (`type Features = Float[Tensor,
  "batch features"]`).
- **Errors**: raise a specific exception type with a message that names the
  bad value; no silent fallbacks to defaults.
- **Comments and docstrings**: docstrings are optional for functions; key
  modules get a module docstring. Comment the reason behind non-obvious
  logic, not what the code does.
- **Vocabulary**: names, messages and docs use `CONTEXT.md` terms (selection
  vs action, classifier, built-in classifier, myopic, hard budget,
  soft-budget parameter, dataset instance).
- **ADRs**: a change that contradicts a decision in `docs/adr/` must say so
  and justify reopening it.
- **Command-line parsing**: new scripts that are not Hydra-configured use
  `typer`, not `argparse`. Existing `argparse` scripts are left alone until
  they are otherwise rewritten.

## Tests

- Test behaviour through public interfaces; a test should survive an
  internal refactor.
- Mark slow tests `optional` and end-to-end pipeline tests `pipeline`, so the
  default suite stays fast: `just qa` has a 1-minute budget (`AGENTS.md`).
  Tests under `test/workflow/` are marked `workflow` automatically.
- Tests use `tmp_path` and generated data, never `data/`, `outputs/`,
  `plots/` or `extra/output/`.
