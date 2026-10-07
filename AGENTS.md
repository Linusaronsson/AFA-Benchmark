# AGENTS.md

AFABench: a benchmark of active feature acquisition (AFA) methods. Library
code is in `afabench/`, pipeline entry points in `scripts/`, Hydra configs,
Snakemake workflows and outputs in `extra/`, tests in `test/`.

## Quality gate

`just qa` (format, lint, type check, fast tests; see `justfile`) must pass
before any code change is reported complete or committed. Focused commands
are for iterating only. If it fails, fix it or report the failure verbatim.

Budget: `just qa` finishes within 1 minute. When adding tests, check
`uv run pytest --durations=20` and keep the gate inside the budget; tests
that spawn processes (Snakemake, scripts, DataLoader workers) or train
models are usually the slow ones. If the gate exceeds the budget, say so
when reporting.

Environment: Python 3.12.10 exactly, managed by uv (`uv sync`). Pytest
markers `optional` and `pipeline` are deselected by default (`pytest.ini`);
select them with `-m`.

## Where to look

- **Domain vocabulary**: `CONTEXT.md` is the glossary and the naming source
  of truth. Architecture decisions are in `docs/adr/`; read the ones touching
  your area. How to consume both: `docs/agents/domain.md`.
- **Review rules**: `CODING_STANDARDS.md`.
- **Bundles**: objects are saved as `.bundle/` folders through
  `afabench.core.bundle_system` (`docs/reference/bundle_format.md`). Any
  class saved as a bundle needs an entry in `REGISTERED_CLASSES` in
  `afabench/core/registry.py`, or `load_bundle` cannot rebuild it.
- **Hydra configs**: script configs live under
  `extra/conf/scripts/<script_group>/<script_name>/`, shared groups under
  `extra/conf/components/`. Read `extra/conf/components/README.md` before
  changing defaults lists or how the pipeline passes overrides.
- **Snakemake**: when editing `extra/workflow/snakefiles/orchestration/`,
  update the docstring at the top of the file (config arguments, required
  files, usage examples).
- **Docs**: before writing, splitting or moving a page in `docs/`, read
  `docs/README.md`: pages follow Diátaxis, one folder per type.
- **Adding a method or dataset**: `docs/how-to/`.

## Agent skills

Development follows the `matt-pocock` skill set; use those skills where they
apply.

- **Issue tracker**: GitHub Issues for `Linusaronsson/AFA-Benchmark` via
  `gh`. See `docs/agents/issue-tracker.md`.
- **Triage labels**: `needs-triage`, `needs-info`, `ready-for-agent`,
  `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.
- **Domain docs**: single context, `CONTEXT.md` plus `docs/adr/`. See
  `docs/agents/domain.md`.
