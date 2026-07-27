# Confirmatory missing-data runbook

This runbook keeps the completed `synthetic_missingness` artifacts immutable.
New results live in `core_group_missingness_v1` and
`nhanes_mortality_v1`. Run every command from the repository root.

Set writable caches once per shell:

```bash
export UV_CACHE_DIR=/tmp/afa-uv-cache
export XDG_CACHE_HOME=/tmp/afabench-cache
export MPLCONFIGDIR=/tmp/afabench-matplotlib
```

## 1. Verify code and sources

```bash
uv sync
(cd extra/data/nhanes_mortality && sha256sum -c SHA256SUMS)
uv run python scripts/dataset_generation/build_nhanes_schema.py
git diff --exit-code -- extra/data/nhanes_mortality/schema.csv
just qa
```

## 2. Local smoke checks

These use one instance, one missingness cell, small datasets, and smoke-test
training. The core smoke includes one-shot and stepwise PVAE.

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_core_group_smoke \
  --cores 4 --rerun-incomplete --printshellcmds

uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_nhanes_mortality_smoke \
  --cores 4 --rerun-incomplete --printshellcmds
```

## 3. Clean CUBE/CUBE-NM validation

The complete matrix has 3,370 evaluation cells. Stepwise PVAE contributes 90
secondary MCAR cells. It does not alter the primary one-shot gate.

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_core_group \
  --cores 4 --rerun-incomplete --printshellcmds

uv run python scripts/analysis/route_redundancy.py \
  --namespace core_group_missingness_v1 \
  --selection-split val --split val --device cuda \
  --seed 0 --k 500 --top-frac 0.1 --max-samples 4096 \
  --route-batch-size 16
```

Open
`extra/output/missing_data/analysis/route_gate_core_group_missingness_v1.csv`.
CUBE-NM proceeds to test only when both AACO and OL pass all three
predeclared gains at MCAR `p=0.7`: adaptive, nongreedy, and one-shot
restoration mean at least `0.01`, with each contrast positive in at least four
of five instances.

If it passes, freeze the validation CSVs and run:

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_core_group \
  --config eval_dataset_split=test \
  --cores 4 --rerun-incomplete --printshellcmds

uv run python scripts/analysis/route_redundancy.py \
  --namespace core_group_missingness_v1 \
  --selection-split val --split test --device cuda \
  --seed 0 --k 500 --top-frac 0.1 --max-samples 4096 \
  --route-batch-size 16
```

## 4. Staged NHANES Mortality

Stage one has 20 evaluations: four methods, five train/validation splits, one
training and evaluation budget of 10. Every split has its own classifier and
all five share one stratified test cohort fixed with seed 100.

```bash
uv run snakemake \
  --profile \
  extra/workflow/profiles/config/missing_data_nhanes_mortality_complete \
  --cores 4 --rerun-incomplete --printshellcmds

uv run python scripts/analysis/route_redundancy.py \
  --namespace nhanes_mortality_v1 \
  --selection-split val --split val --device cuda \
  --seed 0 --k 500 --top-frac 0.1 --max-samples 4096 \
  --route-batch-size 16
```

Continue only if
`extra/output/missing_data/analysis/planning_gate_nhanes_mortality_v1.csv`
has `dataset_concordant=true` for both AACO and OL. Then run the full
restoration profile:

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_nhanes_mortality_full \
  --cores 4 --rerun-incomplete --printshellcmds
```

Apply the same restoration gate. Only after it passes, evaluate test by adding
`--config eval_dataset_split=test` to that Snakemake command, then run route
analysis with `--selection-split val --split test`.

## 5. File-count-safe archiving

Mechanism tables and figures are generated before raw traces are archived:

- `extra/output/missing_data/analysis/`
- `extra/output/missing_data/analysis_figures/`

Archive one completed split at a time. Do not remove the source until all
three verification commands succeed:

```bash
tar --zstd -cf /tmp/core-group-val.tar.zst \
  extra/output/missing_data/eval/val/core_group_missingness_v1
tar --zstd -tf /tmp/core-group-val.tar.zst >/dev/null
sha256sum /tmp/core-group-val.tar.zst
find extra/output/missing_data/eval/val/core_group_missingness_v1 \
  -printf . | wc -c
```

Record the checksum and entry count in `HANDOFF.md`, move the archive to the
chosen durable archive directory, and only then remove the archived raw trace
tree. Keep datasets, classifiers, trained methods, summaries, analysis CSVs,
figures, and manifests through final paper regeneration.
