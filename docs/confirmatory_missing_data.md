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
training. The core smoke includes generative restoration and its stepwise PVAE control.

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
secondary MCAR cells. It does not alter the primary generative-restoration gate.

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_core_group \
  --cores 4 --rerun-incomplete --printshellcmds

uv run python scripts/analysis/route_redundancy.py \
  --namespace core_group_missingness_v1 \
  --selection-split train --split val --device cpu \
  --seed 0 --k 2000 --max-samples 4096 \
  --route-batch-size 16
```

Open
`extra/output/missing_data/analysis/route_gate_core_group_missingness_v1.csv`.
CUBE-NM proceeds to test only when both AACO and OL pass all three
predeclared gains at MCAR `p=0.7`: adaptive, nongreedy, and generative-restoration
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
  --selection-split val --split test --device cpu \
  --seed 0 --k 2000 --max-samples 4096 \
  --route-batch-size 16
```

## 4. Real-data planning diagnostic

Complete-data only, 60 evaluations: four methods, three datasets, five
train/validation splits. Every split has its own preprocessing and its own
classifier, and each dataset's five splits share one stratified test cohort
fixed with seed 100. Each method trains once at its largest budget and is
evaluated at every configured budget.

```bash
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_real_planning \
  --cores 4 --rerun-incomplete --printshellcmds

uv run python scripts/analysis/route_redundancy.py \
  --namespace real_planning_v1 \
  --selection-split train --split val --device cpu \
  --seed 0 --k 2000 --max-samples 4096 \
  --route-batch-size 16
```

A restoration matrix on a real dataset requires
`extra/output/missing_data/analysis/planning_gate_real_planning_v1.csv` to
report `dataset_concordant=true` for both AACO and OL. Without that, the
dataset has no measurable planning gap for restoration to recover, and the
reportable result is the route-redundancy measurement itself.

NHANES Mortality was first run alone under `nhanes_mortality_v1` and failed
this gate, with AACO mean adaptive gain `-0.009867` and OL `-0.009119`, both
`0/5` positive. That namespace is retained unchanged, so the NHANES rows here
double as an exact reproduction check.

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
