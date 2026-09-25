# Missing training data experiments

```sh
uv sync --locked

# full study
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4

# full study on GPU
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4 \
  --config device=cuda --resources gpu=1

# synthetic controlled study only
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4 \
  extra/output/paper/experiments/results/exact_study_combined.pdf

# figures and tables from existing results
uv run snakemake --profile extra/workflow/profiles/config/missing_data_plots --cores 4

# smoke test (CUBE-NM, one seed, MCAR 0.5)
uv run snakemake --profile extra/workflow/profiles/config/missing_data_smoke --cores 4
```

Figures and tables are written to `extra/output/paper/experiments/results/`, and
runs to `extra/output/missing_data/`.

## Configuration

- `extra/workflow/conf/missing_data/study.yaml`: datasets, methods, strategies
- `extra/workflow/conf/missing_data/design.yaml`: missingness mechanisms and rates, training settings
- `extra/workflow/conf/missing_data/smoke.yaml`: smoke test

## Data

Synthetic datasets are generated. Real datasets use the AFABench loaders. NHANES
needs the snapshot in [`extra/data/nhanes_mortality/`](../../extra/data/nhanes_mortality/README.md).
