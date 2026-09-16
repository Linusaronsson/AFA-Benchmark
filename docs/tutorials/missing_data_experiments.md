# Missing-training-data experiments

From the repository root, install the locked Python 3.12.10 environment:

```sh
uv sync --locked
```

Reproduce the current induced-missingness study with one Snakemake command:

```sh
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4
```

The DAG generates datasets, trains predictors and acquisition methods, evaluates
policies, and produces summaries, the exact control study, route descriptors,
and paper figures/tables. Add `--dry-run` to inspect it. Completed outputs are
reused by Snakemake; use `--rerun-incomplete` after an interrupted run.

The `plots` target updates the figures and tables from their dependencies:

```sh
uv run snakemake --profile extra/workflow/profiles/config/missing_data --cores 4 plots
```

This reuses existing results. Missing upstream results are scheduled normally;
use `--dry-run` first when working from a partial result archive.

The study contains eight datasets, nine policy variants, five instances,
four mechanisms, and three missingness rates. Each method has a complete-data
reference plus restricted and label-conditioned PVAE-restored training.
This gives 9,000 evaluation cells. The evaluation split is validation, matching
the current study; this command does not open the sealed test split.

## Configuration and outputs

- `extra/workflow/conf/missing_data/study.yaml` selects the current study.
- `extra/workflow/conf/missing_data/design.yaml` holds shared missingness and
  training settings. The ordinary benchmark catalogs define methods, budgets,
  acquisition groups, and dataset implementations.
- `extra/workflow/conf/missing_data/smoke.yaml` selects a small CUBE-NM check.
- `extra/workflow/conf/missing_data/native.yaml` is the separate factual-native
  study, run with profile `missing_data_native_missingness`. It enforces source
  availability and has no complete-data or oracle ceiling.

Current results use the `induced` namespace under `extra/output/missing_data/`.
The full study writes publication artifacts to `extra/output/missing_data/results/`.
The manuscript is managed separately and remains ignored by Git; reproducing
experiment artifacts does not require its source. No PDF compilation or copying
into the manuscript directory is performed automatically.

Historical output directories are not moved, renamed, or used as implicit
fallback inputs. The retired pilot configurations and AACO doubly robust / DIME
IPW controls have been removed. Existing support-aware AACO bundles remain
loadable; loading a retired doubly robust bundle fails explicitly.

## Data requirements

The synthetic datasets are generated locally. Real datasets use the existing
AFABench loaders and require their source data or network access for loaders
that download data. NHANES requires the source snapshot documented in
[`extra/data/nhanes_mortality/README.md`](../../extra/data/nhanes_mortality/README.md),
including its checksum verification. Large source tables are not included in
Git. See the dataset configuration and loader for each data location; the
workflow does not substitute synthetic data when a real source is absent.

## Smoke check and hardware

```sh
uv run snakemake --profile extra/workflow/profiles/config/missing_data_smoke --cores 4
```

The smoke check uses 128 CUBE-NM instances, one seed, MCAR 0.5, all nine policy
variants, and shortened training. Its namespace is `smoke`; it does not produce
paper-level aggregates or replace the production study.

Both profiles default to CPU. For CUDA, set the device and bound simultaneous
GPU processes separately:

```sh
uv run snakemake --profile extra/workflow/profiles/config/missing_data \
  --cores 4 --config device=cuda --resources gpu=1
```

The existing `scripts/workflow/run_missing_data.sh` remains an optional resource
runner for a Slurm allocation, with `--profile missing_data`. The Arrhenius
submission script remains available. Neither is required for local execution.
Run manifests record the resolved configuration, Git revision, and hardware;
Snakemake benchmark files record timings. A dry-run does not start training.

## Scientific conventions

Missingness masks are fitted on training data and fixed for each instance.
Acquisition groups are masked together; entirely missing instances are rejected.
PVAE pretraining uses only factual observations. Restoration draws one joint
reconstruction per instance and preserves observed values. `pvae` denotes the
complete-data oracle generator; `pvae_missing` denotes incomplete-data training.

Scores use the external predictor's final prediction. Accuracy is primary for
the three synthetic datasets; macro-F1 is primary for real datasets. Five
instances are paired before family averaging and bootstrap intervals.
JAFA, OL, and ODIN retain separate acquired-state and availability-conditioned
policy variants. Internal method training can also train its own predictor.

Route selection uses training data and route scoring uses validation data,
with 2,000 sampled legal routes. Route descriptors do not gate dataset inclusion.
The compute panel uses MCAR 0.5; the main performance panel uses MCAR 0.7.
Timing comparisons require compatible hardware/software and include amortized
restoration costs. Runtimes are not expected to reproduce across hardware.

Run `just qa` before accepting code changes. A smoke run establishes workflow
execution, not reproduction of production effect sizes.
