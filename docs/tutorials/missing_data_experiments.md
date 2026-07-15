# Missing-training-data experiments

The dedicated workflow in
`extra/workflow/snakefiles/orchestration/missing_data.smk` applies the
missing-training-data study to the same dataset and method catalogs as the
ordinary evaluation pipeline. Missingness is limited to immutable training
and validation views. Evaluation always starts cold on the complete
validation or test bundle.

The configuration is split by responsibility:

- ordinary benchmark files define datasets, methods, pretrained models,
  classifiers, unmaskers, and budgets;
- `extra/workflow/conf/missing_data/design.yaml` defines the scientific
  missingness and completion matrix;
- `extra/workflow/conf/missing_data/{smoke,local,full_validation,full_test}.yaml`
  define runtime scale and the evaluation split;
- Snakemake execution profiles define scheduler and hardware resources.

The workflow selects the largest configured hard evaluation budget for every
eligible method-dataset pair and preserves any configured evaluation-to-train
budget mapping. A pair is omitted when that method excludes hard-budget runs
for the dataset. The full profiles use the KDD26 catalogs. Imagenette is
currently excluded because the missingness boundary does not yet define an
image-aware masking mechanism; the other selected datasets are handled
generically.

## Baselines and controls

The shared completion ladder contains:

- restricted observations;
- mean completion;
- PVAE (label-conditioned);
- PVAE (label-free);
- PVAE (oracle);
- true completion.

Complete-data runs provide the ceiling/reference. Zero fill is only an AACO
k-NN-search control and is attached to AACO; it is not presented as a general
imputation baseline. The reweighting controls are also method-specific:
doubly robust support correction is derived from AACO, and feature-marginal
inverse-probability weighting is derived from DIME. They inherit their base
method's shared script, pretraining, dataset eligibility, and budget
configuration.

The incomplete-data PVAE respects the fixed factual support mask. One joint
reconstruction is drawn per row, and factual cells are preserved. The oracle
PVAE is fitted to complete training data. Downstream methods receive ordinary
immutable dataset bundles, keeping missingness-specific logic at the data
boundary.

## Runtime profiles

Run the one-instance, 128-row CUBE-NM smoke matrix with:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_smoke \
  --cores 4
```

The smoke matrix covers AACO, DIME, model-free ODIN, both reweighting
controls, label-conditioned, label-free, and oracle PVAE restoration,
evaluation, summarization, and plotting. It writes only to the `smoke`
namespace.

For small local results, run two 512-row instances of CUBE-NM and CUBE under
MCAR and MAR:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_local \
  --cores 4
```

The local profile uses reduced training schedules without enabling smoke-test
shortcuts. It writes to the separate `local` namespace.

Run full validation before final test evaluation:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_full_validation \
  --cores 4

uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_full_test \
  --cores 4
```

The two full profiles share the `full` namespace and identical training
settings. The test profile changes only `eval_dataset_split`, so it reuses the
frozen validation-run methods and evaluates them once on the complete test
split.

To define another experiment, pair `design.yaml` and the ordinary catalog
files with a new runtime YAML in a profile. Select `datasets` and `methods` in
that runtime file exactly as in the evaluation pipeline. No Snakefile edit or
stage number is needed.

## Devices and hardware resources

Both the ordinary evaluation pipeline and the missing-data pipeline accept a
scalar default device:

```yaml
device: cuda
```

They also accept optional fine-grained overrides:

```yaml
device: cpu
device_overrides:
  datasets:
    cube: mps
  pretrained_models:
    pvae: cuda:0
  methods:
    aaco: cuda:1
  method_datasets:
    aaco:
      cube_nm: cuda:2
```

Resolution proceeds from most specific to least specific:
method-dataset, method, pretrained model, dataset, then the scalar default.
Method overrides apply to training and evaluation. Pretrained-model overrides
apply to shared pretraining jobs. Dataset overrides apply to shared classifier
jobs and provide the fallback for method/pretraining jobs. Missing-data
control variants first look up their own method name, then their base method,
so `aaco` also routes `aaco_doubly_robust` unless the control receives a more
specific override.

The `device` value is an application argument such as `cpu`, `mps`, `cuda`, or
`cuda:1`; it does not reserve hardware. GPU counts, memory, partitions, and
scheduler constraints belong in a Snakemake execution profile. Keep those
resource declarations independent from the experiment definition.

One-off scalar overrides remain possible, for example
`--config device=cpu use_wandb=false`, but checked-in runtime configuration is
preferable for reproducibility.

## Summaries and figures

Each run writes raw per-instance metrics and dataset-scoped mean and standard
error summaries to:

```text
extra/output/missing_data/summary/{eval_split}/{namespace}/
```

The files include per-instance performance, aggregate performance, per-action
acquisition rates, and PVAE restoration RMSE. Results are averaged over
dataset instances only. Metrics are never pooled across different datasets.

The workflow renders PDF and SVG figures under:

```text
extra/output/missing_data/figures/{eval_split}/{namespace}/dataset-{dataset}/
```

Each dataset receives:

- accuracy plots, or macro-F1 for datasets configured to use F1, by
  missingness mechanism and completion strategy;
- corresponding gaps to that method's complete-training reference;
- acquisition-rate heatmaps by mechanism and missingness probability;
- PVAE restoration RMSE plots.

All uncertainty bands and error bars are mean plus or minus one standard error
over instances of that dataset. Rerunning a profile schedules missing or
outdated plots automatically; no separate plotting command is required.
