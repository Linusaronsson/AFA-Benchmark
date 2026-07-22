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
- `extra/workflow/conf/missing_data/smoke.yaml` defines the integration test;
- `extra/workflow/conf/missing_data/{synthetic_missingness,native_missingness,non_uniform_costs}.yaml`
  define the three main experiments;
- `extra/workflow/conf/missing_data/restoration_deployments.yaml` defines the
  focused three-instance comparison of episode-start and stepwise restoration;
- Snakemake execution profiles define scheduler and hardware resources.

The workflow selects the largest configured hard evaluation budget for every
eligible method-dataset pair and preserves any configured evaluation-to-train
budget mapping. A pair is omitted when that method excludes hard-budget runs
for the dataset. Imagenette is currently excluded because the missingness
boundary does not yet define an image-aware masking mechanism; the other
selected datasets are handled generically.

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
reconstruction is drawn per data instance, and factual cells are preserved.
The oracle PVAE is fitted to complete training data. Downstream methods receive
ordinary immutable dataset bundles, keeping missingness-specific logic at the
data boundary.

## Runtime profiles

Run the one-instance, 128-row CUBE-NM smoke matrix with:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_smoke \
  --cores 4
```

The smoke matrix covers AACO, DIME, OL, both reweighting controls,
episode-start and stepwise restoration, oracle and true-completion controls,
evaluation, summarization, and plotting. It writes only to the `smoke`
namespace.

Run one of the three named experiments on validation data:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_synthetic_missingness \
  --cores 4
```

The other profiles are `missing_data_native_missingness` and
`missing_data_non_uniform_costs`. The focused local comparison uses
`missing_data_restoration_deployments`. After model selection, evaluate the
same frozen namespace on test data by overriding only the evaluation split:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_synthetic_missingness \
  --config eval_dataset_split=test \
  --cores 4
```

Because the namespace is unchanged, the test command reuses the validation
run's frozen training artifacts and schedules only test evaluation and its
summaries.

To define another durable experiment, pair `design.yaml` and the ordinary
catalog files with one named experiment YAML and profile. One-off schedule
checks should use command-line overrides and a distinct artifact namespace,
not checked-in configuration files. No Snakefile edit or stage number is
needed.

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

- accuracy plots for CUBE and CUBE-NM, or macro-F1 for the imbalanced
  Diabetes, ACTG175, CKD, and PhysioNet datasets, by
  missingness mechanism and completion strategy;
- corresponding gaps to that method's complete-training reference;
- acquisition-rate heatmaps by mechanism and missingness probability;
- PVAE restoration RMSE plots.

All uncertainty bands and error bars are mean plus or minus one standard error
over instances of that dataset. Rerunning a profile schedules missing or
outdated plots automatically; no separate plotting command is required.

## Route diagnostics

`scripts/analysis/route_redundancy.py` compares the two non-greedy methods,
AACO and OL, separately against DIME and a fixed static reference. The static
reference is searched in the dataset's legal selection space: grouped features
remain indivisible and the sum of selection costs must fit the hard budget.
The script uses random feasible routes, greedy forward selection, and local
one-swap refinement. The result is therefore named `static_reference`, not an
exact best static route.

Two summaries describe fixed-route structure. `route_sensitivity` is the test
score of the validation-selected static reference minus the mean test score of
the sampled legal routes. `top_route_correctness_correlation` is the mean
pairwise correlation of test-set correctness among the top 10% of sampled
routes, where the top routes are chosen on validation data. Thus neither route
selection nor the definition of the top set reads test labels.

Run the diagnostic after `instance_metrics.csv` exists:

```console
uv run python scripts/analysis/route_redundancy.py \
  --namespace synthetic_missingness --split val --selection-split val \
  --device cuda
```

The analysis directory receives four tables:

- legal-route scores by dataset instance and budget;
- fixed-budget planning effects for AACO and OL separately;
- missingness damage and restoration gain for every mechanism and rate;
- the predeclared MCAR 0.7 gate, again separately for AACO and OL.

The missing-data workflow evaluates its largest configured hard budget. Route
curves at smaller budgets are structural context only; method contrasts join
the exact dataset, instance, method, mechanism, rate, strategy, and evaluation
budget and never average across those cells. Route search and every paired
contrast use the same predeclared dataset metric. For a final test analysis, use
`--selection-split val --split test` so the fixed route is chosen without
looking at test labels.
