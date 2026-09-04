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
- `extra/workflow/conf/missing_data/{core_group_missingness,induced_real,induced_nonuniform}.yaml`
  define the three current validation matrices;
- the runner defines hardware resources without changing scientific config.

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
generative and stepwise restoration, oracle and true-completion controls,
evaluation, summarization, and plotting. It writes only to the `smoke`
namespace.

The current validation matrices have three named entry points:

| Profile | Namespace | Datasets |
|---|---|---|
| `missing_data_core_group` | `core_group_missingness_v2` | CUBE, CUBE-NM |
| `missing_data_induced_real` | `induced_real_missingness_v2` | ACTG175, Diabetes, NHANES mortality |
| `missing_data_induced_nonuniform` | `induced_nonuniform_missingness_v2` | CUBE-NUC, heart disease |

Run the same resource-aware entry point locally and inside a Slurm allocation:

```console
scripts/workflow/run_missing_data.sh \
  --profile missing_data_core_group --device cuda \
  --cores 4 --mem-mb 22000 --gpu-workers 1
```

One CUDA-resolved rule consumes one `gpu` token. `--gpu-workers` bounds CUDA
process concurrency independently of the physical GPU count. Keep it at one
on small GPUs; a large-memory GH200 can host several measured workers while
independent data-preparation rules overlap on CPU. The
runner writes Git, hardware, PyTorch/CUDA, resources, namespace, and Slurm
provenance under `extra/output/missing_data/run_manifests/` before execution.

On Arrhenius, submit one allocation rather than thousands of short Slurm jobs:

```console
scripts/workflow/submit_missing_data_arrhenius.sh \
  --profile missing_data_gh200_pilot \
  --cores 16 --mem-mb 128000 --gpus 1 --gpu-workers 4 --time 02:00:00
```

The wrapper reads the live association run-minute cap, refuses an impossible
request, and runs `sbatch --test-only` before submitting. The fresh GH200
acceptance profile is one CUBE-NUC instance at MCAR 0.5: 118 workflow jobs and
39 evaluations. It is implementation evidence, not production data. Inside
the allocation every rule gets a private directory below `$SNIC_TMP`;
scientific artifacts remain on project storage.

After validation analysis and claims are frozen, reuse the frozen training
artifacts for one sealed-test evaluation pass. Do not inspect test outcomes
during model or claim selection.

To define another durable experiment, pair `design.yaml` and the ordinary
catalog files with one named experiment YAML and profile. No Snakefile edit or
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
`cuda:1`; it does not reserve hardware. GPU counts and memory are runner
resources. Hardware selection uses `AFABENCH_DEVICE` through the runner and
does not use Snakemake's command-line `--config`, which can replace a profile's
inline scientific overrides. Durable changes to datasets, instances, or cells
belong in a named config file.

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

## Compute accounting

Every artifact-producing rule records wall time, CPU time, and peak resident
memory. Collect complete namespaces and render the paired compute analysis:

```console
uv run python scripts/analysis/collect_compute.py \
  --namespace core_group_missingness_v2 \
  --namespace induced_real_missingness_v2 \
  --namespace induced_nonuniform_missingness_v2 \
  --output extra/output/missing_data/analysis/compute.csv
uv run python scripts/plotting/plot_compute.py \
  --compute extra/output/missing_data/analysis/compute.csv
```

The primary figure reports, within the same dataset, method, mechanism, rate,
instance, hardware, and software environment, the restored/restricted
wall-time ratio against restoration gain. Generator training is joined by its
dataset, mechanism, rate, and instance, then amortized over all actual restored
consumers. The paired CSV retains per-component wall time, total CPU time,
peak RAM, both method-run commits, generator provenance, workflow concurrency,
architecture, PyTorch, and CUDA. A true hardware or software mismatch rejects
the pair; a commit or workflow-worker change is retained as provenance.

To extend an existing collection with method-owned rows from a selectively
restored archive, pass `--base-input`, repeat `--method` for the newly restored
methods, and point `--output-root` at the local staging tree. Shared restoration
and generator records are deliberately taken from the base input rather than
duplicated from the later archive.

Native-missingness results remain exploratory until evaluation enforces each
instance's legal acquisition mask. Mean-imputed but factually absent values
must never become acquirable measurements.

## Route diagnostics

`scripts/analysis/route_redundancy.py` compares the two non-greedy methods,
AACO and OL, separately against DIME and a fixed static reference. The static
reference is searched in the dataset's legal selection space: grouped features
remain indivisible and the sum of selection costs must fit the hard budget.
The static reference is the selection-split best of the same sampled feasible
routes used to characterize the dataset; it is not claimed to be the global
best subset.

Two summaries describe fixed-route structure. `route_sensitivity` is the
evaluation score of the selection-split static reference minus the mean
evaluation score of the sampled legal routes. `weighted_route_overlap` is the
pairwise Jaccard overlap of the routes' acquisition-action sets, weighted by
each route's positive predictive gain over the empty route. The weights and
the static reference use only the selection split; route sensitivity is
reported on the requested evaluation split. The overlap is normalized to
`[0, 1]` and avoids an arbitrary score cutoff: high values mean that useful
routes repeatedly rely on the same acquisitions, while low values indicate
distinct substitute routes.

Run the diagnostic after `instance_metrics.csv` exists:

```console
uv run python scripts/analysis/route_redundancy.py \
  --namespace synthetic_missingness --split val --selection-split train \
  --k 2000 --device cpu
```

The diagnostic needs only the namespace's dataset and classifier bundles, so
it can run locally after those two trees are copied from archival storage. It
does not require trained AFA methods or evaluation traces to compute route
structure. If `instance_metrics.csv` is present locally, the same invocation
also refreshes the optional planning, missingness, and gate tables.

The analysis directory receives the route table and, when
`instance_metrics.csv` is available, four effect and gate tables:

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
