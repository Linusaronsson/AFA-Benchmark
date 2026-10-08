# Release manifest

`release_manifest.json` is written beside a snapshot's `output/` tree by
`snapshot.py save --release-id` and restored to `extra/release_manifest.json`
([command](snapshot_command.md)). It identifies a benchmark release and
indexes its artifacts, so that a download can choose what to fetch without
reading every artifact first. The index is generated from the artifacts'
own [provenance records](../adr/0002-provenance-recorded-in-artifacts.md),
never from the workflow configuration: it is a cache of those records, and
saving the same output tree again gives the same index
([ADR 0006](../adr/0006-release-manifest-indexes-artifact-provenance.md)).
The schema is `afabench.release.manifest.ReleaseManifest`, version 3.
Version 3 added the [job duration table](#job-duration-table); version 2
manifests are not read, since no release was published with one.

`save --release-id` refuses an output tree holding any bundle, or any
Parquet file under `eval_results/` or `eval_results_transformed/`, without
a provenance record, and lists them. Merged tables, plots and other
derived outputs carry no record and are not indexed; they are part of the
snapshot all the same.

## Fields

All paths inside the manifest are POSIX paths relative to the output root,
except config file paths, which are as given, and the `path` of an input,
which is as the producing job was given it. JSON object keys are strings.
`null` means unknown or not applicable, never a default.

### Release

| Field | Meaning |
| --- | --- |
| `manifest_version` | Schema version, 3. Readers reject versions they do not know. |
| `release_id` | The identity given with `--release-id`. |
| `scope` | `full`, `partial` or `smoke`, as declared. Only `full` and `partial` are published as benchmark releases; `smoke` only as a smoke release. |
| `execution_mode` | `smoke` if any artifact's record or any job record has `smoke_test`, `production` otherwise. Dataset generation has no smoke mode, so dataset bundles always record production. `smoke` implies scope `smoke`. |
| `created_at` | UTC ISO-8601 time the manifest was built. |
| `dataset_redistribution` | Per dataset key any record names, the maintainers' review from the checkout: `status` (`unreviewed`, `permitted` or `restricted`), `license`, `source`, `reviewed_by`, `notes`. See [Dataset redistribution](#dataset-redistribution). |
| `job_duration_table` | The release's [job duration table](#job-duration-table): `size_bytes`, `job_records` (its rows) and `smoke_test` (whether any row is of a smoke test); null when the output tree holds no job record. |

There is no release-wide commit: each index entry has the `code` that
produced it. `save` and `publish` print them per pipeline stage (see
[`save`](snapshot_command.md#save)).

### `workflow_config`

The workflow configuration the maintainer declares with `--profile`,
`--configfile` and `--config`: the one whose targets lay out the release.
A [repository adopter](../how-to/compare_your_method_with_published_results.md)
layers a comparison over `merged`, so that the workflow finds the published
tables where the release has them. It is recorded, never read to describe
an artifact; a release built across commits has the configuration of the
final run, which reused the restored outputs because its targets name them.

| Field | Meaning |
| --- | --- |
| `profile` | The `--profile` directory, or null. |
| `configfiles` | One `{path, sha256}` per config file, in merge order. |
| `overrides` | The `--config` (or profile `config`) values. |
| `merged` | The full config Snakemake sees after merging. |

### `code` of an entry

| Field | Meaning |
| --- | --- |
| `commit` | The record's `code_commit`: `git rev-parse HEAD` of the checkout that produced the artifact; null outside a git work tree. |
| `dirty` | The record's `code_dirty`: whether tracked files differed from the commit; null outside a git work tree. |

An artifact is produced from **clean** code when `commit` is set and
`dirty` is `false`. `publish` refuses an official release holding any
other artifact unless the maintainer allows dirty code.

### `evaluations`

One entry per evaluation in the output tree: its raw table, its
plotting-ready table, or both, paired because the transform copies the raw
table's record. Identity is read from the tables' identity columns
([raw versus plotting-ready tables](#raw-versus-plotting-ready-tables)).

| Field | Meaning |
| --- | --- |
| `raw_path`, `transformed_path` | The raw and plotting-ready Parquet tables; null when that table is not in the release. |
| `raw_size_bytes`, `transformed_size_bytes` | File sizes; null when absent. |
| `code`, `smoke_test` | From the evaluation's record. |
| `method_name`, `dataset_key`, `dataset_realization_index`, `eval_split`, `initializer` | The columns `afa_method`, `dataset`, `dataset_realization_index`, `eval_split` and `initializer`. |
| `budget_setting` | `hard_budget`, or `soft_budget` when `eval_hard_budget` is null. |
| `train_seed`, `train_hard_budget`, `train_soft_budget_param` | Training run of the evaluated method bundle. |
| `eval_seed`, `eval_hard_budget`, `eval_soft_budget_param` | Evaluation settings. A hard-budget evaluation uses forced acquisition; a soft-budget one lets the policy stop. |
| `classifier_variants` | Which of `builtin` and `external` have a non-null prediction, read from the raw table, or from the plotting-ready table without one. |
| `inputs` | The bundles the evaluation loaded, as `{role, path, content_hash}` with roles `eval_dataset`, `method` and `classifier`. |

### `bundles`

One entry per bundle in the output tree, whatever its folder.

| Field | Meaning |
| --- | --- |
| `path` | The bundle folder. |
| `category` | From the record's stage: `dataset_bundle` (dataset generation), `classifier_bundle` (classifier training), `pretrained_model_bundle` (pretraining) or `afa_method_bundle` (training). |
| `class_name` | The bundle's class, which names the registered loader it needs. |
| `content_hash` | The bundle's own `content_hash` ([bundle format](bundle_format.md)). |
| `size_bytes` | Bytes of all files in the bundle. |
| `stage`, `code`, `smoke_test`, `seed` | From the record. |
| `method_name` | The method an AFA-method bundle or a method's own classifier belongs to; null for shared prerequisites. |
| `dataset_key`, `dataset_realization_index` | The dataset realization the bundle was generated or trained from. |
| `split` | `train`, `val` or `test` for a dataset bundle; null otherwise. |
| `inputs` | The bundles the producing job read, as `{role, path, content_hash}` with roles `train_dataset`, `val_dataset`, `classifier` and `pretrained_model`. Dataset bundles have none. |

The full record (resolved configuration, environment, compute, input class
names) stays in the artifact: the bundle's `manifest.json`, or the Arrow
schema metadata of a table.

### Inputs

An input is the bundle in the release with its `content_hash`, not the
bundle at its `path`: a bundle regenerated after the artifact was produced
still has the path, but not the hash. An input no bundle of the release
matches is a **dangling input**. `save` lists them, and a selective download
reports those of the categories it fetches as missing. A partial release
may leave bundles out on purpose.

### `coverage`

Derived from the index: `datasets`, `dataset_realization_indices`,
`methods`, `eval_splits`, `budget_settings` and `classifier_variants` of the
evaluations, and `output_categories` (top-level directories of the output
root that contain files, such as `datasets`, `trained_methods`,
`eval_results`, `eval_results_transformed`, `merged_results` and
`plot_results`, and `failed_job_records` when a job failed or timed out;
see [Job duration table](#job-duration-table)).

`payloads` has one entry per [payload category](#payload-categories), in
the order of that table: `category`, `count`, `size_bytes` and
`class_names` (the bundle classes present; empty for tables). The
`job_duration_table` entry has `count` 1 when the release has the table, 0
otherwise.

## Payload categories

Snapshots copy the output root verbatim, so every payload is restored
byte for byte into the path the workflow expects, and Snakemake reuses it
instead of regenerating it. Native bundles load with
`afabench.core.bundle_system.bundle.load_bundle` (datasets without
arguments, the others with `device=`), which looks up `class_name` in
`afabench/core/registry.py`. No HF-friendly conversion is produced; any
later conversion is added beside the native files, never instead of them.

| Category | Path | Enables | Kind |
| --- | --- | --- | --- |
| `raw_evaluation_table` | `eval_results/` | Acquisition-history analysis with any Parquet reader. Restoring it skips that evaluation. | Results |
| `transformed_evaluation_table` | `eval_results_transformed/` | Aggregation and plotting without the raw tables. | Results |
| `dataset_bundle` | `datasets/<key>/<realization>/<split>.bundle` | Training and evaluating any method on exactly the published dataset realization and split, without regenerating it. | Shared prerequisite |
| `classifier_bundle` | `trained_classifiers/` | External-classifier predictions for any method's evaluation on a dataset realization (`dataset-<key>+realization_index-<realization>.bundle`, one per dataset realization), without retraining it. `method-<name>+dataset-<key>+realization_index-<realization>.bundle` is a classifier one method trains with and is needed only to retrain that method. | Shared prerequisite (external); method-specific otherwise |
| `pretrained_model_bundle` | `pretrained_models/` | Training any method that names the same pretrained model, without repeating pretraining. | Shared prerequisite |
| `afa_method_bundle` | `trained_methods/` | Re-evaluating a published method (other seeds, budgets, splits) or inspecting its policy, without retraining it. | Optional baseline |
| `job_duration_table` | `release_job_duration_table.parquet` beside the manifest, not in `output/` | Estimating the compute of a pipeline invocation from the release's measured job durations, or inspecting them with any Parquet reader, without running anything. | Results |

A bundle is a shared prerequisite when its `method_name` is null. Plotting a
new method against published baselines needs only the evaluation tables of
those baselines; evaluating a new method needs the dataset bundles and the
external classifier of each dataset realization, and the pretrained model it trains
from, if the release has it. AFA-method bundles are never needed for
comparison plots. Follow `inputs` to find what a bundle or table was
produced from.

Pretrained-model and AFA-method bundles are restored together with the
[job record](job_records.md) their job wrote in the same folder.

Bundles are the inference format the evaluator loads. Intermediate training
checkpoints, such as the Lightning checkpoints of classifier training, are
written under `extra/logs/`, outside the output root, so they are never in
a snapshot: they serve resuming or debugging one training run, are not
loadable by `load_bundle`, and are not a reusable payload.

## Job duration table

The release's [job duration table](job_records.md#job-duration-table) has
one row per job record under the output root: the completed records beside
artifacts and every failed or timed-out attempt under
`failed_job_records/`. `save --release-id` builds it from those records,
with `afabench.core.job_duration_table`, and writes it beside the manifest
as `release_job_duration_table.parquet`. It is not the copy the
`collect_job_records` rule wrote to `merged_results/`, which is only as
recent as that rule's last run; that copy stays in the output tree like any
merged table. `restore` and `download` put the table beside the restored
manifest, at `extra/release_job_duration_table.parquet` by default.

It lives outside `output/` because the workflow rebuilds
`merged_results/job_duration_table.parquet` from the local output root on
every `all` run: a release table restored there would be replaced by the
adopter's own records. Beside the manifest it stays the release's table, so
a compute estimate can be pointed at it before or after running anything
(`load_job_duration_table` reads it as it reads any job duration table).

`--payload-category job_duration_table` downloads it alone, without any
output tree; coverage options do not narrow it, since it holds every job of
the release. A release without job records has no table: the manifest
records it as null, and asking for it reports
`job_duration_table: not in the release`.

Each row keeps the record's `smoke_test`. A smoke job record marks the
table `smoke_test` and makes the release's `execution_mode` smoke, so a
`full` or `partial` release never ships durations that a compute estimate
refuses.

`failed_job_records/` is part of the output tree and is snapshotted, listed
in `output_categories` and downloadable with `--output-category` like any
other folder. Its records are already rows of the job duration table, whose
`exit_status` tells them apart, so estimating needs only the table.

## Dataset redistribution

Being able to generate a dataset bundle grants no right to publish it. Each
dataset key any record names is recorded in `dataset_redistribution` from the
maintainers' reviews in `extra/conf/release/dataset_redistribution.yaml` of
the checkout. That file ships empty, so every dataset is `unreviewed`, and
`save`, `restore`, `publish` and `download` print the unreviewed and
restricted keys. Only a `permitted` dataset's bundles may be published;
unreviewed and restricted ones stay out of any public release until
reviewed. `publish` refuses an official release holding any of them
unless the maintainer allows each by its dataset key (see
[`publish`](snapshot_command.md#publish)).

What a dataset bundle holds decides what the review covers:

- Synthetic datasets (CUBE and its variants, synthetic MNIST) are generated
  by this repository's code; their bundles hold generated tensors.
- Tabular datasets come from third parties: diabetes, MiniBooNE and
  PhysioNet from the CSV files in `extra/data/misc/`, ACTG, CKD and bank
  marketing from the UCI repository. Their bundles hold the preprocessed
  data itself.
- MNIST and Fashion-MNIST bundles hold the downloaded data; Imagenette
  bundles hold only generation indices and configuration, so the images must be
  obtained from the source by each user.

A review entry records:

- `status`: `permitted` or `restricted`.
- `license`: the licence or terms the review relied on.
- `source`: where the data comes from, so users can obtain it themselves.
- `reviewed_by`: who reviewed it, and when.
- `notes`: attribution or other conditions, such as whether models trained
  on the data (classifier, pretrained-model and AFA-method bundles) may be
  published when the data itself may not.

Evaluation tables hold predictions, costs and true class labels, not
features; whether labels of a restricted dataset may be published is part
of its review.

## Smoke-scale inventory

Measured with `inventory` on the smoke run of
`test/workflow/test_native_bundle_restore.py`: the `all` profile with
`datasets=[cube]`, `dataset_realization_indices=[0]`,
`methods=[random_dummy, gdfs]`, `eval_hard_budgets={cube: [2]}`, a soft
budget only for `random_dummy`, and `smoke_test=true`. These are
**smoke-scale measurements of one CUBE dataset realization, not production sizes**,
which are unknown until `inventory` is run on a full production output
root.

| Category | Count | Bytes | Classes |
| --- | --- | --- | --- |
| `raw_evaluation_table` | 3 | 20,343 | |
| `transformed_evaluation_table` | 3 | 29,767 | |
| `dataset_bundle` | 3 | 119,065 | `CubeDataset` |
| `classifier_bundle` | 1 | 94,578 | `WrappedMaskedMLPClassifier` |
| `pretrained_model_bundle` | 1 | 95,267 | `GreedyAFAClassifier` |
| `afa_method_bundle` | 3 | 199,310 | `GDFSAFAMethod`, `RandomWithoutClassifierAFAMethod` |

The rest of that output root was `plot_results/` (5.1 MB, 134 files),
`merged_results/` (83 kB) and per-job timing files (10 kB) that job
records have since replaced. A production run multiplies the counts by
datasets, dataset realizations, methods and budgets, and per-payload sizes
change with dataset size, architecture and smoke settings, so these numbers
do not extrapolate.

## Raw versus plotting-ready tables

Both are Parquet and readable with any Parquet reader. They differ in row
granularity:

- **Raw evaluation tables** (`eval_results/.../eval_data.parquet`,
  `raw_path`) have one row per episode and time step, as described in
  [evaluation dataframes](evaluation_dataframes.md): `episode_id`,
  `generation_index`, `split_index`, `step`, `action_performed`, `builtin_predicted_class`,
  `external_predicted_class`, `true_class`, `accumulated_cost`, `forced_stop`,
  and the identity columns `afa_method`, `dataset`,
  `dataset_realization_index`, `eval_split`, `initializer`, `train_seed`,
  `train_hard_budget`, `train_soft_budget_param`, `eval_seed`,
  `eval_hard_budget` and `eval_soft_budget_param`, with their provenance
  record in the Arrow schema metadata. They hold the full acquisition
  history; selection histories can be
  reconstructed from `episode_id`, `step` and `action_performed`.
- **Plotting-ready tables** (`eval_results_transformed/.../eval_data.parquet`,
  `transformed_path`) are produced by
  `scripts/misc/transform_eval_data_pipeline.py`. Each raw row becomes two
  rows, one per `classifier` (`builtin` or `external`) with its
  `predicted_class`; `episode_id` and `step` are dropped in favour of
  `n_selections_performed`, so these are prediction/cost rows, not episode
  logs. They keep `generation_index` and `split_index` (null for tables
  transformed from legacy logs), every identity column and the raw table's
  provenance record.
- **Merged tables** (`merged_results/`) concatenate plotting-ready tables per
  method set and classifier type and carry no per-table identity beyond those
  columns.

A snapshot restores raw tables byte for byte, so values, null and NaN
entries, column types, Arrow schema metadata and the acquisition histories
survive (`test/scripts/test_release_native_payloads.py`).

Raw `episode_id` values are local to one evaluation table: the evaluator
numbers episodes within each batch and offsets them by batch, so they say
where in that run's sampled evaluation an episode came, not which instance
it evaluated. Use `generation_index` for that: together with the dataset key
and dataset realization it identifies the instance, so it supports paired
instance-level comparisons between methods. `split_index` is the instance's
position in the evaluated split's dataset bundle
([ADR 0004](../adr/0004-instances-carry-their-generation-index.md)).
