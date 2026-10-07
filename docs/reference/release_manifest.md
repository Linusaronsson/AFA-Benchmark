# Release manifest

`release_manifest.json` is written beside a snapshot's `output/` tree by
`snapshot.py save --release-id` and restored to `extra/release_manifest.json`
([command](snapshot_command.md)). It identifies a benchmark release and
records what produced the outputs, so that its tables can be interpreted
without knowing the output path layout. The schema is
`afabench.release.manifest.ReleaseManifest`, version 1.

## Fields

All paths inside the manifest are POSIX paths relative to the output root,
except config file and feature-cost paths, which are as given. JSON object
keys are strings. `null` means unknown or not applicable, never a default.

### Release

| Field | Meaning |
| --- | --- |
| `manifest_version` | Schema version, 1. Readers reject versions they do not know. |
| `release_id` | The identity given with `--release-id`. |
| `scope` | `full`, `partial` or `smoke`, as declared. Only `full` and `partial` are published as benchmark releases; `smoke` only as a smoke release. |
| `execution_mode` | `smoke` or `production`, from the merged config's `smoke_test`. `smoke` implies scope `smoke`. |
| `created_at` | UTC ISO-8601 time the manifest was built. |
| `code.commit` | `git rev-parse HEAD` of the checkout; null outside a git work tree. |
| `code.dirty` | Whether tracked files differ from the commit (untracked files are ignored); null outside a git work tree. |

### `workflow_config`

| Field | Meaning |
| --- | --- |
| `profile` | The `--profile` directory, or null. |
| `configfiles` | One `{path, sha256}` per config file, in merge order. |
| `overrides` | The `--config` (or profile `config`) values. |
| `merged` | The full config Snakemake sees after merging. |

### `settings`

Resolved by the workflow's own `load_config`
(`extra/workflow/src/config.py`), so defaults and per-dataset fallbacks match
the run.

| Field | Meaning |
| --- | --- |
| `initializer` | Initializer config name. |
| `eval_split` | The split every evaluation in this config used. |
| `dataset_realization_indices` | Configured dataset realizations. |
| `dataset_splits` | Splits generated per dataset realization: `train`, `val`, `test`. |
| `unmaskers` | Unmasker config name per dataset key. |
| `feature_costs` | Per dataset key, `{path, sha256}` of `extra/data/misc/feature_costs/<key>.csv` in the checkout; both null means unit feature costs. |
| `dataset_redistribution` | Per dataset key, the maintainers' review from the checkout: `status` (`unreviewed`, `permitted` or `restricted`), `license`, `source`, `reviewed_by`, `notes`. See [Dataset redistribution](#dataset-redistribution). |
| `classifiers` | One entry per classifier bundle the config trains: `bundle_path`, `script_name`, `script_params`, `dataset_key`, `method_name` (null for the external classifier shared by every method on a dataset; set for a classifier trained for one method by `method_options.<method>.classifier`), `dataset_realization_index` and `seed` (always instance 0, seed 0). |
| `eval_batch_sizes` | Evaluation batch size per method and dataset key. It can affect results (see ADR 0002). |
| `forcing_policy` | `forced_acquisition_when_eval_hard_budget_is_set`: hard-budget evaluation uses forced acquisition; soft-budget evaluation lets the policy stop. |

### `evaluation_tables`

One entry per evaluation the config schedules, present or not, enumerated
from the resolved config the same way the workflow names its targets. The
entry carries the identity that transformed tables written before the
identity columns existed do not hold in their columns, notably
`dataset_realization_index` and `eval_split`.

| Field | Meaning |
| --- | --- |
| `raw_path`, `transformed_path` | The raw and plotting-ready Parquet tables. |
| `raw_present`, `transformed_present` | Whether each file is in the snapshot. |
| `raw_size_bytes`, `transformed_size_bytes` | File sizes; null when absent. |
| `method_name`, `dataset_key`, `dataset_realization_index`, `eval_split`, `initializer`, `unmasker` | Identity of the evaluation. |
| `dataset_generation_seed` | Seed of the dataset realization (its index). |
| `budget_setting` | `hard_budget` or `soft_budget` (no evaluation hard budget). |
| `pretrained_model_name`, `pretrain_seed` | Null for methods without a pretraining stage. |
| `train_seed`, `train_hard_budget`, `train_soft_budget_param` | Training run of the evaluated method bundle. |
| `eval_seed`, `eval_hard_budget`, `eval_soft_budget_param` | Evaluation settings. |
| `forced_acquisition` | Whether the evaluation forced acquisition. |
| `classifier_bundle_path` | The classifier bundle in `settings.classifiers` that produced the `external` predictions. |
| `eval_batch_size` | Evaluation batch size. |
| `classifier_variants` | Which of `builtin` and `external` predictions the raw table holds (non-null prediction column); null if the raw table is absent. |
| `inputs` | The bundles the evaluation loaded, as `{role, path}`: `eval_dataset`, `method` and `classifier`. |

### `bundles`

One entry per native bundle the config schedules, present or not,
enumerated the way the workflow names its `all_generate_datasets`,
`all_train_classifiers`, `all_pretrain_models` and `all_train_methods`
targets. A trained method bundle shared by several evaluations is listed
once.

| Field | Meaning |
| --- | --- |
| `path` | The bundle folder. |
| `category` | `dataset_bundle`, `classifier_bundle`, `pretrained_model_bundle` or `afa_method_bundle`. |
| `present`, `size_bytes` | Whether the bundle is in the snapshot, and the bytes of all files in it (null when absent). |
| `dataset_key`, `dataset_realization_index` | The dataset realization the bundle was generated or trained from. Classifiers are always trained on dataset realization 0. |
| `split` | `train`, `val` or `test` for a dataset bundle; null otherwise. |
| `method_name` | The method an AFA-method bundle or a method's own classifier belongs to; null for shared prerequisites. |
| `pretrained_model_name` | The pretrained model of a pretrained-model bundle, or the one an AFA-method bundle was trained from. |
| `seed` | Dataset generation, classifier (always 0), pretraining or training seed. |
| `train_hard_budget`, `train_soft_budget_param` | Training budget of an AFA-method bundle; null otherwise. |
| `inputs` | The bundles the producing job read, as `{role, path}` with roles `train_dataset`, `val_dataset`, `classifier` and `pretrained_model`. Dataset bundles have none. |
| `bundle_manifest` | The bundle's own `manifest.json`, verbatim; null when absent. Its `metadata` holds the dataset generation parameters, or the training contract (initializer, Unmasker, seed, budgets, smoke flag, input paths) and method configuration; its `provenance` is the bundle's provenance record and `content_hash` its data hash ([bundle format](bundle_format.md)), both absent from bundles written before ADR 0002. |

### `coverage`

Computed from the evaluation tables actually present (raw or transformed),
not from the config, so a partial run is not described as complete:
`datasets`, `dataset_realization_indices`, `methods`, `eval_splits`,
`budget_settings`, `classifier_variants`, and `output_categories` (top-level
directories of the output root that contain files, such as `datasets`,
`trained_methods`, `eval_results`, `eval_results_transformed`,
`merged_results` and `plot_results`).

`payloads` has one entry per [payload category](#payload-categories), in
the order of that table: `category`, `scheduled` (entries the config
schedules), `present`, `size_bytes` (of the present ones) and `class_names`
(the bundle classes present, which name the registered loaders a restore
needs; empty for tables).

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
| `classifier_bundle` | `trained_classifiers/` | External-classifier predictions for any method's evaluation (`dataset-<key>.bundle`), without retraining it. `method-<name>+dataset-<key>.bundle` is a classifier one method trains with and is needed only to retrain that method. | Shared prerequisite (external); method-specific otherwise |
| `pretrained_model_bundle` | `pretrained_models/` | Training any method that names the same pretrained model, without repeating pretraining. | Shared prerequisite |
| `afa_method_bundle` | `trained_methods/` | Re-evaluating a published method (other seeds, budgets, splits) or inspecting its policy, without retraining it. | Optional baseline |

A bundle is a shared prerequisite when its `method_name` is null. Plotting a
new method against published baselines needs only the evaluation tables of
those baselines; evaluating a new method needs the dataset bundles and the
external classifier of its datasets, and the pretrained model it trains
from, if the release has it. AFA-method bundles are never needed for
comparison plots. Follow `inputs` to find what a bundle or table was
produced from.

Pretrained-model and AFA-method bundles are restored together with the
`pretrain_time.txt` or `train_time.txt` record their job wrote in the same
folder. The workflow's time aggregation reads the record, so without it
the workflow would rerun the job.

Bundles are the inference format the evaluator loads. Intermediate training
checkpoints, such as the Lightning checkpoints of classifier training, are
written under `extra/logs/`, outside the output root, so they are never in
a snapshot: they serve resuming or debugging one training run, are not
loadable by `load_bundle`, and are not a reusable payload.

## Dataset redistribution

Being able to generate a dataset bundle grants no right to publish it. Each
dataset key is recorded in `settings.dataset_redistribution` from the
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

| Category | Present | Bytes | Classes |
| --- | --- | --- | --- |
| `raw_evaluation_table` | 3 | 20,343 | |
| `transformed_evaluation_table` | 3 | 29,767 | |
| `dataset_bundle` | 3 | 119,065 | `CubeDataset` |
| `classifier_bundle` | 1 | 94,578 | `WrappedMaskedMLPClassifier` |
| `pretrained_model_bundle` | 1 | 95,267 | `GreedyAFAClassifier` |
| `afa_method_bundle` | 3 | 199,310 | `GDFSAFAMethod`, `RandomWithoutClassifierAFAMethod` |

The rest of that output root was `plot_results/` (5.1 MB, 134 files),
`merged_results/` (83 kB), `combined_time_results/` (10 kB) and
`eval_time_results/` (30 B). A production run multiplies the counts by
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
  record in the Arrow schema metadata. Tables written before the identity
  columns existed have only `eval_seed` and `eval_hard_budget` of them. They
  hold the full acquisition history; selection histories can be
  reconstructed from `episode_id`, `step` and `action_performed`.
- **Plotting-ready tables** (`eval_results_transformed/.../eval_data.parquet`,
  `transformed_path`) are produced by
  `scripts/misc/transform_eval_data_pipeline.py`. Each raw row becomes two
  rows, one per `classifier` (`builtin` or `external`) with its
  `predicted_class`; `episode_id` and `step` are dropped in favour of
  `n_selections_performed`, so these are prediction/cost rows, not episode
  logs. They keep `generation_index` and `split_index` (null for tables
  transformed from legacy logs), every identity column and the raw table's
  provenance record. A table transformed from a raw table written before the
  identity columns existed has null `dataset_realization_index` and
  `eval_split`: read those from the table's `evaluation_tables` entry.
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
