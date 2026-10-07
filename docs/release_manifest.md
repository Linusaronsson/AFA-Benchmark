# Release manifest

A **release manifest** is the JSON file `release_manifest.json` written beside
an output snapshot's `output/` tree. It identifies a benchmark release and
records what produced the tree, so the snapshot stays reviewable and its
tables stay interpretable without knowing the path layout. The schema is
`afabench.release.manifest.ReleaseManifest`, version 1; publishing (#38) and
selective download (#40) read it. Design background:
[`artifact_publishing.md`](artifact_publishing.md) and
[ADR 0002](adr/0002-provenance-recorded-in-artifacts.md).

## Writing and restoring

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir \
    --release-id 2026-10-kdd26 --scope full \
    --profile extra/workflow/profiles/config/kdd26
```

`--release-id` asks for a manifest; without it `save` and `restore` behave
exactly as in [`tutorials/output_snapshots.md`](tutorials/output_snapshots.md).
With it, `--scope` (`full`, `partial` or `test_only`) is required, and so is
the workflow configuration the run used, given the way Snakemake was given it:

- `--profile DIR`: the profile's `config.yaml` supplies `configfile` and
  `config`.
- `--configfile PATH` (repeatable): replaces the profile's `configfile`.
- `--config KEY=VALUE` (repeatable, values parsed as YAML): replaces the
  profile's `config`, as a CLI `--config` does for Snakemake.

Config files are merged recursively in order and the overrides merged over
them, as Snakemake does. The command prints the profile, config files and
overrides it recorded, the release identity and the commit. `--checkout`
(default: the working directory) names the git checkout whose commit is
recorded and whose feature-cost files are hashed.

Smoke outputs are never a release. If the merged config has
`smoke_test: true`, the manifest records `execution_mode: smoke`, and any
scope other than `test_only` is refused before anything is copied.

`restore` copies the manifest to `release_manifest.json` beside the
destination root (`extra/release_manifest.json` for the default
`extra/output`), mirroring the snapshot layout, and prints its identity.
Like every other restored file it is not overwritten without `--overwrite`.
A manifest whose `manifest_version` this checkout does not know is refused
before anything is restored. A snapshot without a manifest restores without
one.

## Fields

All paths inside the manifest are POSIX paths relative to the output root,
except config file and feature-cost paths, which are as given. JSON object
keys are strings. `null` means unknown or not applicable, never a default.

### Release

| Field | Meaning |
| --- | --- |
| `manifest_version` | Schema version, 1. Readers reject versions they do not know. |
| `release_id` | The identity given with `--release-id`. |
| `scope` | `full`, `partial` or `test_only`, as declared. Only `full` and `partial` are publishable; `test_only` is never promoted. |
| `execution_mode` | `smoke` or `production`, from the merged config's `smoke_test`. `smoke` implies `test_only`. |
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
| `dataset_instance_indices` | Configured dataset instances. |
| `dataset_splits` | Splits generated per dataset instance: `train`, `val`, `test`. |
| `unmaskers` | Unmasker config name per dataset key. |
| `feature_costs` | Per dataset key, `{path, sha256}` of `extra/data/misc/feature_costs/<key>.csv` in the checkout; both null means unit feature costs. |
| `classifiers` | One entry per classifier bundle the config trains: `bundle_path`, `script_name`, `script_params`, `dataset_key`, `method_name` (null for the external classifier shared by every method on a dataset; set for a classifier trained for one method by `method_options.<method>.classifier`), `dataset_instance_index` and `seed` (always instance 0, seed 0). |
| `eval_batch_sizes` | Evaluation batch size per method and dataset key. It can affect results (see ADR 0002). |
| `forcing_policy` | `forced_acquisition_when_eval_hard_budget_is_set`: hard-budget evaluation uses forced acquisition; soft-budget evaluation lets the policy stop. |

### `evaluation_tables`

One entry per evaluation the config schedules, present or not, enumerated
from the resolved config the same way the workflow names its targets. The
entry carries the identity that transformed tables do not hold in their
columns, notably `dataset_instance_index` and `eval_split`.

| Field | Meaning |
| --- | --- |
| `raw_path`, `transformed_path` | The raw and plotting-ready Parquet tables. |
| `raw_present`, `transformed_present` | Whether each file is in the snapshot. |
| `method_name`, `dataset_key`, `dataset_instance_index`, `eval_split`, `initializer`, `unmasker` | Identity of the evaluation. |
| `dataset_generation_seed` | Seed of the dataset instance (its index). |
| `budget_setting` | `hard_budget` or `soft_budget` (no evaluation hard budget). |
| `pretrained_model_name`, `pretrain_seed` | Null for methods without a pretraining stage. |
| `train_seed`, `train_hard_budget`, `train_soft_budget_param` | Training run of the evaluated method bundle. |
| `eval_seed`, `eval_hard_budget`, `eval_soft_budget_param` | Evaluation settings. |
| `forced_acquisition` | Whether the evaluation forced acquisition. |
| `classifier_bundle_path` | The classifier bundle in `settings.classifiers` that produced the `external` predictions. |
| `eval_batch_size` | Evaluation batch size. |
| `classifier_variants` | Which of `builtin` and `external` predictions the raw table holds (non-null prediction column); null if the raw table is absent. |

### `coverage`

Computed from the evaluation tables actually present (raw or transformed),
not from the config, so a partial run is not described as complete:
`datasets`, `dataset_instance_indices`, `methods`, `eval_splits`,
`budget_settings`, `classifier_variants`, and `output_categories` (top-level
directories of the output root that contain files, such as `datasets`,
`trained_methods`, `eval_results`, `eval_results_transformed`,
`merged_results` and `plot_results`).

## Raw versus plotting-ready tables

Both are Parquet and readable with any Parquet reader. They differ in row
granularity:

- **Raw evaluation tables** (`eval_results/.../eval_data.parquet`,
  `raw_path`) have one row per episode and time step, as described in
  [`evaluation_dataframes.md`](evaluation_dataframes.md): `episode_id`,
  `step`, `action_performed`, `builtin_predicted_class`,
  `external_predicted_class`, `true_class`, `accumulated_cost`, `forced_stop`,
  `eval_seed`, `eval_hard_budget`. They hold the full acquisition history;
  selection histories can be reconstructed from `episode_id`, `step` and
  `action_performed`.
- **Plotting-ready tables** (`eval_results_transformed/.../eval_data.parquet`,
  `transformed_path`) are produced by
  `scripts/misc/transform_eval_data_pipeline.py`. Each raw row becomes two
  rows, one per `classifier` (`builtin` or `external`) with its
  `predicted_class`; `episode_id` and `step` are dropped in favour of
  `n_selections_performed`, so these are prediction/cost rows, not episode
  logs. They gain `afa_method`, `dataset`, `initializer`, `train_seed`,
  `train_hard_budget`, `train_soft_budget_param` and `eval_soft_budget_param`
  columns, but **not** the dataset instance index or evaluation split: read
  those from the table's `evaluation_tables` entry.
- **Merged tables** (`merged_results/`) concatenate plotting-ready tables per
  method set and classifier type and carry no per-table identity beyond those
  columns.

Raw `episode_id` values are local to one evaluation table. They do not
identify the same instance across runs, so they do not support paired
instance-level comparisons between methods or releases.

## Limits of this version

- Per-table identity is enumerated from the workflow config, not read from
  the tables. Files under `eval_results/` that the recorded config would not
  produce (stale outputs, other configs) are copied but not listed. Once
  bundles and evaluation tables carry their own provenance record and
  identity columns (ADR 0002, #65/#66), the manifest should collect those
  instead, and the enumeration duplicated from `rules/helpers.smk` (pinned
  by `test/workflow/test_release_manifest_tables.py`) should go.
- The manifest records the producing commit; it does not check that the
  checkout restoring it is compatible.
- Bundle contents are not hashed.
