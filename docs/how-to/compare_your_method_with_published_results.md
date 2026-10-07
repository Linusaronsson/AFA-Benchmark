# Compare your method with published results

For a [repository adopter](../explanation/user_types.md#repository-adopter):
train and evaluate only your own method, and plot it next to the methods of
a benchmark release. The published methods are used as
[reference methods](../explanation/reference_methods.md): the workflow
reads their published tables and never retrains or re-evaluates them.

Only the `download` step needs network access. Everything else reads and
writes local files and needs no Hugging Face account.

## 1. Add your method

Fork the repository and add your method as described in
[add a method](add_method.md). Give it a method name that no published
method uses, since method names identify the rows of the comparison.

A variant of a published method, for example with other hyperparameters,
needs only a new method name in `method_options`:

```yaml
method_options:
  gdfs_low_lr:
    pretrained_model_name: "gdfs"   # reuse the published GDFS pretraining
    train_script_name: "gdfs"
    method_specific_params: ["lr=0.0005"]
    use_max_hard_budget_when_training_soft_budget: true
    eval_batch_size:
      default: 32
```

To make the method appear in plots, add it to
`extra/conf/scripts/plotting/common/default.yaml`:

- `method_name_mapping`: its display name;
- `method_policy_family_mapping`: its colour family, an existing one or a
  new one added to every entry of `method_family_color_schemes`.

## 2. Download the release

Download into the fork's `extra/output`:

```shell
uv run python scripts/release/snapshot.py download --repo-id <repo_id> \
    --payload-category transformed_evaluation_table \
    --payload-category dataset_bundle \
    --payload-category classifier_bundle \
    --payload-category pretrained_model_bundle \
    --dataset cube --method random_dummy --method gdfs
```

This downloads the latest full release. To use an older or a partial
release, name its release id after `download`.

`--dataset` and `--method` select the published evaluations to compare
with; `--method` names published methods, not yours. The payload
categories decide what of those evaluations is downloaded:

| Payload category | Needed? |
| --- | --- |
| `transformed_evaluation_table` | Required: the published methods' tables the comparison plots. |
| `dataset_bundle` | Required: your method uses exactly the published dataset realizations and splits. |
| `classifier_bundle` | Required: your method's external-classifier predictions come from the published external classifier. |
| `pretrained_model_bundle` | Required if your method uses a pretrained model the release has; saves repeating pretraining. |

The command prints what the release lacks of your selection. All options:
[`download`](../reference/snapshot_command.md#download).

## 3. Check the release's provenance

`download` prints the release's provenance and restores its manifest to
`extra/release_manifest.json`. Compare its `code.commit` with your fork's
history, and read the [release notes](../reference/release_notes.md) for
changes since that commit. If a change affects results (data, splits,
preprocessing, classifiers, acquisition semantics or metrics), your
results are not comparable with the published ones, even though every
file still loads ([why](../explanation/benchmark_releases.md#compatibility-and-comparability)).

## 4. Configure the comparison

Keep the configuration of each release you compare with in its own folder:

```shell
mkdir -p extra/workflow/conf/comparisons/<release_id>
jq .workflow_config.merged extra/release_manifest.json \
    > extra/workflow/conf/comparisons/<release_id>/release.json
```

`release.json` is the release's own workflow configuration, so the
workflow looks for the published tables at exactly the release's
evaluation split, initializer, dataset realizations and budgets. Remove
cluster-specific keys such as `execution_site_file` from it if you run
elsewhere.

Next to it, write `comparison.yaml`:

```yaml
methods: [gdfs_low_lr]
reference_methods: [random_dummy, gdfs]
method_sets:
  my_comparison: [random_dummy, gdfs, gdfs_low_lr]
method_options:
  gdfs_low_lr: {...}   # as in step 1, unless it is in your config files
soft_budget_params:
  gdfs_low_lr:
    cube: []
```

- `methods`: only the methods you train and evaluate yourself.
- `reference_methods`: the published methods to compare with.
- `method_sets`: the groups plotted together. Sets without one of your
  methods are skipped; the release has their plots.

`comparison.yaml` is given after `release.json`, so it replaces `methods`
and adds to `method_options` and `soft_budget_params`.

If you compare with a smoke release, label the plots: set `caption` in
`extra/conf/scripts/plotting/common/default.yaml`, for example to
`"Workflow demonstration from smoke runs, not scientific results"`.

## 5. Check the plan

```shell
uv run snakemake \
    --snakefile extra/workflow/snakefiles/orchestration/pipeline.smk \
    --configfile extra/workflow/conf/comparisons/<release_id>/release.json \
                 extra/workflow/conf/comparisons/<release_id>/comparison.yaml \
    --cores 4 --dry-run all
```

The plan should list only:

- your method's `train_method`, `eval_method` and `transform_eval_data`
  jobs and its time records;
- `pretrain_model`, only if you did not download its pretrained model;
- the comparison's `merge_eval_perf`, `split_by_classifier_type`,
  `plot_eval_perf`, `merge_time` and `plot_time`.

If it lists `dataset_generation` or `train_classifier`, a
[shared prerequisite](../../CONTEXT.md) is missing: download it (step 2) rather than letting the
workflow produce a different one.

If it fails with a `MissingInputException`, the named published table is
not in `extra/output`. Either you did not download it, or the release does
not have it: a dataset, dataset realization, budget or evaluation split your
configuration asks for but the release lacks. Download it, or remove that
setting from `comparison.yaml`.

## 6. Run the comparison

Run the same command without `--dry-run`. The plots end up in:

```text
extra/output/plot_results/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-my_comparison+classifier_type-<builtin|external>/<dataset set>/
        hard_budget_normal.{pdf,svg}   hard_budget_traj.{pdf,svg}
        soft_budget_lines.{pdf,svg}    soft_budget_2d_errors.{pdf,svg}
```

and the tables they plot in
`extra/output/merged_results/eval_split-<split>/initializer-<init>/eval_perf/method_set-my_comparison+*.parquet`.

## Download more later

`download` writes nothing if a file it would write already exists, and
lists the conflicts. A second download into the same fork, for example of
more datasets, needs `--overwrite`, if only for `extra/release_manifest.json`.

Use one release per output tree. To compare with another release, use a
separate clone, since the restored manifest describes one release and
outputs of two releases would be mixed.
