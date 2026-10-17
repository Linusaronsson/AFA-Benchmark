# Compare your method with published results

For a [repository adopter](../explanation/user_types.md#repository-adopter):
run your method next to the methods of a benchmark release, without
training and evaluating the published methods again.

You download the published methods into your output root and run your own
pipeline profile, which lists them next to your method. The downloaded
release is a cache
([ADR 0012](../adr/0012-a-downloaded-release-is-a-cache.md)): the run
produces exactly what your profile produces without it. It reuses what the
release holds and produces the rest: your method, and anything your
profile asks for that the release lacks, such as a dataset the release does
not cover, for every method.

Only the `download` step needs network access. Everything else reads and
writes local files and needs no Hugging Face account.

## 1. Add your method and a pipeline profile

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

and a `soft_budget_params` entry, as for any method.

To make the method appear in plots, add it to
`conf/scripts/plotting/common/default.yaml`:

- `method_name_mapping`: its display name;
- `method_policy_family_mapping`: its colour family, an existing one or a
  new one added to every entry of `method_family_color_schemes`.

Then give the comparison its own
[pipeline profile](run_the_pipeline.md#pipeline-profile). Start from the
one the release was produced with, so the published methods are asked for
with the settings the release has them at. `download` prints it as
`Workflow profile`; here it is `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 workflow/profiles/pipeline/my_comparison
```

In `my_comparison/config.yaml`, replace the `methods` and `method_sets`
files with your own, `workflow/conf/methods/my_comparison.yaml`:

```yaml
methods: [random_dummy, gdfs, gdfs_low_lr]
```

and `workflow/conf/method_sets/my_comparison.yaml`:

```yaml
method_sets:
  my_comparison: [random_dummy, gdfs, gdfs_low_lr]
```

- `methods`: the published methods you compare with, and yours.
- `method_sets`: the groups plotted together. The `all` method set, of
  every method in `methods`, is always plotted.

To compare on a dataset the release lacks, add it to the profile's
`datasets` as well.

## 2. Download the published methods

Choose a release id from the [published releases](download_published_results.md#1-choose-a-release),
then download into the fork's `output/production`:

```shell
uv run python scripts/release/snapshot.py download <release_id> --repo-id <repo_id> \
    --method random_dummy --method gdfs
```

This downloads the named release's evaluations of those methods with
their whole chain: the dataset bundles and external classifiers they used,
their pretrained models, their AFA-method bundles, and their raw and
plotting-ready evaluation tables. A run reuses a published method's
evaluations only with all of these; without its AFA-method bundles, it
trains the method again. To download less, narrow the evaluations with
coverage options, for example `--dataset cube`.

The command prints what the release lacks of your selection. All options:
[`download`](../reference/snapshot_command.md#download).

## 3. Check the release's provenance

`download` restores the release's manifest to
`output/production/release_manifest.json`. Each of its `evaluations` and `bundles`
entries has the `code.commit` that produced it; list them with

```shell
jq -r '[.evaluations[], .bundles[]] | .[].code.commit' \
    output/production/release_manifest.json | sort -u
```

Compare those commits with your fork's history, and read the [release notes](../reference/release_notes.md) for
changes since that commit. If a change affects results (data, splits,
preprocessing, classifiers, acquisition semantics or metrics), your
results are not comparable with the published ones, even though every
file still loads ([why](../explanation/benchmark_releases.md#compatibility-and-comparability)).
A restored output is not run again when your fork changes its rule's code
or parameters
([why](../explanation/output_snapshots.md)).

## 4. Regenerate omitted dataset bundles when needed

If the release omitted dataset bundles for local regeneration or
redistribution reasons, prepare any required local sources following the
[dataset guide](../reference/datasets/README.md),
then generate the dataset realizations selected by your profile:

```shell
workflow/run.py my_comparison --target all_generate_datasets
```

On a cluster, add `--site <site>`. Keep the release's dataset generation
settings and environment to reproduce its realization. Generating bundles
locally does not grant permission to redistribute them.

For subsequent invocations, enable checked reuse and turn dataset generation
off so that the checked bundles cannot be replaced during the run:

```shell
workflow/run.py my_comparison reuse_release=true \
    "stages={dataset_generation: false}" -n
```

This requires the downloaded `release_manifest.json`. Reuse verifies the
actual dataset bundle contents against the input hashes recorded for existing
released classifiers, pretrained models, methods and evaluations on your
selected dataset realizations. Identical bundles do not invalidate those
outputs just because they are newer. Unknown or different hashes stop the
plan with an incompatibility error; reproduce the release's dataset settings
or use a separate output root to train on changed data. Missing models and
new methods still run normally.

Hashes identify the bundle's saved data, not external source files. For image
datasets that save indices and configuration instead of pixels, ensure your
local source files match the release's sources too. Reuse does not establish
code or hyperparameter compatibility; still perform step 3.

## 5. Run your pipeline profile

Check the plan first; on a cluster, add `--site <site>`. If you regenerated
omitted bundles, keep `reuse_release=true "stages={dataset_generation: false}"`
on the commands in this step, including the real run:

```shell
workflow/run.py my_comparison -n
```

The plan should list only:

- your method's `train_method`, `eval_method` and `transform_eval_data`
  jobs, and its `pretrain_model` if the release lacks its pretrained model;
- every job of what the release lacks, for example for a new dataset its
  `dataset_generation`, `train_classifier` and `pretrain_model` jobs, and
  the training and evaluation of every method on it;
- the comparison's `merge_eval_perf`, `split_by_classifier_type`,
  `plot_eval_perf`, `collect_job_records` and `plot_time`.

If it lists a published method's training or evaluation on a dataset the
release has, your profile asks for something `output/production` does not
hold: a budget, dataset realization, evaluation split or initializer the
release lacks, or an evaluation you did not download. Download it, or
change the profile to match the release, whose configuration is in the
manifest's `workflow_config`.

If it lists `dataset_generation` or `train_classifier` for a dataset the
release has, a [shared prerequisite](../../CONTEXT.md) is missing or newer
than a released output. Download available prerequisites (step 2). For
omitted dataset bundles, follow step 4 instead. A genuinely missing classifier
must be trained and can cause evaluations to run again.

Then run it without `-n`. The plots end up in:

```text
output/production/plots/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-my_comparison+classifier_type-<builtin|external>/dataset_set-<name>/
        hard_budget_normal.{pdf,svg}   hard_budget_traj.{pdf,svg}
        soft_budget_lines.{pdf,svg}    soft_budget_2d_errors.{pdf,svg}
```

and the tables they plot in
`output/production/merged_results/eval_split-<split>/initializer-<init>/eval_perf/method_set-my_comparison+*.parquet`.

## Download more later

`download` writes nothing if a file it would write already exists, and
lists the conflicts. A second download into the same output root, for
example of more methods, needs `--overwrite`, if only for
`output/production/release_manifest.json`.

Use one release per output root, since the restored manifest describes one
release and outputs of two releases would be mixed. To compare with
another release, download it into another output root and run your
profile on that root:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id <repo_id> --destination-root output/<name> \
    --method random_dummy --method gdfs
workflow/run.py my_comparison output_root=output/<name> -n
```
