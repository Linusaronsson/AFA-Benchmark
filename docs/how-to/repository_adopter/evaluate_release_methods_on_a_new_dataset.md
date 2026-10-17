# Evaluate a release's methods on a new dataset

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add a dataset and train and evaluate every method of a chosen benchmark
release on it.

Run commands from your fork's repository root, with the environment installed
using `uv sync`. This guide runs only the new dataset; you do not need to
download the release's models or evaluation tables. Those models were fitted
on other datasets and do not replace training on yours.

## 1. Choose a release and inspect its configuration

Choose an explicit release id from the
[published releases](../results_only_user/download_published_results.md#1-choose-a-release), and
read its README for coverage and caveats. Download its manifest:

```shell
mkdir -p output/release_info
curl -fL \
    'https://huggingface.co/datasets/afabench/results/resolve/main/releases/<release_id>/release_manifest.json' \
    -o output/release_info/release_manifest.json
jq '{coverage, workflow_config}' output/release_info/release_manifest.json
```

Use `coverage.methods` as the list of published methods, and
`workflow_config` to identify the pipeline profile, configuration files and
overrides. Coverage lists do not guarantee every combination was evaluated;
inspect `evaluations` for exact identities
([manifest fields](../../reference/release_manifest.md#fields)).

Check the producing commits before choosing the code to run:

```shell
jq -r '[.evaluations[], .bundles[]] | .[].code.commit' \
    output/release_info/release_manifest.json | sort -u
```

Use a checkout with the corresponding method implementations and settings.
If your fork changes method behavior, report that difference rather than
presenting the run as the released configuration. The release can contain
more than one producing commit; its workflow configuration records the last
invocation, not every artifact's settings.

## 2. Add your dataset

Follow [add a dataset](add_dataset.md) to implement and register it, add its
Hydra generation config and dataset key, record its redistribution policy,
and supply a display name and dataset reference page. Below, its dataset key
is `my_dataset`.

Check the Unmasker, feature costs, external classifier and method dataset
overrides against your data. Defaults suit tabular datasets; image data
usually needs explicit choices. See
[pipeline configuration](../../reference/pipeline_configuration.md#unmaskers).

## 3. Create a profile containing the release's methods and your dataset

Copy the release's pipeline profile; this example assumes it is `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 \
    workflow/profiles/pipeline/my_dataset
```

Compare the copied configuration with the manifest's `workflow_config.merged`
and `overrides`. Restore the release's effective method options, pretraining
mappings, initializer, evaluation split and dataset realization indices where
they differ. Downloading a manifest does not apply its configuration to a run.

Create `workflow/conf/datasets/my_dataset.yaml`:

```yaml
datasets: [my_dataset]
```

Create `workflow/conf/methods/my_dataset.yaml` with **every** method in
`coverage.methods`, not just the subset you normally run. In
`workflow/profiles/pipeline/my_dataset/config.yaml`, replace the `datasets`
and `methods` configfile paths with these two files. Keep the release's
method options, pretrained-model mappings and method sets. The automatic
`all` method and dataset sets include the entire run.

For configuration groups you need to change, copy their files to
`workflow/conf/<group>/my_dataset.yaml` and update the profile's configfile
paths before editing. In those copies:

- Set meaningful `eval_hard_budgets.my_dataset` values, considering selection
  costs and the chosen Unmasker. Do not silently rely on the tabular default
  for image or non-unit-cost data.
- Check that no method excludes your dataset in
  `hard_budget_ignored_datasets` if every method must run in that setting.
- To run soft budgets, add `my_dataset` to each applicable method's
  `soft_budget_datasets` and choose its `soft_budget_params` pairs. Without
  this opt-in, the method runs only hard budgets on your dataset. Keep a
  `soft_budget_params` entry for every selected method even for a
  hard-budget-only run.
- Add any necessary dataset-specific classifier, Unmasker, batch-size or
  training-budget choices. Keep method hyperparameters at the release's
  values unless the new dataset requires an explicit dataset override.

See [method options and validation](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 4. Inspect the work and run it

Use a fresh output root so unrelated outputs cannot satisfy this run:

```shell
workflow/run.py my_dataset --site <your_site> \
    output_root=output/my_dataset -n
```

On a workstation without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.
See [execution choices](../maintainer/run_the_pipeline.md#execution-file) before running
GPU methods locally.

Verify that the resolved configuration lists only `my_dataset` and every
release method, with the intended budgets and realizations. Expect dataset
generation, classifier training, required pretraining, method training and
evaluation, then transformation, aggregation and visualization. Investigate
any method missing from the planned evaluations before submitting.

Remove `-n` to run. Repeat the same command to resume interrupted work.

## 5. Read the results

Plotting-ready evaluations are under
`output/my_dataset/eval_results_transformed/`; merged comparison tables are
under `output/my_dataset/merged_results/`. The all-method comparison is in:

```text
output/my_dataset/plots/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-all+classifier_type-<builtin|external>/dataset_set-all/
```

External and built-in classifier results are separate. See
[plot layout](../../reference/evaluation_plots.md#output-layout) and
[evaluation table columns](../../reference/evaluation_dataframes.md).
These are new results on your dataset, not results supplied by the release.

To include the release's original datasets and reuse their results in the
same run, follow the download, provenance and omitted-dataset reuse steps in
[compare your method with published results](compare_your_method_with_published_results.md),
but keep the release's methods and add your dataset instead of a new method.
