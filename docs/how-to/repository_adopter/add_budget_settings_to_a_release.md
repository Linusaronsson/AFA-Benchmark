# Add hard budgets or soft-budget parameters to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
keep a release's datasets, methods and seeded repetitions, but evaluate
additional hard budgets or soft-budget parameters. Reuse released training
outputs where the requested evaluation uses the same trained method.

Run commands from your fork's repository root, after `uv sync`. This guide
extends your local results; it does not modify the published release.

## 1. Download the release and match its configuration

Choose an explicit release id from the
[published releases](../results_only_user/download_published_results.md#1-choose-a-release), read
its README for coverage and caveats, and download into a fresh output root:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_budgets
```

Download the full chain, not just evaluation tables: the pipeline needs
AFA-method bundles and shared prerequisites to reuse released evaluations.
Use one release per output root.

Inspect the downloaded manifest:

```shell
jq '{coverage, workflow_config}' output/more_budgets/release_manifest.json
jq -r '[.evaluations[], .bundles[]] | .[].code.commit' \
    output/more_budgets/release_manifest.json | sort -u
```

Copy the pipeline profile printed as `Workflow profile`; here it is `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 \
    workflow/profiles/pipeline/more_budgets
```

Match the copied configuration to `workflow_config.merged` and `overrides`,
keeping the release's datasets, methods, dataset realization indices,
initializer, evaluation split, classifiers, Unmaskers and method settings.
Check the actual `evaluations`, not just coverage lists: the release may
lack some combinations. Downloading it does not apply its configuration.

Check producing commits against your fork for result-affecting differences.
Use corresponding implementations and environments for comparable results;
restored outputs are not automatically rerun when code or parameters change.
See [compatibility and comparability](../../explanation/benchmark_releases.md#compatibility-and-comparability).

## 2. Copy the configuration groups you will change

For hard-budget changes, copy the profile's `eval_hard_budgets` file; for
soft-budget changes, copy its `soft_budget_params` file. For example:

```shell
cp workflow/conf/eval_hard_budgets/kdd26.yaml \
    workflow/conf/eval_hard_budgets/more_budgets.yaml
cp workflow/conf/soft_budget_params/kdd26.yaml \
    workflow/conf/soft_budget_params/more_budgets.yaml
```

In `workflow/profiles/pipeline/more_budgets/config.yaml`, replace the
corresponding configfile paths with these copies. Copy and repoint
`method_options` too if you need to change budget mappings or soft-budget
opt-ins. Leave unrelated groups unchanged.

Follow step 3, step 4, or both. **Retain the existing list entries and append
new ones** if you want plots containing both published and new settings.
Lists replace rather than extend earlier lists; mapping overrides merge key
by key. Name the dataset explicitly: changing `default` does not affect a
dataset with its own entry.

## 3. Append hard budgets and check the training-budget mapping

In your copied hard-budget file, extend the dataset's list. For example,
if the release has CUBE at `[3, 5, 10]`, add budgets 4 and 7:

```yaml
eval_hard_budgets:
  # Keep the other datasets and default entry.
  cube: [3, 4, 5, 7, 10]
```

Choose budgets in accumulated selection cost, not feature count. Check the
Unmasker and feature costs, especially for images or non-unit-cost datasets.

For each method, inspect its `method_options`:

- Without a mapping, each evaluation hard budget is also its training hard
  budget. New budgets therefore require new training.
- `train_hard_budget` fixes one training budget per dataset. If the release
  includes that trained bundle, new evaluation budgets reuse it.
- `eval_to_train_hard_budget_mapping` must map **every** requested evaluation
  budget. Preserve existing mappings and add entries for the new budgets.
  Reuse is possible when they map to a training budget already downloaded.
- `hard_budget_ignored_datasets` excludes datasets from this setting.

Do not change training budgets just to avoid training: the mapping is part
of the experimental configuration, and evaluating a method at a budget it
was not designed for may not be meaningful.

Also inspect `use_max_hard_budget_when_training_soft_budget`. For these
methods, increasing the largest **mapped training hard budget** changes the
training identity of soft-budget runs too. They may need retraining and
reevaluation even if you changed only the hard-budget list. To extend only
hard-budget results, use the launcher's `--no-soft-budget`; that invocation
will not include soft-budget results in its plots.

## 4. Append soft-budget parameter pairs

In your copied soft-budget file, extend the relevant method/dataset lists.
Each entry is `[train, eval]`; follow the method's existing convention:

- `[value, null]`: choose the parameter during training and keep it during
  evaluation. A new training value requires a new method bundle.
- `[null, value]`: train without a soft-budget parameter and set it during
  evaluation. For methods supporting this convention, new evaluation values
  can reuse the same downloaded bundle.

For example, if these are the released lists, append `0.02` for JAFA and
`0.02` for DIME:

```yaml
soft_budget_params:
  # Keep the other methods and dataset entries.
  jafa:
    cube: [[0.005, null], [0.0075, null], [0.01, null], [0.02, null]]
  dime:
    cube: [[null, 0.001], [null, 0.005], [null, 0.01], [null, 0.02]]
```

Use the actual release's lists rather than assuming these example values.
Do not switch a training-time parameter to an evaluation-time parameter to
avoid training. The method must support setting it at evaluation. Keep at
most one non-null value per pair: the plotting pipeline rejects rows with
both training and evaluation parameters set.

Ensure the dataset is listed in each method's `soft_budget_datasets` and
that `soft_budget_setting` is enabled. A parameter list alone does not opt a
dataset in. Methods using `use_max_hard_budget_when_training_soft_budget`
also need the intended hard-budget list to retain their training identity.

## 5. Regenerate omitted dataset bundles when necessary

If the release omitted dataset bundles, prepare required local sources using
the [dataset reference](../../reference/datasets/README.md), then generate the
profile's selected realizations separately:

```shell
workflow/run.py more_budgets --site <your_site> \
    output_root=output/more_budgets --target all_generate_datasets
```

Reproduce the release's generation settings and environment. On subsequent
invocations, add `reuse_release=true "stages={dataset_generation: false}"`.
Checked reuse validates actual dataset bundle hashes against released
consumers' recorded inputs, avoiding reruns caused only by regenerated
bundles' newer timestamps. A mismatch or unknown hash stops planning; fix
the data rather than mixing incompatible outputs. External source files,
code and hyperparameters still need your own compatibility checks.

Skip this step if all needed dataset bundles were downloaded. See
[checked reuse](../../reference/pipeline_configuration.md#reuse_release).

## 6. Inspect the plan and run the extension

```shell
workflow/run.py more_budgets --site <your_site> \
    output_root=output/more_budgets -n
```

If you followed step 5, retain its checked-reuse overrides on this command
and the real run. On a workstation without GPUs, replace
`--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.

Expect new evaluations and transformations, plus training wherever a new
training identity is needed. Existing identical evaluations should be
reused; aggregation, visualization and job-record collection incorporate
the extension. Unexpected old training or evaluation jobs can indicate
missing downloads, changed budget mappings or a changed soft-budget training
hard budget. Partial releases can require additional jobs too.

Keep training enabled unless you have verified that **all** required trained
bundles exist. Remove `-n` to execute, retaining all other options. Repeat the
same command to resume.

## 7. Inspect the extended comparisons

Read `output/more_budgets/eval_results_transformed/` for per-evaluation
tables, and `output/more_budgets/merged_results/` and
`output/more_budgets/plots/` for combined comparisons. Verify
`train_hard_budget`, `eval_hard_budget`, `train_soft_budget_param` and
`eval_soft_budget_param` in the tables; hard- and soft-budget figures remain
separate. See [table columns](../../reference/evaluation_dataframes.md) and
[plot layout](../../reference/evaluation_plots.md#output-layout).

Keep the modified profile and invocation with your results. The downloaded
`release_manifest.json` describes the original release, not your local
extension.
