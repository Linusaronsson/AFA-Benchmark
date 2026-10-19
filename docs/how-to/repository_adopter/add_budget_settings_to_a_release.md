# Add hard budgets or soft-budget parameters to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
evaluate additional budgets with a release's datasets, methods and realizations
while reusing its existing outputs.

## 1. Download the release and its config

[Choose a release](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and download it into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_budgets
```

Check the release README for required
[local dataset sources](../../reference/datasets/README.md) and
[code compatibility](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before reusing its outputs.

Edit the budgets in `output/more_budgets/benchmark.yaml` in step 2, step 3,
or both. Leave the datasets, methods, realization indices and other settings
unchanged. Retain existing list entries to include published settings in the
plots, and edit explicit dataset entries, not just `default`.

## 2. Append hard budgets

Extend the dataset's list under `eval_hard_budgets`. For example, add 4 and 7
to CUBE's `[3, 5, 10]`:

```yaml
eval_hard_budgets:
  # Keep the other entries.
  cube: [3, 4, 5, 7, 10]
```

Choose budgets in selection cost, accounting for the Unmasker and feature costs.
Check each method's entry under `method_options`:

- With no training-budget mapping, allow new training for each new budget.
- With `train_hard_budget`, retain the released training budget to reuse its bundle.
- With `eval_to_train_hard_budget_mapping`, preserve existing entries and map
  every new evaluation budget to an appropriate training budget.
- Check `hard_budget_ignored_datasets` for exclusions.

Do not change training budgets solely to avoid training.
If `use_max_hard_budget_when_training_soft_budget` is enabled, check whether
raising the largest mapped training budget also requires new soft-budget
training. To run only hard-budget results, add `--no-soft-budget` to both
commands in step 4.
[Budget options](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 3. Append soft-budget parameter pairs

Keep the method's released convention under `soft_budget_params`:
`[value, null]` for training-time parameters, or `[null, value]` for
supported evaluation-time parameters. Allow new training for new training-time
values. Set at most one value per pair.

For example, append `0.02` to these released lists:

```yaml
soft_budget_params:
  # Keep the other entries.
  jafa:
    cube: [[0.005, null], [0.0075, null], [0.01, null], [0.02, null]]
  dime:
    cube: [[null, 0.001], [null, 0.005], [null, 0.01], [null, 0.02]]
```

Use your release's actual lists. Ensure the dataset is in each method's
`soft_budget_datasets` and set `soft_budget_setting: true`. Retain the
hard-budget list needed by methods using
`use_max_hard_budget_when_training_soft_budget`.

## 4. Preview and run

Preview the work ([interactive preview](../../reference/interactive_preview.md)):

```shell
uv run workflow/run.py --configfile output/more_budgets/benchmark.yaml \
    --site <your_site> -i
```

Without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml` in both
commands.

Expect `Run` jobs for the new evaluations and any required training, and
`Reuse` for published outputs. Investigate unexpected `Run` jobs of old
budgets for missing downloads, changed mappings or incomplete release
coverage, as in [Add a method to a
release](add_method_to_a_release.md#4-investigate-unexpected-retraining).

Expect `Conditional reuse` for published outputs on datasets the release
omits. If the run stops because a regenerated dataset differs from the
release, fix the data or recompute in a separate output root; do not bypass
the check.

Run the same configuration without `-i`; repeat the command to resume:

```shell
uv run workflow/run.py --configfile output/more_budgets/benchmark.yaml \
    --site <your_site>
```

## 5. Check the results

Open `output/more_budgets/merged_results/` and `output/more_budgets/plots/`.
Verify training and evaluation budget columns in
`output/more_budgets/eval_results_transformed/`.
[Table columns](../../reference/evaluation_dataframes.md).

Save your edited `benchmark.yaml` and invocation. Do not treat the downloaded
manifest as an inventory of your local extension.
