# Add hard budgets or soft-budget parameters to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
evaluate additional budgets with a release's datasets, methods and realizations.

Run commands from your fork's repository root, after `uv sync`.

## 1. Download the release and match its settings

Choose a [release id](../results_only_user/download_published_results.md#1-choose-a-release),
read its README, and download into a fresh root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_budgets
uv run python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("output/more_budgets/release_manifest.json").read_text())
print(json.dumps({key: manifest[key] for key in ("coverage", "workflow_config")}, indent=2))
commits = {entry["code"]["commit"] for entry in manifest["evaluations"] + manifest["bundles"]}
for commit in sorted(commits, key=lambda value: value or ""):
    print("null" if commit is None else commit)
PY
```

Copy the profile printed as `Workflow profile`; here, `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 workflow/profiles/pipeline/more_budgets
```

Match the config files to `workflow_config.merged` and `overrides`. Keep
non-budget settings unchanged and check `evaluations` for missing combinations.
Use corresponding code and environments; check
[comparability](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before combining results.

## 2. Copy the budget files you will edit

For hard budgets, soft-budget parameters, or both, copy the corresponding
files and repoint their configfile paths in `more_budgets/config.yaml`:

```shell
cp workflow/conf/eval_hard_budgets/kdd26.yaml \
    workflow/conf/eval_hard_budgets/more_budgets.yaml
cp workflow/conf/soft_budget_params/kdd26.yaml \
    workflow/conf/soft_budget_params/more_budgets.yaml
```

Copy and repoint `method_options` too if changing budget mappings or opt-ins.
Retain existing list entries to include published settings in the plots.
Edit explicit dataset entries, not just `default`. Complete step 3, step 4,
or both.

## 3. Append hard budgets

Extend the dataset's list. For example, add 4 and 7 to CUBE's `[3, 5, 10]`:

```yaml
eval_hard_budgets:
  # Keep the other entries.
  cube: [3, 4, 5, 7, 10]
```

Choose budgets in selection cost, accounting for the Unmasker and feature costs.
Check each method's `method_options`:

- With no training-budget mapping, allow new training for each new budget.
- With `train_hard_budget`, retain the released training budget to reuse its bundle.
- With `eval_to_train_hard_budget_mapping`, preserve existing entries and map
  every new evaluation budget to an appropriate training budget.
- Check `hard_budget_ignored_datasets` for exclusions.

Do not change training budgets solely to avoid training.
If `use_max_hard_budget_when_training_soft_budget` is enabled, check whether
raising the largest mapped training budget also requires new soft-budget
training. To run only hard-budget results, pass `--no-soft-budget`.
[Budget options](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 4. Append soft-budget parameter pairs

Keep the method's released convention: `[value, null]` for training-time
parameters, or `[null, value]` for supported evaluation-time parameters.
Allow new training for new training-time values. Set at most one value per pair.

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
`soft_budget_datasets` and `soft_budget_setting` is enabled. Retain the
hard-budget list needed by methods using
`use_max_hard_budget_when_training_soft_budget`.

## 5. Generate omitted dataset bundles, if needed

Skip this step if all needed dataset bundles were downloaded. Otherwise,
prepare the [local sources](../../reference/datasets/README.md) and generate
using the release's settings and environment:

```shell
workflow/run.py more_budgets --site <your_site> \
    output_root=output/more_budgets --target all_generate_datasets
```

After generation succeeds, add
`reuse_release=true "stages={dataset_generation: false}"` to all subsequent
commands. If the [reuse check](../../reference/pipeline_configuration.md#reuse_release)
fails, fix the data or recompute in a separate root; do not bypass it.
Verify external source files match too.

## 6. Check the plan and run

```shell
workflow/run.py more_budgets --site <your_site> \
    output_root=output/more_budgets -i
```

Retain step 5's overrides when applicable. Without GPUs, replace
`--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml` throughout.

In the [interactive preview](../../reference/interactive_preview.md), check
for `Run` jobs for the new evaluations and any required training. Investigate
unexpected `Run` jobs of old budgets for missing downloads, changed mappings
or incomplete release coverage, as in [Add a method to a
release](add_method_to_a_release.md#4-investigate-unexpected-retraining).
Keep training enabled unless every required trained bundle exists.

Remove `-i` to run, retaining all options. Repeat the command to resume.

## 7. Check the results

Open `output/more_budgets/merged_results/` and `output/more_budgets/plots/`.
Verify training and evaluation budget columns in
`output/more_budgets/eval_results_transformed/`.
[Table columns](../../reference/evaluation_dataframes.md).

Save the modified profile and invocation. Do not treat the downloaded manifest
as an inventory of your local extension.
