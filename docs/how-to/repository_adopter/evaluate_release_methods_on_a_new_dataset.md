# Evaluate a release's methods on a new dataset

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
train and evaluate every method of a chosen release on your dataset.

Run commands from your fork's repository root, after `uv sync`.

## 1. Inspect the release

Choose a [release id](../results_only_user/download_published_results.md#1-choose-a-release)
and read its README. Download the manifest; skip the models and result tables
for this new-dataset-only run:

```shell
mkdir -p output/release_info
curl -fL \
    'https://huggingface.co/datasets/afabench/results/resolve/main/releases/<release_id>/release_manifest.json' \
    -o output/release_info/release_manifest.json
uv run python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("output/release_info/release_manifest.json").read_text())
print(json.dumps({key: manifest[key] for key in ("coverage", "workflow_config")}, indent=2))
commits = {entry["code"]["commit"] for entry in manifest["evaluations"] + manifest["bundles"]}
for commit in sorted(commits, key=lambda value: value or ""):
    print("null" if commit is None else commit)
PY
```

Use `coverage.methods` for the method list and `workflow_config` for the
profile and settings. Match the released method implementations and report
any result-affecting changes in your fork.
[Manifest fields](../../reference/release_manifest.md#fields).

## 2. Add the dataset

Follow [add a dataset](add_dataset.md); below, use the key `my_dataset`.
Check the feature costs, Unmasker, external classifier and method dataset
overrides against your data.

## 3. Create a pipeline profile

Copy the release's profile; here, `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 workflow/profiles/pipeline/my_dataset
```

Match the method settings, pretraining mappings, initializer, evaluation split
and realization indices to `workflow_config.merged` and `overrides`.

Create `workflow/conf/datasets/my_dataset.yaml`:

```yaml
datasets: [my_dataset]
```

Create `workflow/conf/methods/my_dataset.yaml` listing every method in
`coverage.methods`. In `workflow/profiles/pipeline/my_dataset/config.yaml`,
replace the datasets and methods configfile paths with these files.

Copy any other configuration files you need to change to
`workflow/conf/<group>/my_dataset.yaml` and repoint their paths. Set:

- Explicit `eval_hard_budgets.my_dataset` values appropriate to selection costs.
- Hard-budget exclusions so all intended methods run.
- `soft_budget_datasets` and `soft_budget_params` for each method you want
  in the soft-budget setting. Keep a parameter entry for every method,
  including hard-budget-only methods.
- Dataset-specific classifier, Unmasker, batch-size and training-budget
  choices where needed. Preserve released hyperparameters unless the dataset
  requires an explicit override.

[Configuration keys](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 4. Check the plan and run

Use a fresh output root:

```shell
workflow/run.py my_dataset --site <your_site> \
    output_root=output/my_dataset -n
```

Without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.

Verify that the plan includes only your dataset, every release method, and
the intended budgets and realizations. Investigate missing method evaluations.
Remove `-n` to run; repeat the command to resume.

## 5. Read the results

Open `output/my_dataset/eval_results_transformed/` for evaluations and
`output/my_dataset/merged_results/` for comparison tables. Find all-method plots in:

```text
output/my_dataset/plots/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-all+classifier_type-<builtin|external>/dataset_set-all/
```

[Plot layout](../../reference/evaluation_plots.md#output-layout).

To include the release's original datasets too, follow the download and
reuse steps in [compare your method with published results](compare_your_method_with_published_results.md),
adding your dataset instead of a new method.
