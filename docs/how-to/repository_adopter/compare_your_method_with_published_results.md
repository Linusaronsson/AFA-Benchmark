# Compare your method with published results

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
run your method alongside a release's methods, reusing their published outputs.

Run commands from your fork's repository root, after `uv sync`.

## 1. Add your method and a pipeline profile

Follow [add a method](add_method.md) and choose a method name not used by the
release. For a hyperparameter variant, add a new `method_options` entry and
its `soft_budget_params` entry. For example:

```yaml
method_options:
  gdfs_low_lr:
    pretrained_model_name: "gdfs"
    train_script_name: "gdfs"
    method_specific_params: ["lr=0.0005"]
    use_max_hard_budget_when_training_soft_budget: true
    eval_batch_size:
      default: 32
```

Add its display name and colour family to `method_name_mapping` and
`method_policy_family_mapping` in `conf/scripts/plotting/common/default.yaml`.
For a new family, update every entry in `method_family_color_schemes` too.

Choose a [release id](../results_only_user/download_published_results.md#1-choose-a-release)
and copy its pipeline profile; here, `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 workflow/profiles/pipeline/my_comparison
```

Create `workflow/conf/methods/my_comparison.yaml`:

```yaml
methods: [random_dummy, gdfs, gdfs_low_lr]
```

Create `workflow/conf/method_sets/my_comparison.yaml`:

```yaml
method_sets:
  my_comparison: [random_dummy, gdfs, gdfs_low_lr]
```

Repoint the methods and method-sets configfile paths in
`workflow/profiles/pipeline/my_comparison/config.yaml`. Adapt the example
method lists to your comparison. Add any new dataset to the profile too.

## 2. Download the published methods

Use one release per output root. Download the selected methods with their
bundles and prerequisites; do not restrict this to evaluation tables:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --method random_dummy --method gdfs
```

Check the missing-coverage report. Optionally add `--dataset cube` to narrow
the download. [Download options](../../reference/snapshot_command.md#download).

## 3. Match the released settings and code

Inspect the restored manifest:

```shell
uv run python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("output/production/release_manifest.json").read_text())
print(json.dumps(manifest["workflow_config"], indent=2))
commits = {entry["code"]["commit"] for entry in manifest["evaluations"] + manifest["bundles"]}
for commit in sorted(commits, key=lambda value: value or ""):
    print("null" if commit is None else commit)
PY
```

Match the published methods' settings in your profile to `workflow_config`.
Check the release README and compare producing commits against your fork for
changes to data, splits, preprocessing, classifiers, acquisition or metrics.
Do not combine incompatible results.
[Comparability checks](../../explanation/benchmark_releases.md#compatibility-and-comparability).

## 4. Generate omitted dataset bundles, if needed

Skip this step if all required bundles were downloaded. Otherwise, prepare
[local sources](../../reference/datasets/README.md) and generate with the
release's settings and environment:

```shell
workflow/run.py my_comparison --target all_generate_datasets
```

On a cluster, add `--site <site>`. Once generation succeeds, add these
overrides to every subsequent dry run and real run:

```shell
reuse_release=true "stages={dataset_generation: false}"
```

If the [reuse check](../../reference/pipeline_configuration.md#reuse_release)
fails, reproduce the released data or recompute in another root. Do not
bypass it. Verify external source files match too; still perform step 3's
code and configuration checks.

## 5. Check the plan and run

```shell
workflow/run.py my_comparison -n
```

Retain step 4's overrides when applicable. Add `--site <site>` on a cluster;
without GPUs, use `--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.

Expect jobs for your method, any missing release coverage, and aggregation
and plotting. If published training or evaluations are unexpectedly scheduled:

- Check for missing downloads and differences in budgets, realizations,
  splits and initializer settings.
- Download available prerequisites; follow step 4 for omitted datasets.
- Allow genuinely missing classifiers to train and dependent evaluations to rerun.

Remove `-n` to run. Find the comparison plots in:

```text
output/production/plots/eval_split-<split>/initializer-<init>/eval_perf/
    method_set-my_comparison+classifier_type-<builtin|external>/dataset_set-<name>/
```

Find the comparison tables under
`output/production/merged_results/eval_split-<split>/initializer-<init>/eval_perf/`.

## Download more later

For another download into the same root, use `--overwrite` only after checking
the listed conflicts. To compare with a different release, use another root:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --destination-root output/<name> \
    --method random_dummy --method gdfs
workflow/run.py my_comparison output_root=output/<name> -n
```
