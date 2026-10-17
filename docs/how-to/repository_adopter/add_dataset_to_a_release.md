# Add a dataset to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add your dataset to a release's comparison while reusing its existing outputs.

## 1. Download the release and its config

[Choose a release](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and download it into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/new_dataset
```

Check the release README for required
[local dataset sources](../../reference/datasets/README.md) and
[code compatibility](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before reusing its outputs.

## 2. Add your dataset to the downloaded config

Follow [add a dataset](add_dataset.md) to implement and register your dataset.
Apply its pipeline configuration steps to
`output/new_dataset/benchmark.yaml` instead of a pipeline profile:

- Append your dataset to `datasets`.
- Set its `eval_hard_budgets` in selection cost, and choose its classifier
  and Unmasker where the defaults do not fit.
- Check each method's hard-budget exclusions. For soft-budget runs, add the
  dataset to each applicable method's `soft_budget_datasets` and choose its
  `soft_budget_params` pairs.
- Optionally add the dataset to named `dataset_sets`; the automatic `all`
  set includes it already.

Leave the existing datasets' settings, methods and realization indices
unchanged. Preserve released method hyperparameters unless your dataset
requires an explicit override.
[Configuration keys](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 3. Preview and run the extended comparison

Preview the work:

```shell
workflow/run.py --configfile output/new_dataset/benchmark.yaml \
    --site <your_site> -n
```

Check that every intended release method evaluates your dataset at the chosen
budgets and realizations. Investigate missing evaluations or unexpected
training or evaluation on existing datasets before proceeding. To tell the
jobs that run from the published outputs that are reused, replace `-n` with
`--interactive-preview`
([interactive preview](../../reference/interactive_preview.md)).
Investigate unexpected `Run` jobs as in [Add a method to a
release](add_method_to_a_release.md#4-investigate-unexpected-retraining).

Expect `Conditional reuse` for published outputs on datasets the release
omits. If the run stops because a regenerated dataset differs from the
release, fix the data or recompute in a separate output root; do not bypass
the check.

Run the same configuration without `-n`:

```shell
workflow/run.py --configfile output/new_dataset/benchmark.yaml \
    --site <your_site>
```

Find comparison tables in `output/new_dataset/merged_results/` and
figures in `output/new_dataset/plots/`.
