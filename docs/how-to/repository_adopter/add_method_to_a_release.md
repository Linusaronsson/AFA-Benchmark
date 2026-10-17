# Add a method to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add your method to a release's comparison while reusing its existing outputs.

## 1. Download the release and its config

[Choose a release](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and download it into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/new_method
```

Check the release README for required
[local dataset sources](../../reference/datasets/README.md) and
[code compatibility](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before reusing its outputs.

## 2. Add your method to the downloaded config

Follow [add a method](add_method.md) to implement and register your method.
Use a name not already in the release, and apply its pipeline configuration
steps to `output/new_method/benchmark.yaml` instead of a pipeline profile:

- Append your method to `methods`.
- Add its `method_options` and `soft_budget_params` entries, including an
  empty parameter list for a hard-budget-only method.
- Add any required `pretrain_mapping` entry.
- Optionally add the method to named `method_sets`; the automatic `all` set
  includes it already.

Leave the released methods' settings, datasets, budgets and realization
indices unchanged.
[Configuration keys](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 3. Preview and run the extended comparison

Preview the work:

```shell
workflow/run.py --configfile output/new_method/benchmark.yaml \
    --site <your_site> -n
```

Expect jobs for your method and missing release coverage. Investigate any
unexpected training or evaluation of published methods before proceeding;
check for missing downloads or changed settings. To tell the jobs that run
from the published outputs that are reused, and read the reason for each,
replace `-n` with `--interactive-preview`
([interactive preview](../../reference/interactive_preview.md)).

Expect `Conditional reuse` for published outputs on datasets the release
omits. If the run stops because a regenerated dataset differs from the
release, fix the data or recompute in a separate output root; do not bypass
the check.

Run the same configuration without `-n`:

```shell
workflow/run.py --configfile output/new_method/benchmark.yaml \
    --site <your_site>
```

Find comparison tables in `output/new_method/merged_results/` and
figures in `output/new_method/plots/`.
