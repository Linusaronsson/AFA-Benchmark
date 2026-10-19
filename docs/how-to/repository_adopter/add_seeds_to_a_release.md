# Add more seeds to a release's datasets and methods

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add dataset realizations to a release's comparison while reusing its existing
outputs.

## 1. Download the release and its config

[Choose a release](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and download it into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_seeds
```

## 2. Extend the realization list

Open `output/more_seeds/benchmark.yaml`, find
`dataset_realization_indices`, and append unused non-negative indices.
Retain every existing index and leave the other benchmark settings unchanged.

Alternatively, leave the downloaded config unchanged and append
`"dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]"` to both commands below,
using your release's actual indices. Command-line overrides take precedence
over the config.

## 3. Preview and run the extended comparison

Preview the work ([interactive preview](../../reference/interactive_preview.md)):

```shell
uv run workflow/run.py --configfile output/more_seeds/benchmark.yaml \
    --site <your_site> -i
```

Expect `Run` jobs for the new realizations and missing release coverage, and
`Reuse` for published outputs. Investigate unexpected `Run` jobs as in
[Add a method to a
release](add_method_to_a_release.md#4-investigate-unexpected-retraining).

Expect `Conditional reuse` for published outputs on datasets the release
omits. If the run stops because a regenerated dataset differs from the
release, fix the data or recompute in a separate output root; do not bypass
the check.

Run the same configuration without `-i`:

```shell
uv run workflow/run.py --configfile output/more_seeds/benchmark.yaml \
    --site <your_site>
```
