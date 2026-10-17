# Add more seeds to a release's datasets and methods

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
keep a benchmark release's datasets and methods, compute additional seeded
repetitions, and plot the published and new results together.

The pipeline controls repetitions through
[`dataset_realization_indices`](../../reference/pipeline_configuration.md#dataset_realization_indices).
Each index seeds dataset generation and splitting, classifier training,
pretraining, method training and evaluation. Adding indices therefore adds
[dataset realizations](../../../CONTEXT.md#datasets), not just training seeds
on an unchanged split. This guide does not vary training or evaluation seeds
independently on a fixed realization.

Run commands from your fork's repository root, after `uv sync`.

## 1. Download the chosen release into its own output root

Choose an explicit release id from the
[published releases](../results_only_user/download_published_results.md#1-choose-a-release), and
read its README for coverage, exclusions and caveats. Download its full cache:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_seeds
```

Use a fresh root and one release per root. Downloading only evaluation tables
is not enough: reusing published evaluations in a pipeline run needs their
AFA-method bundles and shared prerequisites too. `--all` obtains everything
the release includes; it cannot restore omitted dataset bundles.

## 2. Match the release's configuration and producing code

Inspect its manifest:

```shell
jq '{coverage, workflow_config}' output/more_seeds/release_manifest.json
jq -r '[.evaluations[], .bundles[]] | .[].code.commit' \
    output/more_seeds/release_manifest.json | sort -u
```

Copy the pipeline profile printed by `download` as `Workflow profile`; this
example assumes `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 \
    workflow/profiles/pipeline/more_seeds
```

Compare its configuration files with `workflow_config.merged` and
`workflow_config.overrides`. Copy any files you need to edit to
`workflow/conf/<group>/more_seeds.yaml`, and update their configfile paths in
`workflow/profiles/pipeline/more_seeds/config.yaml`.

Keep the release's datasets, methods, method options, budgets, soft-budget
parameters, Unmaskers, classifiers, pretrained-model mappings, initializer
and evaluation split. Use `coverage.datasets` and `coverage.methods` to check
which datasets and methods actually have released evaluations. Coverage is
not a Cartesian-product guarantee: use `evaluations` to check individual
combinations. A profile asking for combinations the release lacks will
compute those too.

Check result-affecting differences between the producing commits and your
fork, including generation, preprocessing, splits, method implementations,
classifiers and metrics. Use the corresponding code and environment for the
new repetitions. A restored output is not invalidated by changes to rule
code or parameters, so a successful dry run does not prove comparability.
See [compatibility and comparability](../../explanation/benchmark_releases.md#compatibility-and-comparability).

## 3. Extend the realization list without dropping existing indices

Find the released indices:

```shell
jq '.coverage.dataset_realization_indices' \
    output/more_seeds/release_manifest.json
```

Choose unused non-negative indices and use the union of old and new indices.
For a release covering `[0,1,2,3,4]`, adding five repetitions gives:

```shell
workflow/run.py more_seeds --site <your_site> \
    output_root=output/more_seeds \
    "dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]" -n
```

Use the actual release indices, not this example blindly. Keeping the old
indices ensures the merged tables and plots include the published results;
selecting only `[5,6,7,8,9]` would plot only the new repetitions.

On a workstation without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`. Keep the same
profile, root, realization list and execution options in subsequent commands.

## 4. Generate missing dataset bundles before checked reuse

If the release omitted dataset bundles, first prepare any required local
sources according to the [dataset reference](../../reference/datasets/README.md).
Then generate the selected realizations in a separate invocation:

```shell
workflow/run.py more_seeds --site <your_site> \
    output_root=output/more_seeds \
    "dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]" \
    --target all_generate_datasets
```

This also generates the new realizations. Reproduce the release's generation
settings and environment for existing indices. For datasets backed by
external files, ensure those sources match too.

After generation succeeds, use checked reuse and disable dataset generation
on **both** the dry run and real run:

```shell
workflow/run.py more_seeds --site <your_site> \
    output_root=output/more_seeds \
    "dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]" \
    reuse_release=true "stages={dataset_generation: false}" -n
```

Checked reuse compares existing dataset bundles with released consumers'
recorded input hashes, so identical regenerated bundles do not retrigger
published training and evaluation merely because they are newer. A mismatch
or unknown hash stops planning: reproduce the released data, or use a
separate root to recompute on changed data. Do not bypass the check to mix
incompatible results. Hashes cover saved bundle data, not external image
pixels, code or hyperparameters.

If every needed released dataset bundle was downloaded, this extra step is
unnecessary: use the command from step 3, which generates only missing
realizations under normal dependency tracking.

## 5. Check the plan and compute the additional repetitions

Expect generation (unless completed in step 4), classifier training,
required pretraining, method training, evaluation and transformation for the
new indices. Existing released evaluations should be reused. Aggregation,
visualization and job-record collection run to include the new results.

If old indices have training or evaluation jobs, check for missing downloads
or configuration mismatches. A partial release can require extra jobs for
old indices too. A genuinely missing classifier must be trained and may
invalidate evaluations that depend on it; regenerated omitted datasets need
step 4's checked reuse instead.

Do not disable `training` or `evaluation`: the new repetitions require them.
Remove `-n` from the appropriate command to run; retain all overrides,
including checked reuse when used. Repeat that command to resume.

## 6. Inspect the combined results

The pipeline rebuilds merged tables and plots from every selected index.
Find them under `output/more_seeds/merged_results/` and
`output/more_seeds/plots/`; per-evaluation tables under
`eval_results_transformed/` retain `dataset_realization_index`, `train_seed`
and `eval_seed`. Check those columns to confirm both old and new repetitions
are present. See [evaluation dataframes](../../reference/evaluation_dataframes.md)
and [plot layout](../../reference/evaluation_plots.md#output-layout).

The downloaded `release_manifest.json` still describes the original release,
not the additional local outputs. Keep your profile and full invocation with
the new results so the extended comparison can be reproduced.
