# Add more seeds to a release's datasets and methods

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add [dataset realizations](../../reference/pipeline_configuration.md#dataset_realization_indices)
to a release's comparison. To vary seeds on a fixed split instead, do not use
this procedure.

Run commands from your fork's repository root, after `uv sync`.

## 1. Download the release

Choose a [release id](../results_only_user/download_published_results.md#1-choose-a-release),
read its README, and download into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/more_seeds
```

## 2. Match its configuration and code

Inspect the configuration and producing commits:

```shell
jq '{coverage, workflow_config}' output/more_seeds/release_manifest.json
jq -r '[.evaluations[], .bundles[]] | .[].code.commit' \
    output/more_seeds/release_manifest.json | sort -u
```

Copy the profile printed as `Workflow profile`; here, `kdd26`:

```shell
cp -r workflow/profiles/pipeline/kdd26 workflow/profiles/pipeline/more_seeds
```

Match its config files to `workflow_config.merged` and `overrides`. Keep all
settings except the realization list unchanged. Copy any files you edit to
`workflow/conf/<group>/more_seeds.yaml` and repoint the profile's configfile paths.
Check `evaluations` for combinations missing from the release.

Use code and an environment matching the released results. Check
[result-affecting differences](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before combining old and new results.

## 3. Extend the realization list

Read the existing indices:

```shell
jq '.coverage.dataset_realization_indices' output/more_seeds/release_manifest.json
```

Keep them and append unused non-negative indices. For a release with
`[0,1,2,3,4]`, add five realizations with:

```shell
workflow/run.py more_seeds --site <your_site> \
    output_root=output/more_seeds \
    "dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]" -n
```

Use your release's actual indices. Keep the complete list on every command
below to include both old and new results.

Without GPUs, replace `--site <your_site>` with
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml` throughout.

## 4. Generate omitted dataset bundles, if needed

Skip this step if all required released dataset bundles were downloaded.
Otherwise, prepare the [local sources](../../reference/datasets/README.md)
and generate the selected realizations using the release's settings and environment:

```shell
workflow/run.py more_seeds --site <your_site> \
    output_root=output/more_seeds \
    "dataset_realization_indices=[0,1,2,3,4,5,6,7,8,9]" \
    --target all_generate_datasets
```

After generation succeeds, add these overrides to every subsequent dry run
and real run:

```shell
reuse_release=true "stages={dataset_generation: false}"
```

If the [reuse check](../../reference/pipeline_configuration.md#reuse_release)
fails, reproduce the released data or recompute in a separate root. Do not
bypass the check. Verify external source files match the release too.

## 5. Check the plan and run

Repeat step 3's dry run with step 4's overrides when applicable. Check that:

- New indices have generation, training and evaluation jobs as needed.
- Existing released evaluations are reused. Investigate old-index jobs for
  missing downloads, configuration differences or incomplete release coverage.
- The selected indices include both old and new realizations.

Leave training and evaluation enabled. Remove `-n` to run, retaining every
override. Repeat the same command to resume.

## 6. Check the combined results

Open `output/more_seeds/merged_results/` and `output/more_seeds/plots/`.
Check `dataset_realization_index`, `train_seed` and `eval_seed` in
`output/more_seeds/eval_results_transformed/` for both old and new repetitions.
[Table columns](../../reference/evaluation_dataframes.md).

Save the profile and invocation with your results. Do not treat the downloaded
`release_manifest.json` as an inventory of the extended run.
