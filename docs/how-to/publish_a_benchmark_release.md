# Publish a benchmark release

For a [maintainer](../explanation/user_types.md#maintainer): publish the
outputs of a benchmark run on Hugging Face as a benchmark release.
[What a release is and why it works this way](../explanation/benchmark_releases.md).

## 1. Set up the host (once)

Authenticate on the machine where you will create and publish the release
before creating the repository:

1. Sign in to Hugging Face and [create a user access token](https://huggingface.co/settings/tokens)
   with the `write` role. Treat the token as a secret; do not commit it.
2. Log in and paste the token at the prompt, then verify the active account:

   ```shell
   uv run hf auth login
   uv run hf auth whoami
   ```

   For non-interactive use, supply the token through the `HF_TOKEN`
   environment variable instead of running `hf auth login`. `HF_TOKEN`
   overrides a saved login; if authentication fails, check for an invalid
   or expired token in that variable as well.
3. Create a dataset repository, public for an announced release. Replace
   `<repo_id>` with `<username>/results` or `<organization>/results`;
   for an organization, your account must have permission to create and
   write repositories there:

   ```shell
   uv run hf repo create <repo_id> --repo-type dataset
   ```

   Organization membership does not select the repository's owner
   automatically. Using just `results` creates it under your personal
   account. To create it under the `AFABench` organization, explicitly use:

   ```shell
   uv run hf repo create AFABench/results --repo-type dataset
   ```

   Use the same full repository id when publishing:
   `--repo-id AFABench/results` (or set
   `AFABENCH_RELEASE_REPO=AFABench/results`).

A `401 Unauthorized` error means authentication failed: check
`uv run hf auth whoami` and log in with a valid token before retrying.
The uv hardlink warning is unrelated to Hugging Face authentication.

Instead of `--repo-id <repo_id>` below, the repository can be in
`AFABENCH_RELEASE_REPO`.

## 2. Review dataset redistribution

Only datasets whose bundles may be published publicly can be in a
release. For each dataset of the run, read its source's licence
or terms and record the review in
`conf/release/dataset_redistribution.yaml` (format in the file's
header; [what a review covers](../reference/release_manifest.md#dataset-redistribution)).
A dataset without an entry is `unreviewed`.

## 3. Save the release package

Run the benchmark ([run the pipeline](run_the_pipeline.md)).
Then save its outputs as an
[output snapshot](create_output_snapshots.md) with a release manifest, by
giving a release id, a scope and the workflow configuration whose targets
lay out the outputs, the same way it was given to Snakemake. For a release
built across several runs, that is the configuration of the final run:

```shell
uv run python scripts/release/snapshot.py save /path/to/2026-10-kdd26 \
    --release-id 2026-10-kdd26 --scope full \
    --profile workflow/profiles/pipeline/kdd26
```

This also writes `/path/to/2026-10-kdd26/release_manifest.json`, indexing
every bundle and evaluation table by its provenance record, and prints what
it recorded. Check that the printed profile, config files and overrides are
those of the run. `save` refuses outputs without a provenance record,
written before artifacts recorded their provenance; regenerate them.

It also prints which commits produced each pipeline stage, whether the
release mixes commits, whether any artifact was produced from a dirty
tree or unknown code, and any input no bundle of the release matches (a
bundle regenerated after the artifacts that used it).

Release ids are letters, digits, `.`, `_` and `-`, and are never reused.
Use scope `full` for a run of the whole benchmark configuration and
`partial` for anything less. Smoke-test outputs are never published.

## 4. Review the package

Read `/path/to/2026-10-kdd26/release_manifest.json`
([fields](../reference/release_manifest.md#fields)) and check each item:

- [ ] `execution_mode` is `production`; `publish` refuses smoke-test
      artifacts.
- [ ] The printed producing code has no dirty or unknown commit, so each
      artifact's commit is the code that produced it. If the release mixes
      commits, the release notes say which stage each commit produced.
- [ ] No input was reported as matching no bundle, unless you left that
      bundle out on purpose.
- [ ] `workflow_config` is the configuration of the intended run.
- [ ] `coverage` lists the datasets, dataset realizations, methods,
      evaluation splits, budget settings, classifier variants and output
      categories you mean to publish.
- [ ] Every entry of `evaluations` has a `raw_path` and a
      `transformed_path`.
- [ ] `scope` is `full` only if the release covers the whole benchmark
      configuration, since the newest full release becomes everyone's
      default download. Otherwise it is `partial`.
- [ ] `output/` holds no stale or unrelated files; the snapshot copied the
      output root as it was.
- [ ] Every dataset in `dataset_redistribution` is `permitted`.
      `save` printed those that are not.

## 5. Write the release notes entry

Add an entry for the release at the top of
[`reference/release_notes.md`](../reference/release_notes.md), in the
format given there. To find the changes since the previous release:

1. Review `git log <previous commit>..<new commit>` for each producing
   commit `save` printed, especially changes under `conf/`,
   `workflow/`, `src/afabench/` and `scripts/`.
2. Compare the two manifests' `workflow_config.merged`, and the
   `resolved_config` of corresponding artifacts' provenance records.
3. Sort each change into a compatibility change or a result-affecting
   change ([the difference](../explanation/benchmark_releases.md#compatibility-and-comparability)).

## 6. Publish

```shell
uv run python scripts/release/snapshot.py publish /path/to/2026-10-kdd26 \
    --repo-id AFABench/results
```

The release is uploaded in one commit, and the command prints its
provenance and web address.

If `publish` refuses a dataset as `unreviewed` or `restricted`, finish its
review (step 2) and save the package again (step 3), or remove the dataset
from the run. To publish it anyway, allow it by its dataset key:
`--allow-redistribution <dataset key>`, once per dataset. Its status stays
in the published manifest for everyone to read.

If `publish` refuses artifacts produced from dirty or unknown code,
regenerate them from a clean commit and save the package again. To publish
them anyway, pass `--allow-dirty-code`; each artifact's dirty flag stays in
the published manifest, and the host's commit message records the
allowance.

A published release cannot be replaced. To correct one, publish a new
release id with a release notes entry saying what it corrects.
