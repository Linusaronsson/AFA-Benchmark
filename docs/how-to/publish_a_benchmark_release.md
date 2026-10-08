# Publish a benchmark release

For a [maintainer](../explanation/user_types.md#maintainer): publish the
outputs of a benchmark run on Hugging Face as a benchmark release.
[What a release is and why it works this way](../explanation/benchmark_releases.md).

## 1. Set up the host (once)

Create a Hugging Face dataset repository, public for an announced release,
and log in with a token that can write to it:

```shell
hf repo create <repo_id> --repo-type dataset
hf auth login
```

Instead of `hf auth login`, the token can be in `HF_TOKEN`. Instead of
`--repo-id <repo_id>` below, the repository can be in
`AFABENCH_RELEASE_REPO`.

## 2. Review dataset redistribution

Only datasets whose bundles may be published publicly can be in an
official release. For each dataset of the run, read its source's licence
or terms and record the review in
`extra/conf/release/dataset_redistribution.yaml` (format in the file's
header; [what a review covers](../reference/release_manifest.md#dataset-redistribution)).
A dataset without an entry is `unreviewed`.

## 3. Save the release package

Run the benchmark ([reproduce the full results](reproduce_full_results.md)).
Then save its outputs as an
[output snapshot](create_output_snapshots.md) with a release manifest, by
giving a release id, a scope and the workflow configuration whose targets
lay out the outputs, the same way it was given to Snakemake. For a release
built across several runs, that is the configuration of the final run:

```shell
uv run python scripts/release/snapshot.py save /path/to/2026-10-kdd26 \
    --release-id 2026-10-kdd26 --scope full \
    --profile extra/workflow/profiles/config/kdd26
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
Use scope `full` for a run of the whole benchmark configuration, `partial`
for anything less, and `smoke` for smoke-test outputs, which can have no
other scope.

## 4. Review the package

Read `/path/to/2026-10-kdd26/release_manifest.json`
([fields](../reference/release_manifest.md#fields)) and check each item:

- [ ] `execution_mode` is `production`.
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
   commit `save` printed, especially changes under `extra/conf/`,
   `extra/workflow/`, `afabench/` and `scripts/`.
2. Compare the two manifests' `workflow_config.merged`, and the
   `resolved_config` of corresponding artifacts' provenance records.
3. Sort each change into a compatibility change or a result-affecting
   change ([the difference](../explanation/benchmark_releases.md#compatibility-and-comparability)).

## 6. Try the round trip with a smoke release (optional)

Before announcing a release, check the real host with a smoke run's
package, preferably in a separate scratch repository:

```shell
uv run python scripts/release/snapshot.py publish /path/to/smoke-check \
    --repo-id <scratch_repo_id> --smoke-release
uv run python scripts/release/snapshot.py download smoke-check --all \
    --repo-id <scratch_repo_id> --smoke-release \
    --destination-root /tmp/check/output
```

## 7. Publish

```shell
uv run python scripts/release/snapshot.py publish /path/to/2026-10-kdd26 \
    --repo-id <repo_id>
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
