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
   account. Use the exact organization slug shown by `hf auth whoami`,
   not its display name. For AFABench, the slug is lowercase `afabench`:

   ```shell
   uv run hf repo create afabench/results --repo-type dataset
   ```

   Use the same full repository id when publishing:
   `--repo-id afabench/results` (or set
   `AFABENCH_RELEASE_REPO=afabench/results`).

A `401 Unauthorized` error means authentication failed: check
`uv run hf auth whoami` and log in with a valid token before retrying.
The uv hardlink warning is unrelated to Hugging Face authentication.

Instead of `--repo-id <repo_id>` below, the repository can be in
`AFABENCH_RELEASE_REPO`.

## 2. Save the release package

First, [run the benchmark pipeline](run_the_pipeline.md) and wait for it
to finish. The command below does **not** run the benchmark.

Then run the following command to copy the existing outputs from
`output/production` into an [output snapshot](create_output_snapshots.md)
at `~/data/afabench-releases/example/` and create its release manifest.
Save this package outside the repository. Replace the example path with
a **fresh** destination folder you want to use, and choose a release id.
Overwriting an existing package retains unrelated destination files; use a
fresh destination to avoid carrying them forward.
The pipeline records its configuration automatically when it executes jobs;
`save` uses that record, so you do not need to repeat or manually verify
profiles, config files or overrides. For outputs built across several
invocations, the record is from the last invocation that executed jobs:

```shell
uv run python scripts/release/snapshot.py save ~/data/afabench-releases/example \
    --release-id example
```

The snapshot's outputs are saved under `~/data/afabench-releases/example/output/`.
The command also writes `~/data/afabench-releases/example/release_manifest.json`, indexing
every bundle and evaluation table by its provenance record, and prints what
it recorded, including compact coverage and expected dataset exclusions.
It also records file hashes for validation before publication. Configuration
is read from `output/production/workflow_config.json`,
not from the current checkout's configuration files. If you supply optional
configuration arguments, `save` refuses any mismatch with the recorded run
before copying anything. Nothing is uploaded until step 5.

`save` refuses an output root without a recorded workflow configuration;
execute pipeline jobs to create it (a dry run or an invocation with nothing
to do does not create a record). It also refuses outputs without a
provenance record; regenerate those outputs.

It also prints which commits produced each pipeline stage, whether the
release mixes commits, whether any artifact was produced from a dirty
tree or unknown code, and any input no bundle of the release matches (a
bundle regenerated after the artifacts that used it).

Release ids are letters, digits, `.`, `_` and `-`, and are never reused.
A release can cover any selection of datasets, methods and settings;
its manifest records what is included. Users must name its release id to
download it. Smoke-test outputs are never published.

## 3. Review the package

Use the printed coverage summary to confirm that the saved datasets,
dataset realizations, methods, splits, budget settings, classifier variants
and output categories are the ones you intend to publish. Check the printed
dataset exclusions against the benchmark's
[dataset release policy](../reference/release_manifest.md#dataset-redistribution).

Publication automatically validates file hashes and indexes, complete
raw/transformed evaluation pairs, input links, smoke status, dataset policy
and producing-code cleanliness before uploading anything. Policy-excluded
dataset inputs are expected; genuinely dangling inputs are errors. Mixed
known clean producing commits are allowed; describe their significance in
the release notes.

These checks cannot detect stale or unrelated files that were already in the
output root or destination when saved, or decide whether results from
different commits are scientifically comparable. Keep the output root scoped
to the intended run and save to a fresh destination. Do not edit the package
or modify it while publication is running.

## 4. Write the release notes entry

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

## 5. Publish

```shell
uv run python scripts/release/snapshot.py publish ~/data/afabench-releases/example \
    --repo-id afabench/results
```

The release is uploaded in one commit, and the command prints its
validation success, coverage, exclusions, producing commits and web address.

If validation fails, the command exits with a concise failure report and
uploads nothing. Restore missing inputs or regenerate incomplete evaluations
in the output root, then save to a fresh package (step 2). If files changed
after saving, save again rather than editing the manifest. Dataset bundles
excluded by policy or lacking explicit redistribution permission cannot be
published; there is no redistribution override.

If `publish` refuses artifacts produced from dirty or unknown code,
regenerate them from a clean commit and save the package again. To publish
them anyway, pass `--allow-dirty-code`; each artifact's dirty flag stays in
the published manifest, and the host's commit message records the
allowance.

A published release cannot be replaced. To correct one, publish a new
release id with a release notes entry saying what it corrects.
