# `snapshot.py` command

`scripts/release/snapshot.py` saves, restores, publishes and downloads
output snapshots and benchmark releases. `publish` and `download` are the
only parts of AFABench that contact Hugging Face. Run any subcommand with
`--help` for its options.

```shell
uv run python scripts/release/snapshot.py <subcommand> ...
```

## Overwrite rule

`save`, `restore` and `download` check every file they would write before
writing anything. If any exists, nothing is written and the existing paths
are listed. With `--overwrite`, exactly those files are replaced; nothing
else in the destination is deleted, except that `save --release-id --overwrite` removes previously copied bundles now omitted by the release
manifest.

## Workflow configuration options

The pipeline writes `workflow_config.json` under its output root before
executing jobs, capturing Snakemake's merged configuration, config file
paths and hashes, overrides, and the pipeline profile when launched through
`workflow/run.py`. Dry runs, compute estimates and invocations with nothing
to execute do not replace it. Cluster workers do not write it.

`save --release-id` reads that record automatically into the manifest's
`workflow_config`; it does not reread the checkout's config files. This
configuration is never used to describe an artifact. An output root without
a record is refused; execute pipeline jobs to record their configuration
(a dry run or no-op invocation does not create the record).

These optional arguments assert a configuration against the recorded run.
A mismatch in merged values, overrides, file paths or hashes, or an
explicit profile mismatch, is refused before copying anything:

| Option | Meaning |
| --- | --- |
| `--profile DIR` | Check the profile identity; its `config.yaml` supplies the asserted `configfile` and `config`. |
| `--configfile PATH` (repeatable) | Replaces the profile's `configfile`. |
| `--config KEY=VALUE` (repeatable) | Replaces the profile's `config`, using Snakemake's value parsing. |

Config files are merged recursively in order and the overrides over them,
as Snakemake does. Override scalars are parsed as integers, floats or
Python booleans (`True`/`False`), then as YAML with string scalars. For
example, `false` stays a string, and `[1, 2]` becomes a list of strings.

## `save`

```shell
snapshot.py save SNAPSHOT_DIR [--source-root output/production] [--overwrite]
    [--release-id ID --scope full|partial [CONFIGURATION OPTIONS]]
    [--checkout .]
```

A smoke test's outputs are under `output/smoke`: pass
`--source-root output/smoke` to save them.

Copies every file and directory under the source root to
`SNAPSHOT_DIR/output`, keeping file and directory modification times,
except a restored release's `release_manifest.json` and
`release_job_duration_table.parquet` at the top of the source root. A
missing or empty source root is an error.

With `--release-id`, also writes `SNAPSHOT_DIR/release_manifest.json`
([release manifest](release_manifest.md)), indexing the source root's
artifacts by their provenance records. `--scope` and a recorded workflow
configuration are required; configuration options are optional assertions. If the source root holds any job
record, it also writes the release's
[job duration table](release_manifest.md#job-duration-table) to
`SNAPSHOT_DIR/release_job_duration_table.parquet`. Refused before anything
is copied:

- any bundle, or Parquet file under `eval_results/` or
  `eval_results_transformed/`, without a provenance record; they are
  listed;
- a source root without any artifact.

Dataset bundles are included only when their dataset key has an explicit
`permitted` review. Other dataset bundles are omitted from the output tree,
the manifest's bundle index and payload coverage, and their paths are printed.
Models and evaluation results remain included regardless of dataset status.
A release with no artifacts left after omission is refused. Plain snapshots
without `--release-id` remain verbatim copies.

A smoke test's outputs are saved with execution mode `smoke`, which
`publish` refuses.

`--checkout` names the git checkout whose redistribution reviews are read.
The command prints the recorded configuration, the release identity, every
dataset whose redistribution is `unreviewed` or `restricted`, the
producing code per pipeline stage, and the
[dangling inputs](release_manifest.md#inputs):

```text
Producing code per pipeline stage:
  dataset_generation: 3f2a…, 6 artifact(s)
  classifier_training: 3f2a…, 1 artifact(s)
  training: 9c41… (dirty), 4 artifact(s)
  evaluation: 9c41… (dirty), 4 artifact(s)
The release mixes 2 producing commits.
Some artifacts were produced from dirty or unknown code; publishing the release needs --allow-dirty-code.
Transformed tables carry their evaluation's record; the commits of transformation, aggregation and visualization are not recorded.
```

## `restore`

```shell
snapshot.py restore SNAPSHOT_DIR [--destination-root ROOT] [--overwrite]
```

The destination root defaults to `output/production`, or
`output/smoke` when the snapshot's release manifest has execution mode
`smoke`, so smoke outputs never satisfy a production run.

Copies `SNAPSHOT_DIR/output` into the destination root, keeping
modification times, and the snapshot's release manifest and job duration
table, if any, into the destination root itself, as
`release_manifest.json` and `release_job_duration_table.parquet`. In a
snapshot they sit beside `output/`; in a checkout they sit inside the
root, so the smoke and production output roots each keep their own
release's files. `save` leaves these two files out when it copies a
source root, so a new release never carries a restored one's. A manifest whose
`manifest_version` this checkout does not read is refused before anything
is restored.

## `inventory`

```shell
snapshot.py inventory [--source-root output/production]
```

Pass `--source-root output/smoke` for a smoke test's outputs.

Prints the execution mode of the source root's artifacts and, per
[payload category](release_manifest.md#payload-categories), how many
artifacts are present, their total size and the bundle classes found, then
lists the artifacts without a provenance record, which `save --release-id`
would refuse. Writes nothing.

## `publish`

```shell
snapshot.py publish SNAPSHOT_DIR --repo-id REPO [--allow-dirty-code]
```

Uploads the snapshot as the release its manifest names, in one commit, to
`releases/<release_id>/`. `--repo-id` defaults to `AFABENCH_RELEASE_REPO`.
Needs a Hugging Face token with write access (`uv run hf auth login` or
`HF_TOKEN`); the repository must exist.

Refuses:

- a snapshot without a release manifest, or with a `manifest_version` this
  checkout does not read;
- a release id already published;
- any smoke-test artifact, with no override; up to five of them are
  listed, with their total count;
- any dataset bundle without an explicit `permitted` review in the saved
  manifest, with no override. Models and results from restricted or
  unreviewed datasets are allowed;
- bundles added, removed or changed since saving the manifest, bundles
  outside `output/`, or artifacts without provenance records. The package
  is inspected before any upload;
- a release holding an artifact produced from a dirty tree or
  outside a git work tree (unknown code), unless `--allow-dirty-code` is
  given. Giving it for a release without one is an error. Its use is
  named in the host's commit message.

On success, `publish` prints the
producing code per pipeline stage as `save` does.

## `download`

```shell
snapshot.py download [RELEASE] --repo-id REPO
    (--all | SELECTION OPTIONS)
    [--destination-root ROOT] [--overwrite]
```

The destination root defaults to `output/production`, or `output/smoke`
for a release whose manifest has execution mode `smoke`, which only the
test suite's local host holds. One output root holds one release: download
a second release into another root, with `--destination-root
output/<name>`, and run the pipeline on it with `output_root=output/<name>`.

Downloads one release and restores it as `restore` does, then prints its
scope, execution mode and workflow configuration. A public
repository needs no token.

### Release

`RELEASE` is a release id, or `latest` (the default): the `full` release
whose manifest `created_at` is newest. `latest` never chooses a `partial`
release, and fails, listing the published releases, if there is no full
one.

The release is resolved once, before anything is downloaded, and every
file comes from it.

### Selection

`--all` downloads every file. Otherwise, select evaluations with coverage
options, name categories to download, or both. Coverage options without
`--payload-category` download the whole chain of the selected evaluations:
every payload category but `job_duration_table`, which is what a run needs
to reuse a published method instead of producing it again
([ADR 0012](../adr/0012-a-downloaded-release-is-a-cache.md)). Categories:

| Option (repeatable) | Downloads |
| --- | --- |
| `--payload-category` | `raw_evaluation_table`, `transformed_evaluation_table`, `dataset_bundle`, `classifier_bundle`, `pretrained_model_bundle` or `afa_method_bundle` files of the selected evaluations. |
| `--payload-category job_duration_table` | The release's job duration table alone, beside the restored manifest. Coverage options do not narrow it. |
| `--output-category` | A top-level folder of the output root listed in `coverage.output_categories`, such as `plots` or `merged_results`, whole. Coverage options do not narrow it. |

Coverage options select evaluations:

| Option (repeatable) | Selects evaluations with |
| --- | --- |
| `--dataset` | this dataset key |
| `--method` | this method name |
| `--dataset-realization` | this dataset realization index |
| `--eval-split` | this evaluation split |
| `--initializer` | this initializer |
| `--budget-setting` | `hard_budget` or `soft_budget` |
| `--classifier-variant` | non-null `builtin` or `external` predictions |

Values of one option match any; different options must all match; an
option not given does not restrict.

For the selected evaluations, payload categories download:

- their raw or plotting-ready evaluation tables;
- the bundles they were produced from, found by following the `inputs` of
  the evaluations and, in turn, of the bundles, by content hash, if the
  bundle's category is named. For example, the external classifier of a
  dataset realization was trained on that realization's `train` and `val`
  dataset bundles, so selecting it also selects them.
- with each pretrained-model and AFA-method bundle, the folder of the job
  that wrote it, including its job record.

A method-specific classifier comes only with its method's evaluations.

### Coverage report

What the selection asks for but the release lacks is printed after the
download:

```text
Downloaded 6 file(s) and 1 folder(s).
Missing from release 2026-11-partial:
  dataset 'physionet_challenge_2012': no evaluation of the release matches
  transformed_evaluation_table of the evaluation eval_results/...: not in the release
  classifier_bundle output/production/trained_classifiers/..., classifier input of eval_results/...: not in the release
```

An input is reported only if its category is named; a
[dangling input](release_manifest.md#inputs)'s own inputs are unknown, so
they are not followed.

If nothing selected is in the release, the command fails and writes
nothing.

## Release host layout

```text
releases/<release_id>/
    release_manifest.json
    release_job_duration_table.parquet     if the release has job records
    output_mtimes.json      modification times of everything under output/
    output/                 the snapshot's output tree
```

Hugging Face keeps no modification times, so `publish` records them in
`output_mtimes.json` and `download` sets them again, including on empty
directories.
