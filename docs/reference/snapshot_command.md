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
else in the destination is deleted.

## Workflow configuration options

`save` with `--release-id` takes the workflow configuration whose targets
lay out the release, given the way it was given to Snakemake. It is
recorded as the manifest's `workflow_config`, never read to describe an
artifact:

| Option | Meaning |
| --- | --- |
| `--profile DIR` | The profile's `config.yaml` supplies `configfile` and `config`. |
| `--configfile PATH` (repeatable) | Replaces the profile's `configfile`. |
| `--config KEY=VALUE` (repeatable) | Replaces the profile's `config`, as Snakemake's `--config` does. Values are parsed as YAML. |

Config files are merged recursively in order and the overrides over them,
as Snakemake does.

## `save`

```shell
snapshot.py save SNAPSHOT_DIR [--source-root extra/output] [--overwrite]
    [--release-id ID --scope full|partial|smoke CONFIGURATION OPTIONS]
    [--checkout .]
```

Copies every file and directory under the source root to
`SNAPSHOT_DIR/output`, keeping file and directory modification times. A
missing or empty source root is an error.

With `--release-id`, also writes `SNAPSHOT_DIR/release_manifest.json`
([release manifest](release_manifest.md)), indexing the source root's
artifacts by their provenance records, and `--scope` and the workflow
configuration options are required. If the source root holds any job
record, it also writes the release's
[job duration table](release_manifest.md#job-duration-table) to
`SNAPSHOT_DIR/release_job_duration_table.parquet`. Refused before anything
is copied:

- any bundle, or Parquet file under `eval_results/` or
  `eval_results_transformed/`, without a provenance record; they are
  listed;
- any smoke-test artifact with a scope other than `smoke`;
- a source root without any artifact.

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
Some artifacts were produced from dirty or unknown code; publishing an official release needs --allow-dirty-code.
Transformed tables carry their evaluation's record; the commits of transformation, aggregation and visualization are not recorded.
```

## `restore`

```shell
snapshot.py restore SNAPSHOT_DIR [--destination-root extra/output] [--overwrite]
```

Copies `SNAPSHOT_DIR/output` into the destination root, keeping
modification times, and the snapshot's release manifest, if any, to
`release_manifest.json` beside the destination root
(`extra/release_manifest.json` by default), and its job duration table, if
any, to `release_job_duration_table.parquet` beside it. A manifest whose
`manifest_version` this checkout does not read is refused before anything
is restored.

## `inventory`

```shell
snapshot.py inventory [--source-root extra/output]
```

Prints the execution mode of the source root's artifacts and, per
[payload category](release_manifest.md#payload-categories), how many
artifacts are present, their total size and the bundle classes found, then
lists the artifacts without a provenance record, which `save --release-id`
would refuse. Writes nothing.

## `publish`

```shell
snapshot.py publish SNAPSHOT_DIR --repo-id REPO
    [--smoke-release] [--allow-redistribution DATASET_KEY ...]
    [--allow-dirty-code]
```

Uploads the snapshot as the release its manifest names, in one commit, to
`releases/<release_id>/`, or `smoke_releases/<release_id>/` with
`--smoke-release`. `--repo-id` defaults to `AFABENCH_RELEASE_REPO`.
Needs a Hugging Face token with write access (`hf auth login` or
`HF_TOKEN`); the repository must exist.

Refuses:

- a snapshot without a release manifest, or with a `manifest_version` this
  checkout does not read;
- a release id already published;
- scope `smoke` without `--smoke-release`, and `--smoke-release` with any
  other scope;
- an official release with an `unreviewed` or `restricted` dataset in
  `dataset_redistribution`, unless each such dataset key is given
  with `--allow-redistribution`. Allowing a dataset that is neither is an
  error. Allowed dataset keys are named in the host's commit message;
- an official release holding an artifact produced from a dirty tree or
  outside a git work tree (unknown code), unless `--allow-dirty-code` is
  given. Giving it for a release without one is an error. Its use is
  named in the host's commit message.

Smoke releases take neither allowance. On success, `publish` prints the
producing code per pipeline stage as `save` does.

## `download`

```shell
snapshot.py download [RELEASE] --repo-id REPO
    (--all | SELECTION OPTIONS)
    [--destination-root extra/output] [--overwrite] [--smoke-release]
```

Downloads one release and restores it as `restore` does, then prints its
scope, execution mode and workflow configuration. A public
repository needs no token.

### Release

`RELEASE` is a release id, or `latest` (the default): the `full` release
whose manifest `created_at` is newest. `latest` never chooses a `partial`
release, and fails, listing the published releases, if there is no full
one. Smoke releases are downloaded only with `--smoke-release` and by
release id.

The release is resolved once, before anything is downloaded, and every
file comes from it.

### Selection

`--all` downloads every file. Otherwise, name at least one category:

| Option (repeatable) | Downloads |
| --- | --- |
| `--payload-category` | `raw_evaluation_table`, `transformed_evaluation_table`, `dataset_bundle`, `classifier_bundle`, `pretrained_model_bundle` or `afa_method_bundle` files of the selected evaluations. |
| `--payload-category job_duration_table` | The release's job duration table alone, beside the restored manifest. Coverage options do not narrow it. |
| `--output-category` | A top-level folder of the output root listed in `coverage.output_categories`, such as `plot_results` or `merged_results`, whole. Coverage options do not narrow it. |

and optionally narrow the evaluations with coverage options:

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
  dataset 'physionet': no evaluation of the release matches
  transformed_evaluation_table of the evaluation eval_results/...: not in the release
  classifier_bundle extra/output/trained_classifiers/..., classifier input of eval_results/...: not in the release
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
smoke_releases/<release_id>/    same layout
```

Hugging Face keeps no modification times, so `publish` records them in
`output_mtimes.json` and `download` sets them again, including on empty
directories.
