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

`save` with `--release-id`, and `inventory`, take the workflow
configuration the pipeline ran with, given the way it was given to
Snakemake:

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
([release manifest](release_manifest.md)), and `--scope` and the workflow
configuration options are required. If the merged configuration has
`smoke_test: true`, any scope but `smoke` is refused before anything is
copied. `--checkout` names the git checkout whose commit is recorded,
whose feature-cost files are hashed and whose redistribution reviews are
read. The command prints the recorded configuration, the release identity,
the commit, and every dataset whose redistribution is `unreviewed` or
`restricted`.

## `restore`

```shell
snapshot.py restore SNAPSHOT_DIR [--destination-root extra/output] [--overwrite]
```

Copies `SNAPSHOT_DIR/output` into the destination root, keeping
modification times, and the snapshot's release manifest, if any, to
`release_manifest.json` beside the destination root
(`extra/release_manifest.json` by default). A manifest whose
`manifest_version` this checkout does not read is refused before anything
is restored.

## `inventory`

```shell
snapshot.py inventory [--source-root extra/output] CONFIGURATION OPTIONS
```

Prints, per [payload category](release_manifest.md#payload-categories),
how many payloads the configuration schedules, how many are present, their
total size and the bundle classes found. Writes nothing.

## `publish`

```shell
snapshot.py publish SNAPSHOT_DIR --repo-id REPO
    [--smoke-release] [--allow-redistribution DATASET_KEY ...]
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
  `settings.dataset_redistribution`, unless each such dataset key is given
  with `--allow-redistribution`. Allowing a dataset that is neither is an
  error, and smoke releases take no allowance. Allowed dataset keys are
  named in the host's commit message.

## `download`

```shell
snapshot.py download [RELEASE] --repo-id REPO
    (--all | SELECTION OPTIONS)
    [--destination-root extra/output] [--overwrite] [--smoke-release]
```

Downloads one release and restores it as `restore` does, then prints its
scope, execution mode, commit and workflow configuration. A public
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
| `--classifier-variant` | `builtin` or `external` predictions in the raw table |

Values of one option match any; different options must all match; an
option not given does not restrict.

For the selected evaluations, payload categories download:

- their raw or plotting-ready evaluation tables;
- the bundles they were produced from, found by following the `inputs` of
  the tables and, in turn, of the bundles, if the bundle's category is
  named. For example, the external classifier was trained on dataset
  realization 0, so selecting it also selects that realization's `train` and
  `val` dataset bundles.
- with each pretrained-model and AFA-method bundle, the folder of the job
  that wrote it, including its `pretrain_time.txt` or `train_time.txt`.

A method-specific classifier comes only with its method's evaluations.

### Coverage report

What the selection asks for but the release lacks is printed after the
download:

```text
Downloaded 6 file(s) and 1 folder(s).
Missing from release 2026-11-partial:
  dataset 'physionet': no evaluation of the release matches
  transformed_evaluation_table eval_results_transformed/...: not in the release
```

If nothing selected is in the release, the command fails and writes
nothing.

## Release host layout

```text
releases/<release_id>/
    release_manifest.json
    output_mtimes.json      modification times of everything under output/
    output/                 the snapshot's output tree
smoke_releases/<release_id>/    same layout
```

Hugging Face keeps no modification times, so `publish` records them in
`output_mtimes.json` and `download` sets them again, including on empty
directories.
