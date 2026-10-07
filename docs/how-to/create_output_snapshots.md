# Create output snapshots

For a [maintainer](../explanation/user_types.md#maintainer): copy
everything the pipeline wrote under `extra/output` to a folder outside the
repository, and put it back later exactly where Snakemake expects it. A
benchmark release is published from a snapshot.
[What a snapshot holds and why](../explanation/output_snapshots.md).

## Save a snapshot

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir
```

The outputs are copied to `/path/to/snapshot-dir/output`, with their
modification times.

## Save a snapshot as a benchmark release

To publish the snapshot later, give it a release id and a scope, and the
workflow configuration the pipeline ran with, the same way it was given to
Snakemake:

```shell
uv run python scripts/release/snapshot.py save /path/to/2026-10-kdd26 \
    --release-id 2026-10-kdd26 --scope full \
    --profile extra/workflow/profiles/config/kdd26
```

This also writes `/path/to/2026-10-kdd26/release_manifest.json`
([fields](../reference/release_manifest.md)) and prints what it recorded.
Check that the printed profile, config files and overrides are those of
the run.

Use scope `full` for a run of the whole benchmark configuration, `partial`
for anything less, and `smoke` for smoke-test outputs, which can have no
other scope.

## Preview a snapshot

To see what a snapshot would hold without copying anything, run
`inventory` with the same configuration options:

```shell
uv run python scripts/release/snapshot.py inventory \
    --profile extra/workflow/profiles/config/kdd26
```

It prints each payload category's number of files and size.

## Restore a snapshot

```shell
uv run python scripts/release/snapshot.py restore /path/to/snapshot-dir
```

The outputs are copied back to `extra/output`, and a release manifest, if
the snapshot has one, to `extra/release_manifest.json`. If any of these
files already exists, nothing is copied and the conflicts are listed; add
`--overwrite` to replace them.

## Check that a restore worked

```shell
uv run snakemake --dry-run all
```

If it reports nothing to do, the restore is complete. If it schedules
jobs, an output is missing or older than one of its inputs: the restore
was incomplete, or an input changed after the snapshot was taken.

The dry run does not notice a change in a rule's code or parameters since
the snapshot was taken. If you changed one, delete or regenerate the
outputs of that rule yourself
([why](../explanation/output_snapshots.md#why-a-snapshot-omits-snakemake-metadata)).

All options: [`snapshot.py` reference](../reference/snapshot_command.md).
