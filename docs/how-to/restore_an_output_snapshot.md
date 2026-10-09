# Restore an output snapshot

For a [maintainer](../explanation/user_types.md#maintainer): put a saved
[output snapshot](../explanation/output_snapshots.md) back under
`output/production`, where Snakemake reuses it instead of producing it again.

## 1. Restore the snapshot

```shell
uv run python scripts/release/snapshot.py restore /path/to/snapshot-dir
```

The outputs are copied back to `output/production`, or to
`output/smoke` if the snapshot's release manifest has scope `smoke`.
The release manifest, if the snapshot has one, goes inside that root, to
`release_manifest.json`. If any of these
files already exists, nothing is copied and the conflicts are listed; add
`--overwrite` to replace them.

## 2. Check that the restore worked

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

All options: [`snapshot.py` reference](../reference/snapshot_command.md#restore).
