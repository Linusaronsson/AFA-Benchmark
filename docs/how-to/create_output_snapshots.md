# Create output snapshots

For a [maintainer](../explanation/user_types.md#maintainer): copy
everything the pipeline wrote under `extra/output` to a folder outside the
repository. [What a snapshot holds and why](../explanation/output_snapshots.md).
To put it back, see [restore an output snapshot](restore_an_output_snapshot.md).

## 1. Preview the snapshot (optional)

To see what the snapshot will hold without copying anything, run
`inventory` with the workflow configuration the pipeline ran with:

```shell
uv run python scripts/release/snapshot.py inventory \
    --profile extra/workflow/profiles/config/kdd26
```

It prints each payload category's number of files and size.

## 2. Save the snapshot

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir
```

The outputs are copied to `/path/to/snapshot-dir/output`, with their
modification times. If any file there already exists, nothing is copied
and the conflicts are listed; add `--overwrite` to replace them.

All options: [`snapshot.py` reference](../reference/snapshot_command.md#save).
