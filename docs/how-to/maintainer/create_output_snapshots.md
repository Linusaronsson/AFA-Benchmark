# Create output snapshots

For a [maintainer](../../explanation/user_types.md#maintainer):
save pipeline outputs outside the repository.

## 1. Preview the outputs

```shell
uv run python scripts/release/snapshot.py inventory
```

Check sizes and any missing provenance before saving.

## 2. Save the snapshot

Choose a fresh destination:

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir
```

Find the saved outputs under `/path/to/snapshot-dir/output/`. To save smoke
outputs, add `--source-root output/smoke`. Use `--overwrite` only after
checking any listed conflicts.

[Save options](../../reference/snapshot_command.md#save) ·
[Restore a snapshot](restore_an_output_snapshot.md).
