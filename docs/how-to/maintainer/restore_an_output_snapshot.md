# Restore an output snapshot

For a [maintainer](../../explanation/user_types.md#maintainer):
restore saved pipeline outputs.

## 1. Restore the snapshot

```shell
uv run python scripts/release/snapshot.py restore /path/to/snapshot-dir
```

Check outputs in `output/production/`, or `output/smoke/` for a smoke snapshot.
To use another root, pass `--destination-root <root>`. Use `--overwrite`
only after reviewing the listed conflicts.

## 2. Check the restored run

Dry-run the profile used to produce the snapshot; here, `kdd26`:

```shell
workflow/run.py kdd26 -n
```

Pass `output_root=<root>` if you restored elsewhere. Investigate scheduled
jobs for missing outputs or changed inputs. If code or parameters changed
since the snapshot, explicitly regenerate affected outputs; do not rely on
the dry run to detect those changes.

[Restore options](../../reference/snapshot_command.md#restore) ·
[Reuse limits](../../explanation/output_snapshots.md#why-a-snapshot-omits-snakemake-metadata).
