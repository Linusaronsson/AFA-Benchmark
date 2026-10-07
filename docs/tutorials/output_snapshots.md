# Output snapshots

An **output snapshot** (see `CONTEXT.md`) is a verbatim copy of everything
under the pipeline's output root, saved outside the repository so it can be
put back exactly where Snakemake expects it later. It carries no provenance
of its own and is not curated; a benchmark release is built from one, but
that manifest and provenance work is separate (#63), as is Snakemake
execution metadata (#64).

## Save

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir
```

Copies every file under the source root (`extra/output` by default; override
with `--source-root`) into `/path/to/snapshot-dir/output`, preserving file
and directory mtimes. A missing or empty source root is an error naming the
path, not an empty snapshot.

## Restore

```shell
uv run python scripts/release/snapshot.py restore /path/to/snapshot-dir
```

Copies every file from `/path/to/snapshot-dir/output` back into the
destination root (`extra/output` by default; override with
`--destination-root`).

## The overwrite rule

Both commands check every destination path before writing anything. If any
destination already exists, the whole operation is refused and the target is
left untouched. Pass `--overwrite` to proceed; only the conflicting paths are
replaced, and nothing else already in the target is deleted.

## Why mtimes are preserved

Snakemake decides whether an output is stale by comparing mtimes: a
directory output is timestamped by the mtime of its `.snakemake_timestamp`
file where one exists, otherwise by the directory's own mtime, and a file
output by its own mtime. A snapshot preserves both file and directory
mtimes, including hidden `.snakemake_timestamp` files, so a tree that was
internally consistent (no rule's inputs newer than its outputs) stays
consistent after being saved and restored.

## Verifying a restore

After restoring, run a Snakemake dry run against the same targets. If the
restore reproduced the tree correctly, the dry run reports nothing to do:

```shell
uv run snakemake --dry-run all
```

If it instead schedules jobs, something in the restored tree is not
byte-for-byte and mtime-for-mtime identical to what Snakemake last saw.

## What a snapshot does not contain

- No manifest, commit hash, or timestamp of the snapshot itself (#63).
- No Snakemake execution metadata, i.e. `.snakemake/metadata` (#64).
- No selection by dataset, method, or output category: the whole output root
  is copied verbatim, including stale files no current rule would produce
  (#39, #40).
