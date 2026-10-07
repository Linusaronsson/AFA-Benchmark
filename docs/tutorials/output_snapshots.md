# Output snapshots

An **output snapshot** (see `CONTEXT.md`) is a verbatim copy of everything
under the pipeline's output root, saved outside the repository so it can be
put back exactly where Snakemake expects it later. It is not curated; a
benchmark release is built from one. Saved with `--release-id`, it also
carries a release manifest recording its identity, provenance and coverage
(see [`../release_manifest.md`](../release_manifest.md)). It deliberately
omits Snakemake execution metadata (#64).

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

## Release manifest

```shell
uv run python scripts/release/snapshot.py save /path/to/snapshot-dir \
    --release-id smoke-check --scope test_only \
    --profile extra/workflow/profiles/config/all \
    --config "datasets=[cube]" --config "dataset_instance_indices=[0]" \
    --config smoke_test=true --config use_wandb=false
```

Writes `/path/to/snapshot-dir/release_manifest.json` beside `output/` and
prints the workflow configuration it recorded. Pass the same profile,
config files and overrides the pipeline ran with. Smoke outputs can only be
saved with `--scope test_only`. `restore` puts the manifest beside the
destination root (`extra/release_manifest.json` by default). Fields and
rules: [`../release_manifest.md`](../release_manifest.md). Publishing such a
snapshot as a benchmark release and downloading it again:
[`../release_publishing.md`](../release_publishing.md).

To see what a snapshot would hold before taking one, run `inventory` with
the same configuration options; it prints each payload category's count and
size and copies nothing:

```shell
uv run python scripts/release/snapshot.py inventory \
    --profile extra/workflow/profiles/config/all \
    --config "datasets=[cube]" --config "dataset_instance_indices=[0]" \
    --config smoke_test=true --config use_wandb=false
```

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

After restoring, run a Snakemake dry run against the targets of interest:

```shell
uv run snakemake --dry-run all
```

If the restore reproduced the tree correctly *and* nothing relevant has
changed since the snapshot was taken, the dry run reports nothing to do. If
it instead schedules jobs, either the restored tree is not byte-for-byte and
mtime-for-mtime identical to what Snakemake last saw, or the dry run caught
a genuine change in a rule's code, params, inputs or software environment.

Treat a "nothing to be done" result as advisory, not authoritative, when
the restoring checkout's rule code or params may differ from the snapshot's
producing checkout: see the next section.

## What a snapshot does not contain

- No release manifest unless saved with `--release-id`.
- No selection by dataset, method, or output category: the whole output root
  is copied verbatim, including stale files no current rule would produce
  (#39). Selection happens when downloading a published release
  ([`release_publishing.md`](../release_publishing.md#selecting-what-to-download)).
- No Snakemake execution metadata, i.e. `.snakemake/metadata`. This is a
  deliberate decision (#64, `docs/adr/0003-snapshots-omit-snakemake-metadata.md`),
  not a gap to fill later.

  Snakemake's code, params, input-set and software-environment rerun
  triggers rely on that metadata. Without it, a restored output is judged
  by mtime only: Snakemake never notices that its rule's code or params
  changed since the snapshot was taken. Capturing the metadata instead was
  rejected: every pipeline rule is a `shell:` rule, so the code trigger
  hashes the literal shell command written in the `.smk` file, not the
  invoked script's content, and can change for reasons unrelated to the
  restored output (an unrelated rule's refactor, a reformatted shell line).
  Restoring metadata ties a restored output's fate to that Snakefile-text
  match across checkouts, which risks scheduling reruns of published
  baselines merely because a benchmark adopter's fork's `.smk` files have
  moved on from the producing commit — exactly the automatic compatibility
  enforcement #36 decided against, and exactly the retraining the adopter
  workflow exists to avoid. See
  `docs/adr/0003-snapshots-omit-snakemake-metadata.md` for the full
  reasoning.

  **Practical consequence:** after restoring a snapshot into a checkout
  whose rule code or params have since changed, do not trust a Snakemake
  dry run to detect it. Run `uv run snakemake --dry-run <target>` to catch
  mtime inconsistencies, but if you know a relevant rule's code or params
  changed, regenerate or delete those outputs explicitly instead of relying
  on Snakemake to schedule the rerun for you.
