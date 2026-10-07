---
status: accepted
---

# Output snapshots omit `.snakemake/metadata`

An output snapshot (#37) copies every file under the pipeline's output root,
preserving content and mtimes, so a restored tree looks internally
consistent to Snakemake's mtime check. It does not copy
`.snakemake/metadata`, the per-output execution record Snakemake uses for
its other rerun triggers. Issue #64 asked whether it should.

## Probe findings

A probe against Snakemake 9.12.0 (a scratch chain of three rules, outputs
copied into a fresh working directory) established:

- An output with no execution record is judged by mtime only. The code,
  params, input-set and software-environment rerun triggers never fire for
  it, so a restored output is never regenerated when its rule's code or
  params change.
- Copying `.snakemake/metadata` alongside the outputs restores those
  triggers; its records are keyed by relative path and are portable across
  working directories.
- `snakemake --touch` repairs mtime inversions without running jobs and
  seeds metadata entries for the touched outputs.
- A directory output is timestamped by the `.snakemake_timestamp` file
  inside it; without that file Snakemake falls back to the directory's own
  mtime.

So capturing metadata is technically straightforward: copy
`.snakemake/metadata` the same way `save_snapshot`/`restore_snapshot` already
copy the output tree, keyed by the same relative paths.

## Considered options

- **Copy `.snakemake/metadata` verbatim alongside the outputs.** Restores
  the code/params/input-set/software-environment rerun triggers exactly as
  if the run had never left its original working directory. Rejected: see
  below.
- **Run `snakemake --touch` after restore.** Repairs mtime inversions and
  seeds metadata for the touched outputs, but still ties rerun triggers to
  whatever rule code and params are current in the *restoring* checkout,
  not the snapshot's producing checkout, so it does not actually solve the
  problem the probe describes; it only silences the mtime path. Rejected as
  solving a different problem than the one raised.
- **Document the limitation and recommend a dry run.** Chosen.

## Decision

A snapshot does not carry `.snakemake/metadata`, and `save_snapshot`/
`restore_snapshot` are not changed. A restored output is "ancient" with
respect to Snakemake's code, params, input-set and software-environment
rerun triggers: changing a rule's code or params after a restore does not,
by itself, cause Snakemake to schedule a rerun for outputs that came from
the snapshot. Only the mtime check still applies, because snapshot mtimes
round-trip (`docs/tutorials/output_snapshots.md`, "Why mtimes are
preserved").

This follows directly from #36's position that checkout compatibility is
the user's responsibility, not something the tooling enforces: "Display
provenance and permit older-release selection, but do not add automatic
compatibility enforcement." The rerun triggers metadata restores are
exactly that enforcement, pointed in the wrong direction for this
project's two journeys:

- Every pipeline rule in `extra/workflow/snakefiles/` is a `shell:` rule
  that invokes an external script (`afabench`'s training, evaluation and
  dataset-generation code lives in separate files executed as
  subprocesses, not inlined as Snakemake `run:`/`script:` blocks). For a
  `shell:` rule, Snakemake's code trigger hashes the literal, unresolved
  shell command text written in the `.smk` file, not the invoked script's
  content (`snakemake.persistence.Persistence._code`). So it cannot be
  reading "did the training/evaluation code that actually ran change";
  no amount of restored metadata makes it read that. It reads "did this
  rule's line in the Snakefile change", which can differ between the
  snapshot's producing commit and a restoring checkout for reasons that
  have nothing to do with the restored output: an unrelated rule's
  argument-rendering refactor, a reformatted shell line, a renamed
  wildcard. None of the project's `shell:` rules use `conda:`/`container:`
  either, so the software-environment trigger never applies regardless of
  metadata.
- A benchmark adopter forks the repository, then downloads published
  baseline bundles and evaluation tables so Snakemake only runs the new
  method's missing work (#36, story 26 and the "default aggregation...
  must not recreate published baselines unnecessarily" decision). Their
  checkout's `.smk` files will have moved on from the release's producing
  commit as the project's own Snakefiles keep evolving, independently of
  whether the restored rule's actual behaviour changed. Restoring metadata
  ties every restored output's fate to that Snakefile-text match, risking
  exactly the retraining the adopter journey exists to avoid, as a side
  effect of a checkout difference nobody asked the tooling to police.
- A snapshot is explicitly not curated and carries no provenance of its own
  (`CONTEXT.md`, "Output snapshot"); a benchmark release built from one adds
  that provenance separately (#63). Execution metadata tied to one
  checkout's rule code is exactly the kind of implicit, unreviewed
  compatibility signal the release design keeps out of the snapshot layer,
  leaving provenance and compatibility judgement to the release manifest
  and the user, respectively.

Within a single, unchanged checkout a restored tree behaves like Snakemake
left it: nothing to do unless code or params actually changed. The
documented gap is for the case the probe raises — resuming work after
restoring a snapshot into a checkout whose rule code or params have moved
on.

## Recommended check after restore

Run a dry run against the targets of interest before trusting a restored
tree, and treat its exit code as advisory, not authoritative:

```shell
uv run snakemake --dry-run <target>
```

"Nothing to be done" means the mtime check found no inconsistency; it does
not mean the restored outputs match the current rule code or params. If
anything in a rule the restored outputs belong to changed since the
snapshot was taken, delete or regenerate those outputs explicitly rather
than relying on Snakemake to notice.

## Consequences

- `docs/tutorials/output_snapshots.md` documents this as a known limitation
  instead of an open question, with the dry-run check given above.
- If a future ticket needs rerun-trigger fidelity across a restore within
  one unchanged checkout (e.g. resuming an interrupted run from a snapshot
  on the same machine), that is a narrower case than cross-checkout release
  distribution and would need its own decision; this ADR does not cover it.
- `save_snapshot`/`restore_snapshot` and `scripts/release/snapshot.py` are
  unchanged by this ticket.
