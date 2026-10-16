---
status: accepted
---

# A downloaded release is a cache

_This replaces reference methods, drops smoke releases (amending
[ADR 0006](0006-release-manifest-indexes-artifact-provenance.md)), and
resolves #99 differently from the design the issue proposed._

A repository adopter downloads a benchmark release into their output root
and runs their own pipeline profile, which lists the published methods
they compare with in `methods` next to their own. The release is only a
cache: the run produces exactly what that profile would produce without
it, reusing what the release holds and producing the rest. Adding a
dataset the release lacks generates it, trains its classifier and
evaluates every method on it. The release's own workflow configuration is
never applied.

Downloads keep their recorded mtimes
([ADR 0003](0003-snapshots-omit-snakemake-metadata.md)), so ordinary runs judge
restored outputs by mtime and rerun what is missing or older than an input.
Explicit checked reuse handles locally regenerated dataset bundles, as below.
Ordinary dependency tracking also keeps the comparison consistent: a shared
prerequisite that was not downloaded is produced again, is newer than the
published method's tables, and so re-evaluates that method instead of
plotting its cached rows next to rows made with a different classifier.

## Content-checked dataset reuse (#154)

A release can omit dataset bundles without redistribution permission while
retaining their consumers. Regenerating identical bundles then makes their
mtimes newer than the released models and evaluations. `reuse_release=true`
checks actual dataset content hashes against those consumers' recorded input
hashes and ignores mtimes only on verified dataset edges to existing released
outputs. Unknown or mismatched hashes fail explicitly; missing outputs keep
ordinary dependencies. Validation includes existing released consumers on the
selected dataset realizations even when Snakemake prunes upstream jobs.

Dataset generation must finish in a separate invocation and be disabled during
reuse: a DAG-time check must not bless bytes a scheduled job will replace.
This costs an extra invocation rather than coupling generation to release
handling or manipulating timestamps. Globally ancient dataset inputs would
hide real changes; ordinary development therefore keeps its existing triggers.
Hash identity has ADR 0002's scope: saved bundle data, not external image pixels
or code compatibility. Redistribution review remains unchanged.

## Considered options

- **A separate `releases/<release_id>/` tree** that the workflow reads in
  place and never writes, with each comparison in its own output root.
  Rejected: the adopter's results then depend on which release they were
  built on, and the workflow needs a second, read-only root for shared
  prerequisites.
- **`--compare-with-release` layering the release's configuration over the
  profile** (#99's proposal), with published methods as **reference
  methods**: tables used as they are, never trained, evaluated or
  transformed, and a missing table failing the plan. Rejected: the release
  then decides what the run outputs, and anything the release lacks, such
  as a new dataset, cannot be computed for the published methods.
- **The release as a cache.** Chosen.

## Consequences

- `reference_methods` and `--compare-with-release` go. One release per
  output root still holds; a second release is downloaded into a second
  output root (`download --destination-root`, then `output_root=`), not a
  second clone.
- Reusing a published method needs its whole chain: shared prerequisites,
  pretrained model, AFA-method bundles and its tables. `download --method`
  selects that chain by default; without the AFA-method bundles the run
  retrains the method. `--payload-category` remains for results-only
  users, who never run the pipeline.
- A restored output is not rerun when the adopter's checkout changes the
  rule's code or params (ADR 0003). Whether results stay comparable across
  commits remains the adopter's judgement, as before.
- **No smoke releases.** A smoke test's outputs are never published.
  `publish` refuses a package with any smoke artifact, with no override;
  the `smoke` scope and `--smoke-release` go. The manifest still records
  `execution_mode`, read from the records, so a locally saved smoke run
  can be downloaded in tests. The publish/download round trip is checked
  by the test suite through a local host layout, not on the real host.
