---
status: accepted
---

# The release manifest indexes artifact provenance

_This amends the first bullet of ADR 0002's "Consequences" and the release
manifest introduced by #63._

A benchmark release can be built up across commits: classifiers trained on
one commit, a snapshot restored, methods trained and evaluated on a later
one. Snakemake reuses the restored outputs, so one output tree holds
artifacts produced by different code. Manifest version 1 could not describe
that. Its `code` was the checkout at the time `snapshot.py save` ran, and its
per-table and per-bundle identity was enumerated forward from the *current*
workflow config, so it described what that config would schedule, not what
actually ran. Since ADR 0002, every bundle and evaluation table carries its
own provenance record, so the manifest duplicated the artifacts and the
duplicate could go stale (#74).

The release manifest stays, shrunk to release-level facts plus an index
generated from the artifacts' provenance records at save time. The index
never reads the workflow config: it is a cache of the records, and
rebuilding it from the same output tree gives the same result.

## Considered options

- **Remove the manifest.** Release-level facts would move to the publish
  command's arguments or host metadata, and selective download would read the
  records directly. Rejected: a selective download would fetch every bundle
  manifest and raw table footer from the host before deciding anything,
  thousands of requests for a production release. The scope and the
  redistribution review that `publish` checked would also need another home
  on the host.
- **Shrink it to release-level facts plus an index of the records.** Chosen.

## What the manifest holds

- **Release-level facts**: `release_id`, `scope`, `created_at`, the
  maintainers' `dataset_redistribution` review for every dataset key any
  record names, and `workflow_config`: the configuration the maintainer
  declares, whose targets lay out the release. A repository adopter
  layers a comparison over it so that the workflow finds the published
  tables where the release has them. It stays correct for a release built
  across commits, since the final run reused the restored outputs because
  its targets name them. It is recorded, never read to describe an
  artifact. `execution_mode` is `smoke` if any record has `smoke_test`.
  Dataset generation has no smoke mode and always records production, so
  a smoke run's tree holds both kinds; a `full` or `partial` release
  cannot hold a smoke artifact.
- **An index**: one entry per bundle and per evaluation (its raw and
  transformed tables, paired by equal records). An entry holds a
  projection of the record: stage, code commit and dirty flag, seed, smoke
  flag, the dataset and method identity, and the inputs as
  `{role, path, content_hash}`. Evaluation entries also hold the identity
  columns and the classifier variants. The full record stays in the
  artifact. Copying it verbatim would cost several kilobytes per artifact,
  which every selective download fetches first.
- **Coverage**, derived from the index.

_Amended by [ADR 0007](0007-job-records-beside-artifacts.md): the manifest
also describes the release's job duration table, built from job records and
shipped beside the manifest._

Dropped: the save-time `code` and the resolved `settings`. Each artifact's record holds its producing script's
`resolved_config`; the feature-cost files and other checkout contents are
identified by the producing commit.

## Rules

- **Artifacts without a record are refused.** `save --release-id` lists
  every bundle without a provenance record, and every Parquet file under
  `eval_results/` or `eval_results_transformed/` without one, and writes
  nothing. A release must describe everything it ships, and such outputs
  can be regenerated. A snapshot without a release manifest takes any
  tree.
- **Inputs link by content hash.** An entry's input points to the bundle in
  the tree with the same `content_hash`, not to the recorded path. A path
  that still resolves after its bundle was regenerated would link to
  different code; the hash links to nothing. Such a **dangling input** is
  kept in the index, reported by `save`, and reported as missing by
  selective download. It is not refused, since a maintainer may leave
  bundles out of a partial release on purpose.
- **Producing commits are reported per stage.** `save` and `publish` print,
  for dataset generation, classifier training, pretraining, training and
  evaluation, each producing commit with its artifact count, and flag dirty
  and unknown code and a release that mixes commits. Transformed tables copy
  the evaluation's record, and aggregation and visualization write none, so
  the commits of transformation, aggregation and visualization are not
  reported.
- **Dirty code blocks official releases.** `publish` refuses a `full` or
  `partial` release in which any artifact was produced from a dirty tree or
  outside a git work tree (unknown code), unless the maintainer passes
  `--allow-dirty-code`, which the host's commit message records. Smoke
  releases are never refused for it.

## Consequences

- `snapshot.py inventory` no longer takes `--profile`, `--configfile` or
  `--config`.
- Manifest version 2. Version 1 manifests are not read; none was published.
- The raw and transformed tables are located by their top-level folders
  (`eval_results/`, `eval_results_transformed/`), the locations the payload
  category table already documents. Neither identity nor input links are
  read from paths.
