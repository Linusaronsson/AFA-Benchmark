# Benchmark releases

A benchmark release publishes the outputs of a benchmark run, so that
[results-only users and repository adopters](user_types.md) can use them
without running the benchmark. It is an
[output snapshot](output_snapshots.md) saved with a
[release manifest](../reference/release_manifest.md) and published on
Hugging Face by a maintainer. A release can be built up across commits:
classifiers trained on one commit, a snapshot restored, methods trained and
evaluated on a later one. Every bundle and evaluation table records the
commit that produced it, and the manifest indexes those records
([ADR 0006](../adr/0006-release-manifest-indexes-artifact-provenance.md)),
so a release says which code produced each of its artifacts.

## What a release guarantees

- **One identity, one content.** A release id always names the same files:
  a published release is never replaced, and a correction is a new release.
- **One release per download.** A download resolves its release once and
  takes every file from it, so outputs of different releases are never
  mixed, even if a release is published while it runs. What the release
  lacks is reported, not filled in from elsewhere.
- **A stable default.** Without a release id, downloads use the newest
  `full` release. A newer `partial` release covers only part of the
  benchmark, so it is downloaded only by its id.
- **Smoke outputs are never benchmark results.** `publish` refuses any
  package holding a smoke-test artifact. The test suite checks the publish
  and download round trip on a local host instead.
- **A downloaded release is a cache.** A run over an output root holding a
  release produces exactly what its pipeline profile produces without it:
  it reuses what the release holds and produces the rest. The release's
  own workflow configuration is never applied
  ([ADR 0012](../adr/0012-a-downloaded-release-is-a-cache.md)).
- **Native files.** Tables are the pipeline's own Parquet files and bundles
  its own bundle folders, restored byte for byte to the paths the workflow
  expects. Snakemake then reuses them instead of producing them again, and
  a results-only user reads the same files any Parquet reader can. The
  release's job duration table is the exception: the workflow rebuilds its
  own from local job records, so the release's is restored beside the
  manifest instead
  ([job duration table](../reference/release_manifest.md#job-duration-table)).

## Only the release command talks to the host

Training, evaluation and plotting read and write local files and know
nothing about Hugging Face. Only `scripts/release/snapshot.py publish` and
`download` contact the host, so running the benchmark never needs a
Hugging Face account, and nothing is published by finishing a run or
saving a snapshot: publishing is always a maintainer's explicit step, after
reviewing the package.

## Compatibility and comparability

A user compares results from their checkout with a release produced by
another commit. Whether that is valid has two separate parts:

- **Compatibility**: whether the release's files still load and fit the
  checkout's pipeline: manifest and bundle versions, table schemas, output
  paths, registered class names. A mismatch shows up as an error, such as
  an unknown `manifest_version`, a bundle class missing from the registry,
  or a `MissingInputException` for a table at an unexpected path.
- **Comparability**: whether results produced now would have been produced
  the same way as the release's. Many changes break it silently, with every
  file still loading:
  - dataset generation or preprocessing, dataset realizations or splits;
  - feature costs;
  - Unmaskers, initializers and acquisition semantics, such as stop
    handling, forced acquisition or budget accounting;
  - the external classifier's architecture or training;
  - method defaults and hyperparameters;
  - evaluation batch sizes, seeds and metrics.

AFABench checks neither for a checkout, and does not require users to run
the producing commits (#36). Instead, a release makes its provenance
visible (each artifact's commit and dirty flag, its resolved configuration
in the artifact's record, the workflow configuration) and maintainers
record both
kinds of change in the [release notes](../reference/release_notes.md); the
user decides.

## Dataset redistribution

Being able to generate a dataset bundle grants no right to publish it:
most tabular and image datasets come from third parties under their own
terms. So every dataset starts as `unreviewed`. Release creation omits
unreviewed and restricted dataset bundles, but keeps models trained on
those datasets and evaluation results. Publication independently checks the
package and refuses any dataset bundle without explicit permission; there
is no override. Reviews remain visible in the manifest even when only
models or results for a dataset are included.

## Limits

- Only artifacts carry provenance records. Transformed tables copy their
  evaluation's record, and merged tables and plots have none, so the
  commits of transformation, aggregation and visualization are not
  known. Merged tables and plots are published but not indexed.
- Bundle content hashes are recorded and link inputs, but neither
  `publish` nor `download` verifies them against the bundles' files.
- "Latest" is decided by the manifests' `created_at`, so finding it reads
  the manifest of every published release.
- Output categories other than payload categories, such as `plots`
  and `merged_results`, are downloaded whole, not by dataset or method.
- Redistribution status applies to dataset bundles only; it is not a legal
  determination about models or results. Maintainers remain responsible
  for the terms applying to those outputs.
- A file's bytes and modification time are kept; nothing else about it is.
