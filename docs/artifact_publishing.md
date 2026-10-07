# Publishing benchmark releases

Agreed design for [GitHub issue #16](https://github.com/Linusaronsson/AFA-Benchmark/issues/16).
The implementation spec is [#36](https://github.com/Linusaronsson/AFA-Benchmark/issues/36),
with tickets #37–#41. This is a design and implementation checklist, not
documentation of an existing publishing feature.

## User journeys

1. **Results-only researcher:** downloads evaluation dataframes and optionally
   plots, without installing AFABench or adopting its pipeline.
2. **Benchmark adopter:** forks the repository, implements their method, reuses
   published baseline results and shared inputs, and runs only the missing
   work needed to add their method to comparison plots.
3. **Independent evaluator:** may install AFABench to use its evaluation
   functions without adopting the repository's pipeline. Making the package
   pip-installable is separate from the publishing work.

## Release contract

A benchmark release is a curated collection produced with an identified code
commit and pipeline configuration. Hugging Face is the publication host.

- Record the producing code commit and the configuration used to run the
  pipeline, including resolved settings needed to reproduce it.
- List coverage explicitly: datasets, dataset instances, methods, splits,
  acquisition settings, classifier variants, and available output categories.
- Preserve release identities so users can request an older release.
- Default downloads to the latest full-benchmark release. Curated partial
  releases are allowed but remain distinct from that default.
- Resolve the chosen release once for a download; do not silently assemble
  outputs from multiple releases.
- Users judge whether downloaded results are compatible with their checkout.
  Do not require them to use the producing commit, but make provenance visible.
- Review changes that can affect benchmark results carefully. File/API
  compatibility does not establish scientific comparability. Changes to data,
  splits, preprocessing, acquisition semantics, classifiers, and metrics may
  matter even when existing files still load.

## Published contents

Make the following separately downloadable where available and redistribution
is permitted:

- Plotting-ready transformed evaluation Parquet tables.
- Raw evaluation Parquet tables, including episode acquisition histories.
- Generated plots.
- Native dataset/split bundles, external-classifier bundles, pretrained-model
  bundles, and AFA-method bundles.
- Release provenance describing the inputs and settings behind these outputs.

Retain native AFABench bundles even when publishing also creates HF-friendly
representations. Conversions must preserve the underlying data/results;
consumers should not need to reverse a conversion to reuse the pipeline.
Intermediate training checkpoints are not the primary reusable model format.

## Architectural boundary

Train, evaluation, and plotting scripts must remain unaware of Hugging Face.
They consume and produce local files.

A separate **publish script** takes pipeline outputs, prepares a publication
package and provenance, and may convert native outputs for convenient HF use.
Preparing the package can be automated; public publication is an explicit
maintainer action after review. A completed pipeline run must not automatically
publish smoke-test, incomplete, or restricted outputs.

A separate **download command** retrieves a selected release and restores native
outputs to the pipeline's expected local paths. This allows Snakemake to reuse
existing outputs and run only missing work. It must:

- Select outputs relevant to the requested datasets, methods, and settings.
- Offer an explicit download-everything option.
- Avoid silently overwriting existing local outputs.
- Keep publication provenance alongside downloaded files.
- Report unavailable requested outputs rather than silently substituting them
  or triggering retraining as part of downloading.

Adopters should not need baseline method bundles merely to plot their method
against published baseline results. Shared evaluation inputs, such as exact
splits and external classifiers, can be useful independently of those bundles.

## Current code constraints

Reconnaissance identified the following implementation considerations:

- `scripts/misc/transform_eval_data_pipeline.py` produces transformed tables;
  `scripts/misc/merge_dataframes.py` can combine compatible tables for plotting.
- Transformed tables are prediction/cost rows, not just aggregate scores.
- Some provenance, including dataset-instance index and evaluation split, is
  not preserved explicitly in transformed tables. Release metadata must retain
  this information rather than relying on table columns alone.
  `docs/adr/0002-provenance-recorded-in-artifacts.md` decides how bundles and
  evaluation tables will carry their own provenance record and identity
  columns, which the release manifest reads instead of parsing paths.
- Raw evaluation `idx` is batch-local, not a stable cross-run instance identity.
  Do not promise paired instance-level comparisons using that field.
- Default workflow aggregation enumerates baseline inputs and can cause
  missing upstream work to run. Verify that the download-and-reuse journey
  satisfies the relevant dependencies without retraining published baselines.
- Actual publication sizes and dataset redistribution rights still need an
  inventory; neither is established by this design. The release manifest
  records every dataset as unreviewed until a maintainer reviews it
  ([`release_manifest.md`](release_manifest.md#dataset-redistribution)).

## Implementation checklist

1. Inventory the official pipeline's outputs, dependencies, sizes, and
   redistribution constraints. #39 adds the `inventory` command, smoke-scale
   measurements and the dataset redistribution review; production sizes
   and every dataset's review remain open. See
   [`release_manifest.md`](release_manifest.md#smoke-scale-inventory).
2. Define a release manifest that maps each downloadable file to its producing
   run/configuration and records coverage and provenance. #63 defines it
   for release identity, workflow configuration, resolved settings,
   evaluation tables and coverage; #39 adds native bundles, payload
   categories and their dependencies; see
   [`release_manifest.md`](release_manifest.md). Bundles are identified by
   their own provenance records once ADR 0002 lands.
3. Implement local package preparation and validation, followed by explicit
   HF publication. Preserve native formats; add HF conversions where useful.
   #38 publishes a snapshot with its manifest and downloads one named
   release; see [`release_publishing.md`](release_publishing.md).
4. Implement release selection and selective downloading into native pipeline
   locations, including existing-file handling and missing-output reporting.
   #40 resolves the latest full or a pinned release once per download and
   selects payloads by category and coverage; see
   [`release_publishing.md`](release_publishing.md#downloading-a-release-into-a-checkout).
5. Verify both main journeys: independent Parquet/plot downloads, and a fork
   adding one method without retraining or reevaluating published baselines.
   No real benchmark results are currently available: use small smoke runs,
   synthetic fixtures, and fake HF transport. Preserve smoke provenance and
   keep test packages distinct from official scientific releases.
6. Document release provenance, download examples, raw versus plotting-ready
   tables, and result-affecting changes between releases.
7. Run `just qa` for implementation changes.

HF repository layout, namespace, manifest schema, command syntax, and specific
conversion formats remain implementation decisions. No additional archival
host or mandatory HF-native model/dataset API integration is required.
