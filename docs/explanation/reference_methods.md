# Reference methods

A reference method is a published method whose plotting-ready evaluation
tables are restored from a benchmark release and compared with locally
produced methods, without the workflow ever training, evaluating or
transforming it. Repository adopters list published methods under
`reference_methods` and their own under `methods`
([how](../how-to/compare_your_method_with_published_results.md)).

## Why published methods cannot be in `methods`

Default aggregation enumerates the evaluations of every method in
`methods`, and Snakemake schedules the upstream work of each one. Listing
a published method in `methods`, even with all its tables downloaded,
therefore retrains and re-evaluates it:

- the time plot of the `all` target needs every method's `train_time.txt`
  and `eval_time.txt`, which only training and evaluation write;
- a restored table is rebuilt whenever a prerequisite upstream of it is
  newer, or is missing and produced again.

A reference method's plotting-ready tables are plain input files: the
transformation rule does not match reference methods, so Snakemake has no
rule to explore upstream of them. That is why a dry run lists none of
their jobs, and why a missing table fails the plan with a
`MissingInputException` instead of being produced by evaluating the
published method locally.

The workflow never writes a reference method's tables. It does rebuild
the merged tables and plots of every method set that contains a local
method, so downloaded `merged_results` and `plot_results` of those method
sets are replaced by the comparison; method sets of only reference methods
are skipped and stay as downloaded.

## How rows stay comparable

- **No duplicate rows.** Each table is enumerated once, at its own path,
  and a method cannot be in both `methods` and `reference_methods`.
- **Hard and soft budgets stay apart.** They are separate tables with
  separate `eval_hard_budget` and soft-budget-parameter columns, plotted in
  separate hard-budget and soft-budget plots.
- **Built-in and external classifiers stay apart.** Each row records its
  `classifier`, and the comparison is split by classifier type before
  plotting.
- **Other evaluation splits and initializers are never mixed in.** Tables
  are looked up under the configuration's `eval_split-*` and
  `initializer-*` folders, so a release made with other settings fails the
  plan instead of being combined.

## Limits

- The comparison covers only evaluations the release has. Datasets,
  dataset instances, budgets or splits the release lacks fail the plan
  until they are removed from the configuration.
- The time plot covers local methods only: published methods' time
  records are not downloaded with their tables.
- Whether local methods are evaluated with the published external
  classifier depends on downloading it; the dry run lists
  `train_classifier` when it is missing.
- Whether the checkout is compatible with the release's commit is the
  user's judgement ([why](benchmark_releases.md#compatibility-and-comparability)).
