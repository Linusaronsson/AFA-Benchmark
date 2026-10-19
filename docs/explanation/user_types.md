# User types

AFABench serves four kinds of user. They need different parts of the
repository and different parts of a benchmark release, so each how-to
guide is written for one of them.

## Library and benchmark pipeline

AFABench is one project. Its **library** provides reusable Python components;
its **benchmark pipeline** orchestrates configured experiments using those
components. Here, these are usage modes, not separate products. “Framework”
can describe the project overall, but does not distinguish one mode from
the other.

The distribution is named `afabench`, matching `import afabench`. Its wheel
contains the Python library under `src/afabench/`, not the repository-root
`workflow/`, `scripts/`, `conf/` or `data/` trees. Running the benchmark
pipeline needs a repository checkout and the pipeline launcher,
`workflow/run.py`; a library user calls Python interfaces from their own
code without invoking that launcher. Pipeline-related modules in the
library are not standalone replacements for the checkout.

Library installation currently includes pipeline dependencies such as
Snakemake; it is not yet a lightweight dependency subset. Separating those
dependencies is tracked in
[#163](https://github.com/MachineLearning-DecisionMaking/AFABench/issues/163).
A Python distribution is distinct from a
[release package](../../CONTEXT.md#pipeline) of benchmark outputs.

## Results-only user

Wants the published numbers or plots, for example to cite them or to
compare with results produced elsewhere. Downloads tables and plots of a
benchmark release from Hugging Face, reads the tables with any Parquet
reader, and never installs AFABench or runs its pipeline.

- [Download published results](../how-to/results_only_user/download_published_results.md)

## Repository adopter

Has a new AFA method, or a variant of a published one, and wants it
compared with the published methods. Forks the repository, adds the
method, downloads the published methods it compares with into its output
root, then runs its own pipeline profile listing those methods next to its
own. The downloaded release is a cache: the run reuses what it holds, so
only the adopter's method is trained and evaluated, and produces whatever
it lacks, such as a dataset the release does not cover
([ADR 0012](../adr/0012-a-downloaded-release-is-a-cache.md)).

- [Add a method](../how-to/repository_adopter/add_method.md) or
  [add a dataset](../how-to/repository_adopter/add_dataset.md)
- [Add a method to a release](../how-to/repository_adopter/add_method_to_a_release.md)
- [Add a dataset to a release](../how-to/repository_adopter/add_dataset_to_a_release.md)
- [Add more seeds to a release's datasets and methods](../how-to/repository_adopter/add_seeds_to_a_release.md)
- [Add hard budgets or soft-budget parameters to a release](../how-to/repository_adopter/add_budget_settings_to_a_release.md)

## Library user

Wants AFABench's library code, such as its evaluation functions or models,
without adopting its pipeline, for example to evaluate a method inside
their own code base. AFABench is not published on PyPI, so this user
installs the Python distribution from the repository.

- [Install and use the library](../how-to/library_user/use_the_library.md)
- [Evaluation dataframes](../reference/evaluation_dataframes.md#loading-and-plotting)

## Maintainer

Runs the full benchmark and publishes its results as benchmark releases
for the other users.

- [Run the pipeline](../how-to/maintainer/run_the_pipeline.md)
- [Create output snapshots](../how-to/maintainer/create_output_snapshots.md)
- [Restore an output snapshot](../how-to/maintainer/restore_an_output_snapshot.md)
- [Publish a benchmark release](../how-to/maintainer/publish_a_benchmark_release.md)
