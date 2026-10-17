# User types

AFABench serves four kinds of user. They need different parts of the
repository and different parts of a benchmark release, so each how-to
guide is written for one of them.

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
- [Compare your method with published results](../how-to/repository_adopter/compare_your_method_with_published_results.md)
- [Evaluate a release's methods on a new dataset](../how-to/repository_adopter/evaluate_release_methods_on_a_new_dataset.md)
- [Add more seeds to a release's datasets and methods](../how-to/repository_adopter/add_seeds_to_a_release.md)
- [Add hard budgets or soft-budget parameters to a release](../how-to/repository_adopter/add_budget_settings_to_a_release.md)

## Library user

Wants AFABench's library code, such as its evaluation functions or models,
without adopting its pipeline, for example to evaluate a method inside
their own code base. AFABench is not
published as a package, so this user installs it from the repository. No
guide covers this yet; the evaluation code is in `src/afabench/evaluation/`,
and plotting-ready evaluation tables load and plot through the interfaces in
[Evaluation dataframes](../reference/evaluation_dataframes.md#loading-and-plotting).

## Maintainer

Runs the full benchmark and publishes its results as benchmark releases
for the other users.

- [Run the pipeline](../how-to/maintainer/run_the_pipeline.md)
- [Create output snapshots](../how-to/maintainer/create_output_snapshots.md)
- [Restore an output snapshot](../how-to/maintainer/restore_an_output_snapshot.md)
- [Publish a benchmark release](../how-to/maintainer/publish_a_benchmark_release.md)
