# User types

AFABench serves four kinds of user. They need different parts of the
repository and different parts of a benchmark release, so each how-to
guide is written for one of them.

## Results-only user

Wants the published numbers or plots, for example to cite them or to
compare with results produced elsewhere. Downloads tables and plots of a
benchmark release from Hugging Face, reads the tables with any Parquet
reader, and never installs AFABench or runs its pipeline.

- [Download published results](../how-to/download_published_results.md)

## Repository adopter

Has a new AFA method, or a variant of a published one, and wants it
compared with the published methods. Forks the repository, adds the
method, downloads the release's evaluation tables and shared prerequisites,
then trains and evaluates only their own method. Published methods enter
the comparison as reference methods and are never retrained or
re-evaluated (see [reference methods](reference_methods.md)).

- [Add a method](../how-to/add_method.md) or
  [add a dataset](../how-to/add_dataset.md)
- [Compare your method with published results](../how-to/compare_your_method_with_published_results.md)

## Independent evaluator

Wants AFABench's evaluation functions without adopting its pipeline, for
example to evaluate a method inside their own code base. AFABench is not
published as a package, so this user installs it from the repository. No
guide covers this yet; the evaluation code is in `afabench/evaluation/`.

## Maintainer

Runs the full benchmark and publishes its results as benchmark releases
for the other users.

- [Reproduce the full results](../how-to/reproduce_full_results.md)
- [Create output snapshots](../how-to/create_output_snapshots.md)
- [Restore an output snapshot](../how-to/restore_an_output_snapshot.md)
- [Publish a benchmark release](../how-to/publish_a_benchmark_release.md)
