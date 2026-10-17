# Add a method to a release

For a [repository adopter](../../explanation/user_types.md#repository-adopter):
add your method to a release's comparison while reusing its existing outputs.

## 1. Download the release and its config

[Choose a release](https://huggingface.co/datasets/afabench/results/tree/main/releases)
and download it into a fresh output root. Use one release per root.

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --all \
    --destination-root output/new_method
```

Check the release README for required
[local dataset sources](../../reference/datasets/README.md) and
[code compatibility](../../explanation/benchmark_releases.md#compatibility-and-comparability)
before reusing its outputs.

## 2. Add your method to the downloaded config

Follow [add a method](add_method.md) to implement and register your method.
Use a name not already in the release, and apply its pipeline configuration
steps to `output/new_method/benchmark.yaml` instead of a pipeline profile:

- Append your method to `methods`.
- Add its `method_options` and `soft_budget_params` entries, including an
  empty parameter list for a hard-budget-only method.
- Add any required `pretrain_mapping` entry.
- Optionally add the method to named `method_sets`; the automatic `all` set
  includes it already.

Leave the released methods' settings, datasets, budgets and realization
indices unchanged.
[Configuration keys](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

## 3. Preview the extended comparison

Open the [interactive preview](../../reference/interactive_preview.md) of
the run:

```shell
workflow/run.py --configfile output/new_method/benchmark.yaml \
    --site <your_site> --interactive-preview
```

Without an interactive terminal, replace `--interactive-preview` with `-n`.

Expect `Run` jobs for your method and missing release coverage, and `Reuse`
for published outputs. Expect `Conditional reuse` for published outputs on
datasets the release omits.

To list your method's jobs with the prerequisites they need, filter by your
method ([filters](../../reference/interactive_preview.md#grouping-filters-and-search)).

## 4. Investigate unexpected retraining

Check every `Run` job of a published method or shared prerequisite before
proceeding:

1. Filter by the status `Run`
   ([filters](../../reference/interactive_preview.md#grouping-filters-and-search)).
2. Under each stage, open each published method's and `Shared
   prerequisites`' jobs and read their
   [reasons](../../reference/interactive_preview.md#reasons).
3. If a job runs because another job of the plan rewrites its inputs,
   follow its `Needs` relations to a `Run` job with a reason of its own
   ([following relations](../../reference/interactive_preview.md#following-relations)).
4. Act on that job's reason:
   - Missing outputs: check that the download into this output root
     included the job's outputs. If the budgets, seeds or realization in
     its details differ from the release's, restore those settings in
     `benchmark.yaml`.
   - Changed params, inputs or code: restore the release's settings for
     the job, or check out the release's compatible code.
5. Compare the job's training and evaluation hard budgets and soft-budget
   parameters with the release's. Expect a new evaluation budget to add
   evaluation jobs only, with the training job reused.
6. Quit the preview, fix the cause and reopen it until only expected jobs
   are `Run`.

If the run stops because a regenerated dataset differs from the release,
fix the data or recompute in a separate output root; do not bypass the
check.

## 5. Run the extended comparison

Run the same configuration without `--interactive-preview`:

```shell
workflow/run.py --configfile output/new_method/benchmark.yaml \
    --site <your_site>
```

Find comparison tables in `output/new_method/merged_results/` and
figures in `output/new_method/plots/`.
