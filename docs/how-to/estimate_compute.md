# Estimate the compute of a run

For a [maintainer](../explanation/user_types.md#maintainer) applying for a
compute allocation, or a
[repository adopter](../explanation/user_types.md#repository-adopter)
checking that their quota suffices: predict the CPU core-hours and GPU-hours
a pipeline invocation will consume before running it. The
[compute estimate](../../CONTEXT.md) comes from the job durations of earlier
runs, either your own or a benchmark release's. Options and output columns:
[`estimate-compute` command](../reference/estimate_compute_command.md).

## 1. Write the invocation you will run

Take the Snakemake arguments of the real run, for example from
[reproduce the full results](reproduce_full_results.md):

```shell
--profile extra/workflow/profiles/config/kdd26 \
--workflow-profile extra/workflow/profiles/<your_site> all
```

Include every `--config`, `--set-resources` and target exactly as you will
submit them: the estimate plans the same jobs, and gives each the CPUs and
GPUs your site profile will request.

## 2. Choose the job durations

- **From your own runs**: by default the estimate reads every job record
  under `extra/output/`, including those of a run that is still going. Use
  this to estimate what is left of a run.
- **From a benchmark release**: download the release's job duration
  table alone:

  ```shell
  uv run python scripts/release/snapshot.py download <release_id> \
      --repo-id <repo_id> --payload-category job_duration_table
  ```

  It lands at `extra/release_job_duration_table.parquet`, beside the
  release manifest, where the pipeline never rewrites it
  ([`download`](../reference/snapshot_command.md#download)). Pass it with
  `--job-durations extra/release_job_duration_table.parquet`.

The durations are used as measured: they are not scaled to your hardware.
The report names the hosts, CPU models and GPU models they came from;
scale the totals yourself if your cluster is faster or slower.

## 3. Run the estimate

```shell
just estimate-compute [--job-durations extra/release_job_duration_table.parquet] \
    --profile extra/workflow/profiles/config/kdd26 \
    --workflow-profile extra/workflow/profiles/<your_site> all
```

Nothing is submitted or run. Jobs whose outputs already exist are not
planned, as in the real run. An invocation Snakemake would reject fails the
same way.

## 4. Read the report

- The first line counts the planned jobs per match level: `exact` jobs
  were measured with the same identity and device; `pooled` jobs use every
  measurement of their stage, method or model, dataset key and device;
  `unestimated` jobs have no measurement there.
- The `Job durations from` line names the source, with the release id and
  scope of a downloaded release's table, and the hosts, CPU models and GPU
  models of the job records that were matched.
- The totals table gives job-hours, core-hours and GPU-hours per pipeline
  stage and device, as a mean and a pessimistic p90, and a `total` row.
  Request the p90 total for a safety margin.
- Unestimated jobs are listed by stage, name, dataset key and device, and
  are not in the totals. Aggregation and visualization jobs write no job
  record and are never estimated; they are short.

A pooled estimate ignores how job duration depends on seeds, hard budgets
and soft-budget parameters, so it is rougher than an exact one.

## 5. Break the estimate down

- Group the totals by other columns with `--by`, for example by method or
  pretrained-model name and dataset key: `--by name --by dataset_key`.
- Write every planned job's estimate to a CSV with `--output
  estimate.csv`, for your own analysis.
- Make the estimate fail when any planned job is unestimated with
  `--strict`, so that an allocation request never rests on an incomplete
  estimate.
