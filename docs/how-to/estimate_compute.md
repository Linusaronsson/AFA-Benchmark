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

The [report](../reference/estimate_compute_command.md#report) gives
job-hours, core-hours and GPU-hours per pipeline stage and device, as a
mean and a pessimistic p90.

- Request the p90 `total` for a safety margin.
- Check the counts of `pooled` and `unestimated` jobs on the first line
  ([match levels](../reference/estimate_compute_command.md#matching)).
  Unestimated jobs are not in the totals; estimate them some other way, or
  first run a few of them to record their job durations.

## 5. Act on failure warnings

For each job type listed under `Failed or timed out before`, before you
launch:

- **Timed out**: raise the time limit of that stage's rule above the limit
  shown, in your site profile's `set-resources` or with `--set-resources
  <rule>:runtime=<minutes>` (rule names:
  [SLURM integration](slurm_integration.md)), and pass the same arguments
  to the estimate and to the real run.
- **Failed**: find the failed job in Snakemake's output of that run, and
  its job record under `extra/output/failed_job_records/`; fix the cause,
  or the jobs will fail again.

## 6. Break the estimate down

- Group the totals by other columns with `--by`, for example by method or
  pretrained-model name and dataset key: `--by name --by dataset_key`.
- Write every planned job's estimate to a CSV with `--output
  estimate.csv`, for your own analysis. Its `failure_history` column flags
  the jobs of the job types the report warns about.
- Make the estimate fail when any planned job is unestimated with
  `--strict`, so that an allocation request never rests on an incomplete
  estimate.
