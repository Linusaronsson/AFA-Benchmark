# Estimate the compute of a run

For a [maintainer](../../explanation/user_types.md#maintainer) or
[repository adopter](../../explanation/user_types.md#repository-adopter):
estimate CPU core-hours and GPU-hours before submitting.

## 1. Prepare the invocation

Take the intended `workflow/run.py` command, including every override,
`--target` and argument after `--`. Replace `-n`, if present, with `-e`.

## 2. Choose job durations

Use the default records under `output/production/`, or download a release's table:

```shell
uv run python scripts/release/snapshot.py download <release_id> \
    --repo-id afabench/results --payload-category job_duration_table
```

For the downloaded table, pass
`--job-durations output/production/release_job_duration_table.parquet`.
Check [download conflicts](../../reference/snapshot_command.md#overwrite-rule)
before downloading into an occupied root.

## 3. Run the estimate

For local job records:

```shell
workflow/run.py kdd26 --site <your_site> -e
```

For released durations:

```shell
workflow/run.py kdd26 --site <your_site> -e \
    --job-durations output/production/release_job_duration_table.parquet
```

For sites with architecture-specific venvs,
[build them first](slurm_integration.md#run-on-a-cluster-of-mixed-architectures).
Alternatively, run `just estimate-compute` with the invocation's arguments
without `-e` from the uv environment.

## 4. Review totals and warnings

- Use the p90 `total` when requesting an allocation with a safety margin.
- Check the reported hardware before applying durations to a different cluster;
  adjust for hardware differences yourself.
- Cover `unestimated` jobs separately or run representative jobs to collect durations.
- For prior timeouts, increase the relevant
  [walltime](../../reference/requested_walltime.md), then rerun the estimate.
- For prior failures, inspect `output/production/failed_job_records/` and
  fix the cause before submitting.

[Report fields and match levels](../../reference/estimate_compute_command.md).

## 5. Export or narrow the report

Add the options you need:

- `--by name --by dataset_key` to group by method/model and dataset.
- `--output estimate.csv` for per-job estimates.
- `--strict` to fail on unestimated jobs.

[Command options](../../reference/estimate_compute_command.md#options).
