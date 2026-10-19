# Configure CPU-only dataset and result processing

For a [maintainer](../../explanation/user_types.md#maintainer):
allocate CPUs and memory for dataset generation, transformation, aggregation
and visualization. Do not add execution-device overrides for these CPU-only stages.

## 1. Configure a CPU allocation

In your [site profile](slurm_integration.md), set `execution_site.cpu` to an
authorized partition and account. Do not request GPUs there, including via
`slurm_extra`, or put `slurm_extra` in `default-resources`.

Set CPU counts and memory with the profile's ordinary resource settings.
Set [walltime](../../reference/requested_walltime.md) in `site.yaml`.
Do not override partition, account or GPU resources per rule.

## 2. Inspect the plan and run

```shell
uv run workflow/run.py kdd26 --site <your_site> -i \
    execution_file=execution.yaml
```

Check that the processing jobs' details show the CPU allocation's Placement,
no GPUs, and suitable CPUs and `mem_mb`
([job details](../../reference/interactive_preview.md#panes)). Remove `-i` to submit.
Use an authorized submit host with shared inputs, outputs and software; do not run processing on a login
node outside site policy.

## 3. Process existing evaluations only

To transform, aggregate and plot existing tables without upstream jobs:

```shell
uv run workflow/run.py kdd26 --site <your_site> \
    "stages={dataset_generation: false, training: false, evaluation: false}"
```

For a local run, omit `--site` and pass
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml`.
[Stage controls](../../reference/pipeline_configuration.md#stages).
