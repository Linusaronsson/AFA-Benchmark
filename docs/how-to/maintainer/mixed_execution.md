# Choose per-method training and evaluation execution

For a [maintainer](../../explanation/user_types.md#maintainer):
choose CPU or GPU separately for each method's training and evaluation.

## 1. Set execution choices

Copy `workflow/profiles/execution/default.yaml` to your own execution file,
then edit its defaults and per-method overrides. For example:

```yaml
execution:
  defaults:
    training: cpu
    evaluation: cpu
  methods:
    jafa:
      training: cuda
      evaluation: cpu
```

Use only `cpu` or `cuda`. Choose an evaluation device that supports both the
policy and classifier. Do not put `device` in `method_specific_params` or
combine a global `device` override with `execution`.
[Execution keys and precedence](../../reference/pipeline_configuration.md#execution).

## 2. Configure the computing site

For SLURM, [configure a site profile](slurm_integration.md) with the CPU and
GPU allocations you need. Keep partitions, accounts and GPU requests out of
the execution file and per-rule `set-resources`.

Set CPU counts and memory in the site profile; set
[walltime](../../reference/requested_walltime.md) in `site.yaml`.
Verify account access and device support yourself.

## 3. Check the plan and run

```shell
workflow/run.py kdd26 --site <your_site> -i \
    execution_file=execution.yaml --target all_eval_methods
```

Check each job's Device, CPUs, GPUs and Placement
([job details](../../reference/interactive_preview.md#panes)).
Remove `-i` to submit. Run from an authorized submit host with shared repository, environment,
inputs and output storage.

To evaluate already-trained bundles, also pass
`"stages={dataset_generation: false, training: false}"`.
For a local CPU run, omit `--site` and use
`--jobs 8 execution_file=workflow/profiles/execution/cpu.yaml` instead.

For other stages, see [classifier and pretrained-model execution](prerequisite_execution.md)
and [CPU-only processing](cpu_processing_execution.md).
