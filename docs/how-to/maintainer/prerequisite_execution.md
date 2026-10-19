# Choose classifier and pretrained-model execution

For a [maintainer](../../explanation/user_types.md#maintainer):
choose CPU or GPU for classifier training and pretraining.

## 1. Set the execution file

Copy `workflow/profiles/execution/default.yaml` to your own file and edit:

```yaml
execution:
  defaults:
    classifier_training: cuda
    pretraining: cpu
    training: cpu
    evaluation: cpu
  methods:
    jafa:
      classifier_training: cpu
      training: cuda
      evaluation: cpu
  pretrained_models:
    pvae: cuda
```

- Set `defaults.classifier_training` for shared external classifiers.
- Set `methods.<name>.classifier_training` only for a method with a
  method-specific classifier.
- Set `pretrained_models.<name>` using the name in `pretrain_mapping`, not
  the script or requesting method. Do not set method-level `pretraining`.
- Use `cpu` or `cuda`; remove `device` from classifier `script_params`,
  `pretrain_params` and invocation-wide overrides.

[Execution keys](../../reference/pipeline_configuration.md#execution).

## 2. Configure resources and inspect the plan

[Configure your site](slurm_integration.md) with the needed allocations.
Set CPU and memory requests for `train_classifier`,
`train_classifier_for_method` and `pretrain_model`; set
[walltime](../../reference/requested_walltime.md) in `site.yaml`.

```shell
workflow/run.py kdd26 --site <your_site> -i \
    --target all_eval_methods execution_file=execution.yaml
```

Verify the classifier and pretraining jobs' Device, CPUs, GPUs and Placement
([job details](../../reference/interactive_preview.md#panes)) and the script
arguments in their [shell commands](../../reference/interactive_preview.md#shell-commands).
Remove `-i` to submit from an authorized host with shared storage.
[Run the pipeline](run_the_pipeline.md).
