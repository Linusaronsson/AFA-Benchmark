# Independent classifier and pretrained-model execution

Classifier training and named pretrained models now use the same execution/site
mapping as [method training and evaluation](mixed_execution.md). This extends the
method-routing slice described there; CPU-only processing is a separate slice.
Use the ordinary `pipeline.smk` invocation, not separate hardware-specific runs.
Keep existing scientific configuration files and add an execution YAML file:

```yaml
execution:
  defaults:
    classifier: cuda
    pretraining: cpu
    training: cpu
    evaluation: cpu
  methods:
    jafa:
      classifier: cpu
      training: cuda
      evaluation: cpu
  pretrained_models:
    pvae: cuda
```

- External classifiers shared across methods use `defaults.classifier`, never a
  downstream method's training/evaluation choice. They retain one native bundle
  per dataset and initializer, trained on dataset instance 0 with seed 0.
- Method-specific classifiers use `methods.<method name>.classifier`, then
  `defaults.classifier`. The existing `method_options.<name>.classifier` script
  selection and parameters still decide which variant is trained, not hardware.
  A `classifier` override for a selected method without a method-specific
  classifier is rejected.
  A CPU policy does not imply that its classifier should run on CPU.
- Pretraining uses `pretrained_models.<named model>`, then `defaults.pretraining`.
  The key is the name in `pretrain_mapping`, not the script name or requesting
  method; a name missing from `pretrain_mapping` is rejected. A method-level
  `pretraining` override is rejected. Two methods sharing
  one pretrained model still depend on the same bundle, regardless of method
  order. The classifier input remains the external classifier, even if a
  downstream method uses a method-specific classifier.
- Each unspecified pipeline stage defaults to `cpu`. Without `execution`, the deprecated
  global `device` still applies with a warning; it cannot coexist with
  `execution`.
  Values are exactly `cpu` and `cuda`, with no hardware inference or fallback.

The same resolved choice supplies the existing script's `device` argument and
allocation intent. `execution_site_file` in the cluster profile maps CPU/GPU to
partitions, accounts and GPU request syntax. CPU prerequisites explicitly clear
GPU resources, including inherited profile defaults. CPU count, memory and
runtime remain the profile's ordinary sizing settings: configure `train_classifier`,
`train_classifier_for_method`, and `pretrain_model` under `set-resources` as needed.
Existing Vera/Alvis sizing and scientific hyperparameters are not rewritten.

Selected invalid choices, incompatible site mappings and final allocation-resource
conflicts fail during DAG planning, before any submission. Classifier
`script_params` and model `pretrain_params` may not set `device` (including Hydra
`+device` forms); use the execution mapping instead. Unselected model choices
are not resolved. Validation does not prove cluster availability or script support
for a chosen device.

## Invoke and inspect

With the original scientific config files combined in `benchmark.yaml`:

```sh
uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk \
  --workflow-profile extra/workflow/profiles/mixed-gres \
  --configfile benchmark.yaml execution.yaml -n -p all_eval_methods
```

Remove `-n` to submit from one authorized environment with shared filesystem and
access to both allocation types. The same graph schedules prerequisites according
to their dependencies. Dataset instances, seeds, Initializers, Unmaskers, budget
settings, method-owned scripts, plain pretraining/training contract, native
bundle directories and `pretrain_time.txt` outputs are unchanged. Classifier
scripts retain their existing Hydra arguments. For the single full-benchmark
command, see [Reproducing full results](reproduce_full_results.md).

## Boundary verification

```sh
uv run pytest test/workflow/test_prerequisite_execution.py
uv run pytest test/workflow/test_prerequisite_execution.py -m pipeline
```

The fast fixtures invoke real orchestration to check independent devices,
order-independent shared pretraining, preserved contract arguments and invalid
selected prerequisites before any submissions. The pipeline fixtures reuse the
shared fake SLURM harness and capture actual executor submissions for both
illustrative profiles (Snakemake 9.12.0 / SLURM plugin 1.8.0). One graph includes
external and method-specific classifiers, two methods sharing one named model,
another named model, method training and evaluation, with differing prerequisite
allocations despite CPU downstream choices and inherited GPU defaults. Fixtures
use temporary bundles and cheap script-boundary stubs, not real training or SLURM.
`just qa` remains mandatory in addition to these explicit submission checks.
