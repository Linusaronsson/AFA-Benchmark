# Reproducing full results

The full benchmark is one Snakemake dependency graph, from dataset generation
through classifiers, pretrained models, method training and evaluation to
the final plots. One ordinary invocation of its final target, `all`, submits
CPU and GPU jobs to SLURM as their own dependencies complete. You do not
coordinate stages or hardware by hand. The full benchmark creates many jobs, so
running it on a workstation is possible but not recommended.

## Which preset to use

| Preset | Hardware | Use it for |
| --- | --- | --- |
| `--profile extra/workflow/profiles/config/kdd26` | Bundles `extra/workflow/conf/execution/kdd26.yaml`: classifiers, pretrained models and five methods on GPU | Cluster reproduction of the KDD '26 results (the command below) |
| `--profile extra/workflow/profiles/config/all_cluster` | Bundles `extra/workflow/conf/execution/all.yaml`: classifiers, pretrained models and seven methods on GPU | Cluster runs of the full method set ([below](#full-method-set)) |
| `--profile extra/workflow/profiles/config/all` | No execution file: every job runs on CPU | Local smoke tests and development, no GPU or SLURM needed |

> **`config/all` never requests a GPU.** Do not use it for a mixed-device
> cluster run; it would train every method on CPU. Use `config/kdd26` or
> `config/all_cluster`.

## One command

Run from the repository root on an authorized SLURM submit host (see
[Where to run it](#where-to-run-it)). First inspect the planned work with a
dry run (`-n`), which also prints every job's resources and script command:

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/kdd26 \
    --workflow-profile extra/workflow/profiles/<your_site> \
    -n -p all
```

Then remove `-n -p` to submit. The `<your_site>` profile maps CPU and GPU
execution to your cluster's partitions, accounts and GPU request syntax; adapt
it from `mixed-gres` or `mixed-gpus` as described in
[SLURM integration](slurm_integration.md).

> **Repeat the site file whenever you add `--config`.** Snakemake replaces the
> site profile's whole `config` section with the `--config` values, so its
> `execution_site_file` disappears. Submitting without a site map fails before
> any job is submitted, so pass the site file again:
>
> ```shell
> uv run snakemake \
>     --profile extra/workflow/profiles/config/kdd26 \
>     --workflow-profile extra/workflow/profiles/<your_site> \
>     -n -p all \
>     --config \
>       "datasets=[cube]" \
>       execution_site_file=extra/workflow/profiles/<your_site>/site.yaml
> ```
>
> Check `slurm_partition`, `slurm_account` and the GPU request in the dry-run
> `resources:` lines before submitting.

Likewise, `--configfile` on the command line replaces the preset's list of
config files instead of adding to it. To change hardware, for example to train
one method on GPU, copy the preset and its execution file, and edit the copies:

```shell
cp -r extra/workflow/profiles/config/kdd26 extra/workflow/profiles/config/my_run
cp extra/workflow/conf/execution/kdd26.yaml extra/workflow/conf/execution/my_run.yaml
# In profiles/config/my_run/config.yaml, replace execution/kdd26.yaml with
# execution/my_run.yaml; then edit execution/my_run.yaml.
uv run snakemake \
    --profile extra/workflow/profiles/config/my_run \
    --workflow-profile extra/workflow/profiles/<your_site> \
    -n -p all
```

## Inspecting planned work

- `-n -p all` lists every job with its `resources:` line (`slurm_partition`,
  `slurm_account`, `gpu`, `gres`, `gpu_model`, runtime, CPUs, memory) and the
  script command, including the `device=` argument passed to computational
  scripts. The job counts at the end summarize the graph.
- Use a narrower target such as `all_train_classifiers`, `all_pretrain_models`,
  `all_train_methods` or `all_eval_methods` to inspect or run only part of the
  graph. These are subsets of the same graph, not required stages.
- Invalid execution choices, incompatible site maps, `device` arguments in
  `method_specific_params`, classifier `script_params` or `pretrain_params`,
  and `set-resources` overrides of allocation resources fail during planning,
  before any job is submitted. Planning cannot prove that a partition is
  available to your account or that a script supports a device.

## Declaring hardware

`extra/workflow/conf/execution/kdd26.yaml` declares where each computational
job runs. Values are exactly `cpu` and `cuda`:

```yaml
execution:
  defaults:            # per pipeline stage; unspecified ones default to cpu
    classifier: cuda   # external and method-specific classifiers
    pretraining: cuda  # named pretrained models
    training: cpu
    evaluation: cpu
  methods:
    jafa:              # per method: training, evaluation, classifier
      training: cuda
      evaluation: cuda
  pretrained_models:   # per named model in pretrain_mapping (none here)
    pvae: cuda
```

Precedence, resolved independently for every job:

1. `execution.methods.<method>.<training|evaluation|classifier>`, or
   `execution.pretrained_models.<named model>` for pretraining.
2. `execution.defaults.<classifier|pretraining|training|evaluation>`.
3. `cpu`.

Training and evaluation are independent, so a method can train on GPU and
evaluate on CPU. Evaluation runs the classifier too, so declare its end-to-end
needs. Shared external classifiers use only `defaults.classifier`; a pretrained
model shared by several methods uses its own name, never a requesting method's
choice. Dataset generation, transformations, aggregation and plotting always
run on CPU and cannot be configured. Hardware is never inferred from a method's
implementation, taxonomy or the selected method list, and a `cuda` job never
falls back to CPU. `cuda` both passes `device=cuda` to the script and requests
a GPU allocation; `cpu` passes `device=cpu` and requests none.

The shipped files declare the same hardware the former six-invocation workflow
used: classifiers and pretrained models on GPU, and the methods of the removed
`methods/gpu.yaml` (`jafa`, `odin_model_free`, `odin_model_based`, `gdfs`,
`eddi_builtin`, `eddi_external`, `dime`) trained and evaluated on GPU. Partitions
and accounts never belong in these files. Details:
[method training and evaluation](mixed_execution.md),
[classifiers and pretrained models](prerequisite_execution.md),
[CPU-only processing](cpu_processing_execution.md).

## Full method set

`config/all_cluster` bundles the `config/all` scientific files with
`execution/all.yaml`; `config/all` itself is the CPU-only local preset:

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/all_cluster \
    --workflow-profile extra/workflow/profiles/<your_site> \
    -n -p all
```

The `--config` note and the copy-a-preset override above apply here too.

## Where to run it

- Run Snakemake on one authorized submit host of a single SLURM cluster that
  can reach both its CPU and GPU partitions. The controller process stays
  alive for the whole run, so use a persistent session (for example `tmux`)
  if your site allows it, or follow site policy for long-running controllers.
  The controller only plans and submits; all jobs, including plotting, are
  submitted to compute nodes.
- The repository, the `uv` environment, inputs and `extra/output/` must be on
  a filesystem shared by the submit host and all compute nodes.
- Dispatching jobs across separate clusters is not supported. Nothing in the
  workflow checks that the partitions, accounts or GPUs in your site profile
  are available.

## Local smoke test

To check that the pipeline executes, without SLURM or a GPU, run the CPU-only
preset locally with the smoke-test setting and a small selection:

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/all \
    all \
    --jobs 8 \
    --config \
      "datasets=[cube]" \
      "dataset_instance_indices=[0]" \
      "methods=[random_dummy,gdfs]" \
      smoke_test=true \
      use_wandb=false
```

This uses the same graph and execution resolution as a cluster run, with every
job resolved to CPU. Smoke-test settings make the scripts fast; the resulting
metrics only show that the pipeline runs and are not meaningful benchmark
results.

## Migrating from the six-invocation workflow

The former workflow ran six invocations with the `config/cpu_methods` and
`config/gpu_methods` profiles and a global `--config device=...`. Those
profiles and `methods/{cpu,gpu}.yaml` have been removed; the hardware they
expressed is now declared in `execution/{kdd26,all}.yaml`, and the single
invocation above replaces all six. The global `device` option is deprecated:
alone, it still applies to computational jobs with a warning, and combining it
with `execution` in any way is rejected before submission. Remove it from your
commands and config files. Every SLURM submission now needs a site map, and
`slurm_extra` in a profile's `default-resources` is rejected: move partitions,
accounts, GPU requests and other scheduler flags to the profile's `site.yaml`
as described in [SLURM integration](slurm_integration.md). The `vera` (CPU
only) and `alvis` profiles have been migrated this way; `alvis` now requests
its T4 GPU only for `cuda` jobs.

## Verification

The workflow tests run the real orchestration with tiny fixtures, stub
scripts and a fake SLURM `sbatch`, so no cluster, GPU or training is needed:

```shell
uv run pytest test/workflow/test_full_reproduction.py
uv run pytest test/workflow/test_full_reproduction.py -m pipeline
```

The fast tests plan both cluster presets, the local preset and the `alvis`
site map, check that a `--config` without the site file fails before
submission, and capture the real first submissions when the site file is
repeated. The pipeline-marked tests submit the whole tiny graph in one
invocation and check mixed CPU/GPU submissions, one submission per shared
prerequisite, dependency order, script devices and contract arguments, both
presets, and the full local CPU run. The pinned SLURM plugin waits 40 seconds after every
dependency wave, so they take about 15 minutes together. `just qa` remains the
required quality gate.
