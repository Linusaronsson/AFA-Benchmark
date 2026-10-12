# Run the pipeline

For the [maintainer](../explanation/user_types.md#maintainer).

The benchmark is one Snakemake dependency graph, from dataset generation
through classifiers, pretrained models, method training and evaluation to
the final plots. One invocation of its final target, `all`, runs every job
whose inputs are ready, and under SLURM submits CPU and GPU jobs as their
dependencies complete. The pipeline profile decides what the graph contains;
overrides narrow or change it for one run.

Every command on this page runs the
[pipeline launcher](../reference/pipeline_launcher_command.md),
`workflow/run.py`, from the repository root.

## 1. Choose the configuration

A run combines three independent choices, all under `workflow/profiles/`:

| Choice | Selected with | Decides | Needed |
| --- | --- | --- | --- |
| Pipeline profile | `workflow/run.py <name>` | What runs: datasets, methods, budgets | Defaults to `all` |
| Execution file | the override `execution_file=<path>` | Which stages and methods run on CPU or GPU | Defaults to `execution/default.yaml` |
| Site profile | `--site <name>` | Where jobs run: SLURM partitions, accounts, resources | Only on a cluster |

### Pipeline profile

A pipeline profile is a Snakemake profile whose `config.yaml` lists one
config file per configuration group under `workflow/conf/`: datasets,
methods, method options, hard budgets, soft-budget parameters, unmaskers,
classifiers and pretrained models. Together they are the whole pipeline
configuration, described in
[pipeline configuration](../reference/pipeline_configuration.md). Two ship:

| Pipeline profile | Contains |
| --- | --- |
| `all` | Every dataset and method, with their budgets |
| `kdd26` | The same configuration as the KDD '26 submission. The code has changed since, so it does not necessarily reproduce the submitted results |

### Execution file

The default, `workflow/profiles/execution/default.yaml`, puts classifiers,
pretrained models and some methods on GPU. A machine without GPUs, or a
smoke test, must instead run every job on CPU:

```shell
workflow/run.py ... execution_file=workflow/profiles/execution/cpu.yaml
```

> **Without `execution_file=.../cpu.yaml`, a local run requests GPU jobs.**
> Use the CPU file for local runs and smoke tests without a GPU or SLURM.

### Site profile

A site profile maps CPU and GPU execution to one cluster's partitions,
accounts and GPU request syntax. Running locally needs none. To write one,
see [SLURM integration](slurm_integration.md).

## 2. Run the base command

On a SLURM cluster, run on an authorized submit host (see
[where to run it](#where-to-run-it)). First inspect the planned work with a
dry run (`-n`), which prints the resolved configuration and how many jobs
of each rule would run:

```shell
workflow/run.py --site <your_site> -n
```

Swap `-n` for `-e` to print the compute it will take instead
([estimate the compute of a run](estimate_compute.md)). Then remove `-n` to
submit. Adapt `<your_site>` from
`examples/mixed-gres` or `examples/mixed-gpus` as described in
[SLURM integration](slurm_integration.md). On a site whose jobs run in an
image, such as `arrhenius`, build the images and host environments first;
see [run jobs in an image](slurm_integration.md#run-jobs-in-an-image).

On a workstation, use the CPU execution file and say how many jobs run in
parallel:

```shell
workflow/run.py --jobs 8 execution_file=workflow/profiles/execution/cpu.yaml
```

Snakemake only runs jobs whose outputs are missing or out of date, so
running the same command again resumes an interrupted run.

A run prints the resolved configuration, how many jobs of each rule it runs,
its progress (`N of M steps done`), each job's rule and wildcards when it is
submitted (with its SLURM job ID) or starts locally and when it finishes,
and any job's error, but not every job's rule block and shell command. Those are in the log file under
`.snakemake/log` whose path it prints at the end; the override
`list_jobs=true` prints them in the terminal too, without the lines above.

## Log to Weights & Biases

Runs log nothing to W&B unless you add `--wandb`. Log in once with
`uv run wandb login`, and set the entity and project the runs go to.
With [direnv](https://direnv.net/), the checkout's `.envrc` sets
`WANDB_PROJECT=afabench` and loads the gitignored `.envrc.local`, where you
set your entity:

```shell
echo 'export WANDB_ENTITY=<entity>' >> .envrc.local
workflow/run.py ... --wandb
```

Without direnv, export both variables in the shell you run the launcher
from; SLURM jobs inherit them.

Each classifier training, pretraining, training and evaluation job then
starts a [W&B run](../../CONTEXT.md), named after the job's rule and
wildcards, and the terminal links it when the job is submitted and when it
finishes:

```
[Mon Oct 12 10:52:37 2026] Submitted train_method (method=dime, dataset=cube, ...) as SLURM job 4818273: https://wandb.ai/<entity>/afabench/runs/train_method-dime-cube-4818273
```

The run ID ends in the SLURM job ID, or in random characters for a local
job, so a retried job gets a new W&B run. Without either variable the run
fails before any job.

All W&B runs of one invocation join one [W&B group](../../CONTEXT.md), which
the resolved configuration links:

```
  W&B group: https://wandb.ai/<entity>/afabench/groups/all-20261012T121005Z-k3f9
```

A rerun starts a new group. To add its runs to an earlier group, name it
instead of passing `--wandb`:

```shell
workflow/run.py ... --wandb-group all-20261012T121005Z-k3f9
```

The sections below each add overrides (`key=value`) to the base command.
They combine freely; see [combine them](#combine-them).

## Run a subset of methods

List the methods to train and evaluate:

```shell
workflow/run.py ... "methods=[gdfs,dime]"
```

The list replaces the pipeline profile's `methods`. Every method needs an
entry in the pipeline profile's `method_options` and `soft_budget_params`
files. To compare your own
method with published methods without rerunning them, follow
[compare your method with published results](compare_your_method_with_published_results.md)
instead.

## Run a subset of datasets

List the dataset keys to run:

```shell
workflow/run.py ... "datasets=[cube,mnist]"
```

The list replaces the pipeline profile's `datasets`. Each key needs a file
`conf/components/dataset_key/<key>.yaml`.

## Change the number of dataset realizations

Results average over several
[dataset realizations](../../CONTEXT.md). The default is five, indices
`[0,1,2,3,4]`. Choose fewer for a quicker run, or more, such as
`[0,1,2,3,4,5,6,7,8,9]`, for tighter results. Two realizations:

```shell
workflow/run.py ... "dataset_realization_indices=[0,1]"
```

Each index seeds its own dataset realization, so realization 2 is the same
whichever other indices you choose. Each realization is generated by its own
job, so you can start with a few and add more later: going from `[0,1]` to
`[0,1,2]` runs only realization 2's jobs and reuses the rest.

> **Output roots from before per-realization generation.** Dataset
> generation used to be one job per dataset, so an output root written then
> holds one `datasets/<key>/dataset_generation.job_record.json` instead of a
> record per realization. Its dataset realizations stay valid and are not
> regenerated. The old record still enters the job duration table, timing
> all of the key's realizations at once; delete it to keep compute estimates
> per realization.

## Run a smoke test

To check that the pipeline executes, without SLURM or a GPU, run with the CPU
execution file and `smoke_test=true` and a small selection:

```shell
workflow/run.py --jobs 8 \
    "datasets=[cube]" \
    "dataset_realization_indices=[0]" \
    "methods=[random_dummy,gdfs]" \
    smoke_test=true \
    execution_file=workflow/profiles/execution/cpu.yaml
```

Smoke-test settings make every script fast; the resulting metrics only show
that the pipeline runs and are not benchmark results. A smoke test writes
under its own output root, `output/smoke`, never
`output/production`, so a real run afterwards still runs every job.
`smoke_test=true` also works with a site profile and the default execution
file, to check the submissions themselves.

## Use existing outputs without rerunning their stages

To evaluate methods already trained under the output root, without
generating datasets or training anything, turn those pipeline stages off
with the `stages` override:

```shell
workflow/run.py kdd26 --site <your_site> -n --target all_eval_methods \
    "stages={dataset_generation: false, training: false}"
```

`training: false` turns off pretraining, training and classifier training
together. To only transform, aggregate and plot existing evaluation tables,
also turn off evaluation:

```shell
workflow/run.py kdd26 --site <your_site> -n \
    "stages={dataset_generation: false, training: false, evaluation: false}"
```

A stage that is off never schedules a job: the run uses its outputs as they
are, and planning fails if one it needs is missing. The dry run lists the
stages that are off.

## Select metrics and add per-dataset figures

Choose the headline metric for usual plots with the existing plotting config
mapping, not a separate metric flag. Quote the whole override:

```shell
workflow/run.py ... -n "plotting={dataset_metrics: {cube: auroc}}"
```

Add `--per-dataset-plots` to request additional Accuracy, Macro F1 and AUROC
figures for each dataset, with separate external and built-in classifier rows:

```shell
workflow/run.py ... --per-dataset-plots -n
```

Remove `-n` after checking the plan. Usual figures remain in place; the
additional method-set roots appear under `eval_perf_per_dataset/`, beside
`eval_perf/`. See [plot configuration and output layout](../reference/evaluation_plots.md).

### Reevaluate class-only tables

Tables written before the probability-vector contract cannot supply AUROC.
Reevaluate from the existing trained methods; do not infer probabilities from
predicted classes. Keep dataset generation and training off, but evaluation
on, and force the evaluation rule for all selected methods/budgets:

```shell
workflow/run.py ... --per-dataset-plots \
    "stages={dataset_generation: false, training: false}" \
    -- --forcerun eval_method
```

Keep the original profile, dataset realizations, initializer and evaluation
split. This rewrites raw tables and regenerates their downstream outputs.
The trained bundles must satisfy the current classifier output declarations.
No migration of old class-only raw or plotting-ready tables is provided.
[Probability contract](../reference/evaluation_dataframes.md#class-probabilities).

### Rerun visualization with different metrics

Once probability-bearing tables exist, change the mapping while disabling
upstream stages:

```shell
workflow/run.py ... --per-dataset-plots \
    "stages={dataset_generation: false, training: false, evaluation: false}" \
    "plotting={dataset_metrics: {cube: auroc}}"
```

For locally produced outputs with Snakemake job metadata, the forwarded
metric choices are tracked as plot parameters and visualization reruns when
they change. Restored snapshots omit that metadata (ADR 0003), so force
visualization after changing metric choices for restored plots. To force
visualization, append `-- --forcerun plot_eval_perf plot_eval_perf_per_dataset`
with the flag on; with the flag off, force only `plot_eval_perf`.

## Change the hard budgets

The [hard budgets](../../CONTEXT.md) evaluated per dataset come from the
pipeline profile's `eval_hard_budgets` file, for example:

```yaml
eval_hard_budgets:
  default: [5, 10, 15]
  cube: [3, 5, 10]
```

A budget is a number of actions, not features; with an image patch unmasker
one action reveals a patch. Override the budgets of a dataset by naming its
dataset key:

```shell
workflow/run.py ... "eval_hard_budgets={cube: [2, 4], mnist: [10]}"
```

> **Overrides merge mappings key by key.** Datasets you do not name keep
> the pipeline profile's budgets, and `default` only applies to datasets the
> pipeline profile does not list. `"eval_hard_budgets={default: [2]}"`
> leaves `cube` at `[3, 5, 10]` in every shipped pipeline profile. Name each
> dataset you run.

To run only the [soft-budget setting](../../CONTEXT.md), give the datasets
you run no hard budgets:

```shell
workflow/run.py ... "eval_hard_budgets={cube: [], mnist: []}"
```

Methods train with the hard budget they are evaluated with, unless their
`method_options` entry sets `eval_to_train_hard_budget_mapping` (see
[method option keys](../reference/pipeline_configuration.md#method-option-keys-and-validation)).

## Change the soft-budget parameters

[Soft-budget parameters](../../CONTEXT.md) are set per method and dataset key
in the pipeline profile's `soft_budget_params` file. Each entry is a pair
`[train, eval]`; the pipeline trains one model per pair. `null` as the eval
value means evaluation keeps the parameter the method was trained with:

```yaml
soft_budget_params:
  jafa:
    default: []
    cube:
      - [0.005, null]
      - [0.01, null]
```

Override the parameters of one method on one dataset:

```shell
workflow/run.py ... "soft_budget_params={jafa: {cube: [[0.001, null], [0.02, null]]}}"
```

The same merging applies as for hard budgets: other methods, and datasets
you do not name, keep the pipeline profile's values. To run only the
[hard-budget setting](../../CONTEXT.md), turn the soft-budget setting off:

```shell
workflow/run.py ... --no-soft-budget
```

It is short for the override `soft_budget_setting=false`. Pretraining then
also skips the datasets a method ignores in the hard-budget setting.

Most methods train soft-budget runs with no hard budget; methods with
`use_max_hard_budget_when_training_soft_budget` train under the largest one
(see [method option keys](../reference/pipeline_configuration.md#method-option-keys-and-validation)).

## Make a change durable

For a run you will repeat or publish, put the change in config files rather
than on the command line: copy the pipeline profile and edit the copy.

```shell
cp -r workflow/profiles/pipeline/all workflow/profiles/pipeline/my_run
cp workflow/conf/soft_budget_params/all.yaml \
   workflow/conf/soft_budget_params/my_run.yaml
# In profiles/pipeline/my_run/config.yaml, replace soft_budget_params/all.yaml
# with soft_budget_params/my_run.yaml; then edit soft_budget_params/my_run.yaml.
workflow/run.py my_run --site <your_site> -n
```

The same works for any file the pipeline profile lists. To train one more method on
GPU, edit `workflow/profiles/execution/default.yaml`
([per-method execution](mixed_execution.md)). The shipped
`soft_budget_params/{fast,single,none}.yaml` and
`eval_hard_budgets/{fast,single,none}.yaml` are ready-made variants, but the
soft-budget ones only have entries for some methods; a run whose `methods`
include one without an entry fails during planning.

## Combine them

Two dataset realizations of CUBE and MNIST, for two methods, in the
hard-budget setting with custom budgets, on a cluster:

```shell
workflow/run.py --site <your_site> -n \
    "methods=[gdfs,dime]" \
    "datasets=[cube,mnist]" \
    "dataset_realization_indices=[0,1]" \
    "eval_hard_budgets={cube: [3, 5], mnist: [10, 20]}" \
    "soft_budget_params={gdfs: {cube: [], mnist: []}, dime: {cube: [], mnist: []}}"
```

## Inspect planned work

- A dry run prints the resolved configuration (output root, the
  [stages that are off](#use-existing-outputs-without-rerunning-their-stages),
  initializer, dataset realizations, datasets with their unmasker and classifier,
  methods with their classifier and pretrained model, method sets, dataset
  sets, and the
  hard budgets and soft-budget parameters of every method on every dataset),
  then how many jobs of each rule would run. Check them after every
  change to the overrides.
- To list every planned job too, add the override `list_jobs=true`. Each
  job then shows its `resources:` line (`slurm_partition`, `slurm_account`,
  `gpu`, `gres`, `gpu_model`, runtime, CPUs, memory) and, in a dry run, the
  script command, including the `device=` argument passed to computational
  scripts and, on a site whose jobs run in an image, the `apptainer exec`
  prefix.
- Pass a narrower target with `--target`, such as `all_train_classifiers`,
  `all_pretrain_models`, `all_train_methods` or `all_eval_methods`, to
  inspect or run only part of the graph. These are subsets of the same graph, not
  required stages.
- Invalid configuration fails during planning, before any job runs: a method
  without `method_options` or `soft_budget_params`, an unknown dataset key,
  invalid execution choices, incompatible site maps, `device` arguments in
  `method_specific_params`, classifier `script_params` or `pretrain_params`,
  and `set-resources` overrides of allocation resources. Planning cannot
  prove that a partition is available to your account or that a script
  supports a device.

Which hardware each job runs on, and how to change it, is described in
[per-method execution](mixed_execution.md),
[classifiers and pretrained models](prerequisite_execution.md) and
[CPU-only processing](cpu_processing_execution.md); the format is in
[pipeline configuration](../reference/pipeline_configuration.md#execution).

## Where to run it

- Run the launcher on one authorized submit host of a single SLURM cluster that
  can reach both its CPU and GPU partitions. The controller process stays
  alive for the whole run, so use a persistent session (for example `tmux`)
  if your site allows it, or follow site policy for long-running
  controllers. The controller only plans and submits; all jobs, including
  visualization, are submitted to compute nodes.
- The repository, the environment, inputs and the output root must be
  on a filesystem shared by the submit host and all compute nodes.
- Dispatching jobs across separate clusters is not supported. Nothing in the
  workflow checks that the partitions, accounts or GPUs in your site profile
  are available.
- The full benchmark creates many jobs; running it on a workstation is
  possible but not recommended.

The workflow tests in `test/workflow/test_full_reproduction.py` check these
commands against tiny fixtures and a fake `sbatch`.
