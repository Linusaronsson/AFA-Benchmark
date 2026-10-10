# Run the pipeline

For the [maintainer](../explanation/user_types.md#maintainer).

The benchmark is one Snakemake dependency graph, from dataset generation
through classifiers, pretrained models, method training and evaluation to
the final plots. One invocation of its final target, `all`, runs every job
whose inputs are ready, and under SLURM submits CPU and GPU jobs as their
dependencies complete. The pipeline profile decides what the graph contains;
`--config` narrows or changes it for one run.

## 1. Choose the configuration

A run combines three independent choices, all under `workflow/profiles/`:

| Choice | Selected with | Decides | Needed |
| --- | --- | --- | --- |
| Pipeline profile | `--profile workflow/profiles/pipeline/<name>` | What runs: datasets, methods, budgets | Always |
| Execution file | `--config execution_file=<path>` | Which stages and methods run on CPU or GPU | Defaults to `execution/default.yaml` |
| Site profile | `--workflow-profile workflow/profiles/site/<name>` | Where jobs run: SLURM partitions, accounts, resources | Only on a cluster |

### Pipeline profile

A pipeline profile is a Snakemake profile whose `config.yaml` lists one
config file per configuration group under `workflow/conf/`: datasets,
methods, method options, hard budgets, soft-budget parameters, unmaskers,
classifiers and pretrained models. Together they are the whole pipeline
configuration, described in
[pipeline configuration](../reference/pipeline_configuration.md). Two ship:

| Pipeline profile | Contains |
| --- | --- |
| `pipeline/all` | Every dataset and method, with their budgets |
| `pipeline/kdd26` | The same configuration as the KDD '26 submission. The code has changed since, so it does not necessarily reproduce the submitted results |

### Execution file

The default, `workflow/profiles/execution/default.yaml`, puts classifiers,
pretrained models and some methods on GPU. A machine without GPUs, or a
smoke test, must instead run every job on CPU:

```shell
--config execution_file=workflow/profiles/execution/cpu.yaml
```

> **Without `execution_file=.../cpu.yaml`, a local run requests GPU jobs.**
> Use the CPU file for local runs and smoke tests without a GPU or SLURM.

### Site profile

A site profile maps CPU and GPU execution to one cluster's partitions,
accounts and GPU request syntax. Running locally needs none. To write one,
see [SLURM integration](slurm_integration.md).

## 2. Run the base command

On a SLURM cluster, run from the repository root on an authorized submit host
(see [where to run it](#where-to-run-it)). First inspect the planned work
with a dry run (`-n`), which also prints every job's resources and script
command:

```shell
uv run snakemake \
    --profile workflow/profiles/pipeline/all \
    --workflow-profile workflow/profiles/site/<your_site> \
    -n -p all
```

Then remove `-n -p` to submit. Adapt `<your_site>` from
`site/examples/mixed-gres` or `site/examples/mixed-gpus` as described in
[SLURM integration](slurm_integration.md).

On a workstation, use the CPU execution file and say how many jobs run in
parallel:

```shell
uv run snakemake \
    --profile workflow/profiles/pipeline/all \
    all \
    --jobs 8 \
    --config execution_file=workflow/profiles/execution/cpu.yaml
```

Snakemake only runs jobs whose outputs are missing or out of date, so
running the same command again resumes an interrupted run.

> **Repeat the site file whenever you add `--config`.** Snakemake replaces
> the site profile's whole `config` section with the `--config` values, so
> its `execution_site_file` disappears and the submission fails before any
> job is submitted. Every cluster command with `--config` therefore ends
> with:
>
> ```shell
>     --config \
>       ... \
>       execution_site_file=workflow/profiles/site/<your_site>/site.yaml
> ```

The sections below each add `--config` values to the base command. They
combine freely; see [combine them](#combine-them).

## Run a subset of methods

List the methods to train and evaluate:

```shell
    --config "methods=[gdfs,dime]"
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
    --config "datasets=[cube,mnist]"
```

The list replaces the pipeline profile's `datasets`. Each key needs a file
`conf/components/dataset_key/<key>.yaml`.

## Change the number of dataset realizations

Results average over several
[dataset realizations](../../CONTEXT.md). The default is five, indices
`[0,1,2,3,4]`. Choose fewer for a quicker run, or more, such as
`[0,1,2,3,4,5,6,7,8,9]`, for tighter results. Two realizations:

```shell
    --config "dataset_realization_indices=[0,1]"
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
uv run snakemake \
    --profile workflow/profiles/pipeline/all \
    all \
    --jobs 8 \
    --config \
      "datasets=[cube]" \
      "dataset_realization_indices=[0]" \
      "methods=[random_dummy,gdfs]" \
      smoke_test=true \
      use_wandb=false \
      execution_file=workflow/profiles/execution/cpu.yaml
```

Smoke-test settings make every script fast; the resulting metrics only show
that the pipeline runs and are not benchmark results. A smoke test writes
under its own output root, `output/smoke`, never
`output/production`, so a real run afterwards still runs every job.
`smoke_test=true` also works with a site profile and the default execution
file, to check the submissions themselves.

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
    --config "eval_hard_budgets={cube: [2, 4], mnist: [10]}"
```

> **`--config` merges mappings key by key.** Datasets you do not name keep
> the pipeline profile's budgets, and `default` only applies to datasets the
> pipeline profile does not list. `"eval_hard_budgets={default: [2]}"`
> leaves `cube` at `[3, 5, 10]` in every shipped pipeline profile. Name each
> dataset you run.

To run only the [soft-budget setting](../../CONTEXT.md), give the datasets
you run no hard budgets:

```shell
    --config "eval_hard_budgets={cube: [], mnist: []}"
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
    --config "soft_budget_params={jafa: {cube: [[0.001, null], [0.02, null]]}}"
```

The same merging applies as for hard budgets: other methods, and datasets
you do not name, keep the pipeline profile's values. To run only the
[hard-budget setting](../../CONTEXT.md), give each method you run an empty
list for each dataset you run:

```shell
    --config "soft_budget_params={jafa: {cube: []}, gdfs: {cube: []}}"
```

Most methods train soft-budget runs with no hard budget; methods with
`use_max_hard_budget_when_training_soft_budget` train under the largest one
(see [method option keys](../reference/pipeline_configuration.md#method-option-keys-and-validation)).

## Make a change durable

For a run you will repeat or publish, put the change in config files rather
than on the command line. `--configfile` on the command line replaces the
pipeline profile's list of config files instead of adding to it, so copy the
pipeline profile and edit the copy:

```shell
cp -r workflow/profiles/pipeline/all workflow/profiles/pipeline/my_run
cp workflow/conf/soft_budget_params/all.yaml \
   workflow/conf/soft_budget_params/my_run.yaml
# In profiles/pipeline/my_run/config.yaml, replace soft_budget_params/all.yaml
# with soft_budget_params/my_run.yaml; then edit soft_budget_params/my_run.yaml.
uv run snakemake \
    --profile workflow/profiles/pipeline/my_run \
    --workflow-profile workflow/profiles/site/<your_site> \
    -n -p all
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
uv run snakemake \
    --profile workflow/profiles/pipeline/all \
    --workflow-profile workflow/profiles/site/<your_site> \
    -n -p all \
    --config \
      "methods=[gdfs,dime]" \
      "datasets=[cube,mnist]" \
      "dataset_realization_indices=[0,1]" \
      "eval_hard_budgets={cube: [3, 5], mnist: [10, 20]}" \
      "soft_budget_params={gdfs: {cube: [], mnist: []}, dime: {cube: [], mnist: []}}" \
      execution_site_file=workflow/profiles/site/<your_site>/site.yaml
```

## Inspect planned work

- `-n -p` lists every job with its `resources:` line (`slurm_partition`,
  `slurm_account`, `gpu`, `gres`, `gpu_model`, runtime, CPUs, memory) and
  the script command, including the `device=` argument passed to
  computational scripts and, on a site whose jobs run in an image, the
  `apptainer exec` prefix. The job counts at the end summarize the graph.
  Check them after every `--config` change.
- Replace `all` with a narrower target such as `all_train_classifiers`,
  `all_pretrain_models`, `all_train_methods` or `all_eval_methods` to inspect
  or run only part of the graph. These are subsets of the same graph, not
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

- Run Snakemake on one authorized submit host of a single SLURM cluster that
  can reach both its CPU and GPU partitions. The controller process stays
  alive for the whole run, so use a persistent session (for example `tmux`)
  if your site allows it, or follow site policy for long-running
  controllers. The controller only plans and submits; all jobs, including
  visualization, are submitted to compute nodes.
- The repository, the `uv` environment, inputs and the output root must be
  on a filesystem shared by the submit host and all compute nodes.
- Dispatching jobs across separate clusters is not supported. Nothing in the
  workflow checks that the partitions, accounts or GPUs in your site profile
  are available.
- The full benchmark creates many jobs; running it on a workstation is
  possible but not recommended.

The workflow tests in `test/workflow/test_full_reproduction.py` check these
commands against tiny fixtures and a fake `sbatch`.
