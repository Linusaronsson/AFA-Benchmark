# Run the pipeline

For a [maintainer](../../explanation/user_types.md#maintainer).
Run commands from the repository root, after `uv sync`.
[Launcher options](../../reference/pipeline_launcher_command.md).

## 1. Choose the configuration

### Pipeline profile

Choose `all` for all configured datasets and methods, or `kdd26` for the
KDD '26 configuration. Do not assume current code reproduces the submitted
results. To customize a profile, [copy it](#make-a-change-durable).

### Execution file

Without GPUs, pass the CPU execution file:

```shell
workflow/run.py ... execution_file=workflow/profiles/execution/cpu.yaml
```

Otherwise, check the CPU/GPU choices in
`workflow/profiles/execution/default.yaml` or select your own file.
[Per-method execution](mixed_execution.md).

### Site profile

On SLURM, select `--site <your_site>` after
[configuring the site](slurm_integration.md). For mixed architectures,
[build the venvs first](slurm_integration.md#run-on-a-cluster-of-mixed-architectures).

## 2. Run the base command

On a cluster, dry-run from an [authorized submit host](#where-to-run-it):

```shell
workflow/run.py --site <your_site> -n
```

Check the resolved datasets, methods, budgets and job counts. Remove `-n`
to submit. Repeat the command to resume interrupted work.

On a CPU workstation:

```shell
workflow/run.py --jobs 8 execution_file=workflow/profiles/execution/cpu.yaml -n
```

Adjust `--jobs` to available CPUs and memory, then remove `-n` to run.
For a compute estimate, use `-e` instead of `-n`;
[estimate compute](estimate_compute.md).

## Log to Weights & Biases

Log in and set the entity and project before adding `--wandb`:

```shell
uv run wandb login
export WANDB_ENTITY=<entity>
export WANDB_PROJECT=afabench
workflow/run.py ... --wandb
```

With direnv, put the entity in gitignored `.envrc.local` instead; `.envrc`
already sets the project. To join an existing group, replace `--wandb` with
`--wandb-group <group_name>`. Follow the run and group links printed in the terminal.

## Run a subset of methods

```shell
workflow/run.py ... "methods=[gdfs,dime]"
```

Ensure each method has `method_options` and `soft_budget_params` entries in
the profile. To reuse published comparisons, follow
[add a method to a release](../repository_adopter/add_method_to_a_release.md).

## Run a subset of datasets

```shell
workflow/run.py ... "datasets=[cube,mnist]"
```

Ensure each key has `conf/components/dataset_key/<key>.yaml`.

## Change the number of dataset realizations

Specify the complete list of indices, retaining existing indices to include
their results. For two realizations:

```shell
workflow/run.py ... "dataset_realization_indices=[0,1]"
```

To extend published results, follow
[add more seeds to a release](../repository_adopter/add_seeds_to_a_release.md).

## Run a smoke test

```shell
workflow/run.py --jobs 8 \
    "datasets=[cube]" "dataset_realization_indices=[0]" \
    "methods=[random_dummy,gdfs]" smoke_test=true \
    execution_file=workflow/profiles/execution/cpu.yaml
```

Check outputs under `output/smoke/`; do not report smoke metrics as benchmark
results. To test SLURM submissions, use your site and execution file instead.

## Use existing outputs without rerunning their stages

To evaluate trained bundles without generating datasets or training:

```shell
workflow/run.py kdd26 --site <your_site> -n --target all_eval_methods \
    "stages={dataset_generation: false, training: false}"
```

To process existing evaluation tables only:

```shell
workflow/run.py kdd26 --site <your_site> -n \
    "stages={dataset_generation: false, training: false, evaluation: false}"
```

Supply all outputs required from disabled stages. Remove `-n` after checking
the plan. [Stage controls](../../reference/pipeline_configuration.md#stages).

## Select metrics and add per-dataset figures

Set a headline metric with:

```shell
workflow/run.py ... -n "plotting={dataset_metrics: {cube: auroc}}"
```

Add `--per-dataset-plots` for additional Accuracy, Macro F1 and AUROC figures:

```shell
workflow/run.py ... --per-dataset-plots -n
```

Remove `-n` to run. Read figures under `eval_perf/` and, with the flag,
`eval_perf_per_dataset/`. [Plot options and layout](../../reference/evaluation_plots.md).

### Rerun visualization with different metrics

```shell
workflow/run.py ... --per-dataset-plots \
    "stages={dataset_generation: false, training: false, evaluation: false}" \
    "plotting={dataset_metrics: {cube: auroc}}"
```

For restored plots, append
`-- --forcerun plot_eval_perf plot_eval_perf_per_dataset`.
Without `--per-dataset-plots`, force only `plot_eval_perf`.

## Change the hard budgets

Override each dataset explicitly, in selection-cost units:

```shell
workflow/run.py ... "eval_hard_budgets={cube: [2, 4], mnist: [10]}"
```

Retain old entries if you want them in the comparison. Do not change only
`default` for a dataset with its own budget entry.
Check training-budget mappings in `method_options` before running.
[Budget options](../../reference/pipeline_configuration.md#method-option-keys-and-validation).

For soft-budget-only evaluation, empty the hard-budget lists:

```shell
workflow/run.py ... "eval_hard_budgets={cube: [], mnist: []}"
```

## Change the soft-budget parameters

Ensure the dataset is in the method's `soft_budget_datasets`, then supply
its complete list of `[train, eval]` pairs:

```shell
workflow/run.py ... "soft_budget_params={jafa: {cube: [[0.001, null], [0.02, null]]}}"
```

For hard-budget-only evaluation:

```shell
workflow/run.py ... --no-soft-budget
```

To extend a release with either setting, follow
[add budget settings](../repository_adopter/add_budget_settings_to_a_release.md).

## Make a change durable

Copy the profile and each configuration file you need to edit, then repoint
the configfile paths in the copied profile:

```shell
cp -r workflow/profiles/pipeline/all workflow/profiles/pipeline/my_run
cp workflow/conf/soft_budget_params/all.yaml workflow/conf/soft_budget_params/my_run.yaml
# Repoint soft_budget_params/all.yaml to my_run.yaml in my_run/config.yaml.
# Edit workflow/conf/soft_budget_params/my_run.yaml.
workflow/run.py my_run --site <your_site> -n
```

Check that every selected method has the required parameter entries before
using alternative budget files.

## Inspect planned work

- Pass `-n list_jobs=true` to inspect job commands, `device=` arguments and allocations.
- Replace `-n` with `--interactive-preview` to browse the jobs in the
  terminal with their [job status](../../../CONTEXT.md) and its reasons
  ([interactive preview](../../reference/interactive_preview.md)).
- Add `--group-by dataset,method` to the preview to compare methods on each
  dataset. Regroup, filter and search in the preview
  ([controls](../../reference/interactive_preview.md#grouping-filters-and-search)).
- To find what makes a job run, follow its `Needs` relations to a `Run` job
  with a reason of its own
  ([following relations](../../reference/interactive_preview.md#following-relations)).
- Use `--target all_train_classifiers`, `all_pretrain_models`,
  `all_train_methods` or `all_eval_methods` to narrow the target.
- Read detailed logs under `.snakemake/log/` when troubleshooting.
- Check actual partition access and script device support before submission.

## Where to run it

- Use one authorized submit host with access to the site's CPU and GPU
  allocations. Keep the controller alive with `tmux` or the site's approved mechanism.
- Put the repository, environment, inputs and output root on a filesystem
  shared by the submit host and compute nodes.
- Do not attempt cross-cluster dispatch in one invocation.
