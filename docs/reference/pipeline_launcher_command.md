# `workflow/run.py` command

The [pipeline launcher](../../CONTEXT.md) runs the pipeline from a pipeline
profile, an optional site profile and pipeline profile overrides. Steps:
[run the pipeline](../how-to/run_the_pipeline.md).

```shell
workflow/run.py [<pipeline>] [--site <site>] [-n | -e [<estimate options>]]
    [--no-soft-budget] [--per-dataset-plots] [--wandb] [--wandb-group <name>] [--target <rule>] [--jobs <n>] [key=value ...] [-- <Snakemake arguments>]
```

It runs on the host's `python3` (3.9 or newer) with the standard library
only, so it works on a login node before any environment is built. It runs
Snakemake from the checkout root, wherever it is started.

## Arguments

| Argument | Meaning |
| --- | --- |
| `-h`, `--help` | Show command-line help and exit. |
| `--list-methods`, `--list-datasets`, `--list-unmaskers`, `--list-initializers` | List shipped names and exit without an environment or running jobs. Flags can be combined. These list all available names, not the selection of a pipeline profile or its overrides. |
| `<pipeline>` | A pipeline profile under `workflow/profiles/pipeline/`, by name (`all`, `kdd26`). Default `all`. An argument containing `=` is never a pipeline profile. |
| `--site <site>` | A site profile under `workflow/profiles/site/`, by name (`arrhenius`, `examples/mixed-gres`). Without one, jobs run locally. |
| `-n`, `--dry-run` | Plan the run and print its configuration, job counts and commands without running jobs. |
| `-e`, `--estimate` | Print the [compute estimate](estimate_compute_command.md) of the run instead of running it; see [below](#the-compute-estimate). Not with `-n`. |
| `--job-durations`, `--by`, `--output`, `--strict` | The [compute estimate's options](estimate_compute_command.md#options), with `-e` only. |
| `--no-soft-budget` | Run only the hard-budget setting: adds the override [`soft_budget_setting=false`](pipeline_configuration.md#soft_budget_setting). Not with a `soft_budget_setting=` override. |
| `--per-dataset-plots` | Add per-dataset Accuracy, Macro F1 and AUROC figures with both classifier rows, alongside usual plots: adds [`per_dataset_plots=true`](evaluation_plots.md). Default off. Not with a `per_dataset_plots=` override. |
| `--wandb` | Log to Weights & Biases: adds the override [`use_wandb=true`](pipeline_configuration.md#use_wandb), and the terminal links each job's W&B run. Needs `WANDB_ENTITY` and `WANDB_PROJECT` set. Not with a `use_wandb=` override. Without `--wandb-group` or a `wandb_group=` override, it also adds [`wandb_group=`](pipeline_configuration.md#wandb_group) a new W&B group, such as `all-20261012T121005Z-k3f9`; so does a `use_wandb=true` override. |
| `--wandb-group <name>` | Log to Weights & Biases as `--wandb` does, every W&B run in the W&B group `<name>`, such as an earlier invocation's to continue it: adds `use_wandb=true wandb_group=<name>`. Not with a `use_wandb=` or `wandb_group=` override. |
| `--target <rule>` | The rule to run. Default `all`. |
| `--jobs <n>` | How many jobs run at once locally. Refused with `--site`: the site profile sets it. |
| `key=value` | A pipeline profile override, as Snakemake's `--config` takes it (`"methods=[gdfs,dime]"`, `"eval_hard_budgets={cube: [2, 4]}"`). Repeat for several. |
| `-- <Snakemake arguments>` | Passed to Snakemake unchanged, such as `--rerun-incomplete` or `--set-resources`. Not `--config`: give overrides as `key=value` instead. |

A profile name that does not exist fails, listing the existing ones.

The method and dataset listings read `workflow/conf/{methods,datasets}/all.yaml`.
The Unmasker and Initializer listings read the config file names under
`conf/components/{unmasker,initializer}/`. For example:

```shell
workflow/run.py --list-methods
workflow/run.py --list-datasets --list-unmaskers --list-initializers
```

## The Snakemake command

The launcher prints the command it runs, then runs it:

```shell
<interpreter> --profile workflow/profiles/pipeline/<pipeline> \
    [--workflow-profile workflow/profiles/site/<site>] <rule> \
    [--jobs <n>] [--dry-run --printshellcmds] [<Snakemake arguments>] \
    [--config <key=value ...> [execution_site_file=workflow/profiles/site/<site>/site.yaml]]
```

- **Interpreter.** `workflow/venvs/bin/python -m snakemake`, the venv of the
  host's architecture built by `workflow/venvs/build.sbatch`, when the
  site's `site.yaml` names a `venv` in either allocation; otherwise
  `uv run snakemake`.
- **Venv check.** When the site's `site.yaml` names venvs, the launcher
  fails before running anything while one of them is missing or was built
  from another `uv.lock`.
- **Site file.** With `--site`, `execution_site_file` is added to the
  overrides, so the site's allocations apply whatever overrides are given.

## The compute estimate

With `-e`, the launcher prints and runs the
[`estimate-compute` command](estimate_compute_command.md) with the other
arguments unchanged, so swapping `-n` for `-e` estimates the same run:

```shell
<interpreter> scripts/compute_estimate/estimate_compute.py <arguments without -e>
```

The interpreter is `workflow/venvs/bin/python`, the venv of the host's
architecture, when the site's `site.yaml` names a `venv`; otherwise
`uv run python`. `just estimate-compute` is the same as `-e` with
`uv run python`.
