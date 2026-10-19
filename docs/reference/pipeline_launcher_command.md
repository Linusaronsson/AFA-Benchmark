# `workflow/run.py` command

The [pipeline launcher](../../CONTEXT.md) runs the pipeline from a pipeline
profile or benchmark config file, an optional site profile and overrides. Steps:
[run the pipeline](../how-to/maintainer/run_the_pipeline.md).

```shell
workflow/run.py [<pipeline> | --configfile <path>] [--site <site>] [-n | -e [<estimate options>] | -i [--group-by <dimensions>]]
    [--no-soft-budget] [--per-dataset-plots] [--wandb] [--wandb-group <name>] [--target <rule>] [--jobs <n>] [key=value ...] [-- <Snakemake arguments>]
```

It runs on the host's `python3` (3.9 or newer) with the standard library
only, so it works on a login node before any environment is built. Config-file
execution delegates YAML loading and dataset preparation to the selected
Python environment. It runs
Snakemake from the checkout root, wherever it is started.

## Arguments

| Argument | Meaning |
| --- | --- |
| `-h`, `--help` | Show command-line help and exit. |
| `--list-methods`, `--list-datasets`, `--list-unmaskers`, `--list-initializers` | List shipped names and exit without an environment or running jobs. Flags can be combined. These list all available names, not the selection of a pipeline profile or its overrides. |
| `<pipeline>` | A pipeline profile under `workflow/profiles/pipeline/`, by name (`all`, `kdd26`). Default `all`. An argument containing `=` is never a pipeline profile. |
| `--configfile <path>` | A runnable benchmark YAML config instead of a named pipeline profile. Overrides apply last. Downloads include `benchmark.yaml`, with its output root set to the download destination; copies retain that destination unless `output_root=` overrides it. |
| `--site <site>` | A site profile under `workflow/profiles/site/`, by name (`arrhenius`, `examples/mixed-gres`). Without one, jobs run locally. |
| `-n`, `--dry-run` | Plan the run and print its configuration, job counts and commands without running jobs. |
| `-e`, `--estimate` | Print the [compute estimate](estimate_compute_command.md) of the run instead of running it; see [below](#the-compute-estimate). Not with `-n` or `-i`. |
| `--job-durations`, `--by`, `--output`, `--strict` | The [compute estimate's options](estimate_compute_command.md#options), with `-e` only. |
| `-i`, `--interactive-preview` | Browse the jobs the run would run and the jobs whose existing outputs it would reuse in the terminal, the [interactive preview](interactive_preview.md), instead of running it; see [below](#the-interactive-preview). Runs no job, so `-n` changes nothing. Not with `-e`, `--wandb` or `--wandb-group`. Needs a terminal on stdin and stdout; without one it fails and points to `-n`. |
| `--group-by <dimensions>` | The interactive preview's initial [grouping](interactive_preview.md#grouping-filters-and-search): `stage`, `method` and `dataset`, or some of them, each once, comma-separated in order, such as `dataset,method`. Default `stage,method,dataset`. With `-i` only. |
| `--no-soft-budget` | Run only the hard-budget setting: adds the override [`soft_budget_setting=false`](pipeline_configuration.md#soft_budget_setting). Not with a `soft_budget_setting=` override. |
| `--per-dataset-plots` | Add per-dataset Accuracy, Macro F1 and AUROC figures with both classifier rows, alongside usual plots: adds [`per_dataset_plots=true`](evaluation_plots.md). Default off. Not with a `per_dataset_plots=` override. |
| `--wandb` | Log to Weights & Biases: adds the override [`use_wandb=true`](pipeline_configuration.md#use_wandb), and the terminal links each job's W&B run. Needs `WANDB_ENTITY` and `WANDB_PROJECT` set. Not with a `use_wandb=` override or `-i`. Without `--wandb-group` or a `wandb_group=` override, it also adds [`wandb_group=`](pipeline_configuration.md#wandb_group) a new W&B group, such as `all-20261012T121005Z-k3f9`; so does a `use_wandb=true` override. |
| `--wandb-group <name>` | Log to Weights & Biases as `--wandb` does, every W&B run in the W&B group `<name>`, such as an earlier invocation's to continue it: adds `use_wandb=true wandb_group=<name>`. Not with a `use_wandb=` or `wandb_group=` override, or `-i`. |
| `--target <rule>` | The rule to run. Default `all`. |
| `--jobs <n>` | How many jobs run at once locally. Refused with `--site`: the site profile sets it. |
| `key=value` | A pipeline profile override, as Snakemake's `--config` takes it (`"methods=[gdfs,dime]"`, `"eval_hard_budgets={cube: [2, 4]}"`). Repeat for several. |
| `-- <Snakemake arguments>` | Passed to Snakemake unchanged, such as `--rerun-incomplete` or `--set-resources`. Not `--config` or `--configfile`: give overrides as `key=value` and the config file through the launcher instead. |

A profile name that does not exist fails, listing the existing ones. A missing
config file or combining `--configfile` with a named profile also fails.

With `--configfile`, the launcher uses the pipeline Snakefile directly rather
than loading a named profile. When the selected output root contains a release
manifest, it first completes dataset generation, then validates dataset content
and runs the requested target with checked release reuse. A failed preparation
or validation stops the invocation. Disabling dataset generation skips preparation,
not validation. Targeting `all_generate_datasets` only prepares datasets.

A dry run, estimate or interactive preview generates nothing. Each previews
release reuse conditional on regenerated datasets passing validation during
execution; existing datasets are still checked. The interactive preview marks
such jobs [`Conditional reuse`](interactive_preview.md#statuses). The normal
run performs those checks after preparation, before scheduling consumers. No
user-supplied stage switches or reuse flags are needed. Named-profile runs
retain their ordinary dependency behavior.

A benchmark config preserves pipeline-level settings and defaults, not a
self-contained checkout, environment, external data, or transitive Hydra
configuration archive. The release's producer-specific site, execution,
logging and stage switches are not replayed. Choose execution independently.
The manifest remains the original release inventory, not the extended run's.

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

## The interactive preview

With `-i`, the launcher prints and runs the
[interactive preview](interactive_preview.md) with the Snakemake arguments it
resolved for the run, so swapping `-n` for `-i` browses the
same run:

```shell
<interpreter> -m afabench.pipeline_preview [--group-by <dimensions>] -- <Snakemake arguments>
```

The interpreter is `workflow/venvs/bin/python` when the site's `site.yaml`
names a `venv`, as for the compute estimate; otherwise `uv run python`. The
launcher refuses to run it without a terminal on stdin and stdout, and exits
with its exit status: 1 after a planning failure, else 0.
