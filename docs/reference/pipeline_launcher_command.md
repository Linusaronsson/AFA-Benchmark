# `workflow/run.py` command

The [pipeline launcher](../../CONTEXT.md) runs the pipeline from a pipeline
profile, an optional site profile and pipeline profile overrides. Steps:
[run the pipeline](../how-to/run_the_pipeline.md).

```shell
workflow/run.py [<pipeline>] [--site <site>] [-n] [--target <rule>]
    [--jobs <n>] [key=value ...] [-- <Snakemake arguments>]
```

It runs on the host's `python3` (3.9 or newer) with the standard library
only, so it works on a login node before any environment is built. It runs
Snakemake from the checkout root, wherever it is started.

## Arguments

| Argument | Meaning |
| --- | --- |
| `<pipeline>` | A pipeline profile under `workflow/profiles/pipeline/`, by name (`all`, `kdd26`). Default `all`. An argument containing `=` is never a pipeline profile. |
| `--site <site>` | A site profile under `workflow/profiles/site/`, by name (`arrhenius`, `examples/mixed-gres`). Without one, jobs run locally. |
| `-n`, `--dry-run` | Plan the run and print its configuration, job counts and commands without running jobs. |
| `--target <rule>` | The rule to run. Default `all`. |
| `--jobs <n>` | How many jobs run at once locally. Refused with `--site`: the site profile sets it. |
| `key=value` | A pipeline profile override, as Snakemake's `--config` takes it (`"methods=[gdfs,dime]"`, `"eval_hard_budgets={cube: [2, 4]}"`). Repeat for several. |
| `-- <Snakemake arguments>` | Passed to Snakemake unchanged, such as `--rerun-incomplete` or `--set-resources`. Not `--config`: give overrides as `key=value` instead. |

A profile name that does not exist fails, listing the existing ones.

## The Snakemake command

The launcher prints the command it runs, then runs it:

```shell
<interpreter> --profile workflow/profiles/pipeline/<pipeline> \
    [--workflow-profile workflow/profiles/site/<site>] <rule> \
    [--jobs <n>] [--dry-run --printshellcmds] [<Snakemake arguments>] \
    [--config <key=value ...> [execution_site_file=workflow/profiles/site/<site>/site.yaml]]
```

- **Interpreter.** `containers/bin/python -m snakemake`, the orchestration
  environment built by `containers/build.sbatch`, when the site's
  `site.yaml` names an `image` in either allocation; otherwise
  `uv run snakemake`.
- **Site file.** With `--site`, `execution_site_file` is added to the
  overrides, so the site's allocations apply whatever overrides are given.

`just estimate-compute` takes the same arguments
([`estimate-compute` command](estimate_compute_command.md)).
