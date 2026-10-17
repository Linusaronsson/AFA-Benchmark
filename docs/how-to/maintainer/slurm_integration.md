# SLURM integration

The pipeline submits jobs through
[Snakemake's SLURM executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
(Snakemake 9.12.0, plugin 1.8.0 in `uv.lock`). A cluster's settings live in a
site profile, `workflow/profiles/site/<site>/`, selected with the pipeline
launcher's `--site <site>`. For the full benchmark command, see
[Run the pipeline](run_the_pipeline.md).

## Portable execution versus site allocation

The benchmark's execution file (`workflow/profiles/execution/default.yaml`) only says
whether each job runs on `cpu` or `cuda`. The site profile maps those two
kinds of execution to SLURM allocations, so the same execution file works on
any cluster:

```text
workflow/profiles/site/<site>/
    config.yaml   # executor, sizing, and the path of site.yaml
    site.yaml     # CPU/GPU allocations and requested walltime lookup
```

## Example profiles

| Profile (under `workflow/profiles/site/`) | Purpose |
| --- | --- |
| `examples/mixed-gres/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gres=gpu:T4:1` |
| `examples/mixed-gpus/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gpus=a100:1` |
| `vera/` | Our team's CPU cluster; its site map has only a CPU allocation, so `cuda` jobs fail before submission |
| `alvis/` | Our team's GPU cluster; `cuda` jobs request `--gres=gpu:T4:1` |
| `arrhenius/` | Our team's mixed cluster, x86_64 CPU nodes and aarch64 GPU nodes; see [Run on a cluster of mixed architectures](#run-on-a-cluster-of-mixed-architectures) |

Except `arrhenius/`, none of these is verified against a live cluster, and
they are unlikely to work for you unchanged.

> **Unverified: whether Alvis accepts CPU-only jobs.** Every graph contains
> CPU-only jobs (dataset generation, transformations, aggregation and
> visualization), so `alvis/site.yaml` maps CPU jobs to the `alvis` partition
> without a GPU request. If Alvis rejects jobs without a GPU, `sbatch` fails
> when such a job is submitted; produce those outputs with a CPU profile
> such as `vera` instead.

## Adapting a profile to your site

1. Copy `site/examples/mixed-gres/` or `site/examples/mixed-gpus/` to `workflow/profiles/site/<site>/`.
2. At the end of `<site>/config.yaml`, replace `examples/mixed-gres` (or
   `examples/mixed-gpus`) with `<site>` in the `execution_site_file` path.
   The pipeline launcher finds `<site>/site.yaml` itself; the line serves
   Snakemake run without it.
3. In `<site>/site.yaml`, set the partitions and accounts, and the GPU request
   in exactly one of the two conventions your site uses:

   ```yaml
   execution_site:
     cpu:
       slurm_partition: <cpu partition>
       slurm_account: <cpu account>
     gpu:
       slurm_partition: <gpu partition>
       slurm_account: <gpu account>
       gres: gpu:<type>:1        # --gres=gpu:<type>:1
       # or: gpu: 1 and optionally gpu_model: <type>   # --gpus=<type>:1
   ```

   Both allocations are needed as soon as any job resolves to them, and every
   graph has CPU-only processing jobs, so every site needs a `cpu`
   allocation. The CPU allocation may not request GPUs. Either allocation may
   add other scheduler flags with `slurm_extra` (for example
   `slurm_extra: --qos=short`), but not GPU requests (`--gres`, `--gpus*`,
   `-G`). Invalid maps fail before submission.
4. Size memory and CPUs in `<site>/config.yaml` with `default-resources`
   and `set-resources` (`mem_mb`, `cpus_per_task`). Set requested walltime in
   `<site>/site.yaml` using the hours-based
   [walltime lookup](../../reference/requested_walltime.md), per rule,
   method/pretrained model and dataset. Remove any profile
   `set-resources.*.runtime` entries that would shadow the table; an explicit
   command-line `--set-resources <rule>:runtime=<minutes>` still wins.
   `default-resources.runtime` remains the fallback for unmatched jobs.
   Rule names you can
   size: `dataset_generation`, `train_classifier`,
   `train_classifier_for_method`, `pretrain_model`, `train_method`,
   `eval_method`, `transform_eval_data`, `merge_eval_perf`,
   `split_by_classifier_type`, `collect_job_records`, `plot_eval_perf`,
   `plot_eval_actions` and `plot_time`. The `vera` profile shows sizes we have
   used. Do not set `slurm_partition`, `slurm_account`, `gpu`, `gres`,
   `gpu_model` or `slurm_extra` per rule; the site map owns them, and
   conflicting rule settings are rejected before submission. `slurm_extra`
   in `default-resources` is rejected as well, since every job's allocation
   would replace it.
5. Check the result with a dry run and read the `resources:` lines.
   `list_jobs=true` makes the dry run list every job, which it otherwise
   leaves out:

   ```shell
   workflow/run.py kdd26 --site <site> -n list_jobs=true
   ```

## Check requested walltime before submission

After editing the site's `walltime` table, inspect a narrowed dry run:

```shell
workflow/run.py --site arrhenius -n list_jobs=true \
  "methods=[dime,l2m_real_feature_prior,l2m_synthetic_feature_prior,l2m_official]" \
  "datasets=[miniboone,miniboone_subset]" "dataset_realization_indices=[0,1]"
```

Read each job's `resources: runtime=...` in minutes. Use measured job
durations with a safety margin, not the illustrative limits in the
[reference example](../../reference/requested_walltime.md#miniboone-example).
Changes affect future submissions only, not running or pending jobs.
To override one rule for a run, append, for example,
`-- --set-resources pretrain_model:runtime=240`.

## Run on a cluster of mixed architectures

When a cluster's nodes differ in CPU architecture, one venv cannot serve
every job, so each architecture gets its own, `.venv-<arch>` in the
checkout, where `<arch>` is the nodes' `uname -m`. The `arrhenius/` profile
does this: its login and CPU nodes are x86_64 and its GPU nodes aarch64
([ADR 0015](../../adr/0015-one-venv-per-architecture.md)). The checkout must be
on a filesystem every node shares.

1. In `<site>/site.yaml`, name each allocation's venv, relative to the
   repository root:

   ```yaml
   execution_site:
     cpu:
       ...
       venv: .venv-x86_64
     gpu:
       ...
       venv: .venv-aarch64
   ```

2. Build the venvs on a node of each architecture your allocations use,
   from the checkout root:

   ```shell
   python3 workflow/venvs/submit_builds.py arrhenius
   ```

   It submits `workflow/venvs/build.sbatch` once per venv, to the
   allocation naming it, with that allocation's account, partition and GPU
   request from `site.yaml`; `--dry-run` prints the `sbatch` commands
   instead. `just build-venvs arrhenius` does the same where `just` is
   installed. It needs only the host's `python3` and `uv`, since no venv
   exists yet. Logs land in `workflow/venvs/build-<job id>.log`.

   Each job syncs `.venv-<arch>` from `uv.lock` without the dev group and
   records the `uv.lock` it synced from as `.venv-<arch>/afabench.uv.lock`.
   Rebuild after any change to `uv.lock`, while no job of the pipeline
   runs: the pipeline launcher refuses to run while a venv is missing or
   built from another lock. The build copies packages out of uv's cache
   rather than hardlinking them, since the cache may sit on another
   filesystem, so each venv counts its files in full against a file quota
   (about 85,000 each).
3. In `<site>/config.yaml`, make each job start Snakemake and its scripts
   from `workflow/venvs/bin/python`, which runs the venv of the node's
   architecture:

   ```yaml
   shared-fs-usage: [persistence, input-output, sources, source-cache, storage-local-copies]
   precommand: export PATH=$PWD/workflow/venvs/bin:$PATH
   ```

4. Run the pipeline as on any site:

   ```shell
   workflow/run.py kdd26 --site arrhenius -n
   ```

   Because the site file names venvs, the pipeline launcher runs Snakemake
   from `workflow/venvs/bin/python` instead of uv. To run a single command
   by hand on any node, start it with `workflow/venvs/bin/python` too.

## A site map is required for SLURM

Submitting to SLURM without a site map fails during planning, before any job
is submitted. Local runs and dry runs need none. Without one, a dry run shows
the profile's `default-resources` partition and account, and `cuda` jobs
declare one generic GPU (`gpu=1`), which local runs can bound with
`--resources gpu=<n>`.

## Related documentation

- [Run the pipeline](run_the_pipeline.md) - the base
  command, how to narrow or change a run, and where to run it
- [Pipeline configuration](../../reference/pipeline_configuration.md) - overview of the pipeline
  and its configuration
