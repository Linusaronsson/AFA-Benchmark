# SLURM integration

The pipeline submits jobs through
[Snakemake's SLURM executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
(Snakemake 9.12.0, plugin 1.8.0 in `uv.lock`). A cluster's settings live in a
workflow profile, `extra/workflow/profiles/<site>/`, passed with
`--workflow-profile`. For the full benchmark command, see
[Reproducing full results](reproduce_full_results.md).

## Portable execution versus site allocation

The benchmark's execution files (`extra/workflow/conf/execution/`) only say
whether each job runs on `cpu` or `cuda`. The site profile maps those two
kinds of execution to SLURM allocations, so the same execution file works on
any cluster:

```text
extra/workflow/profiles/<site>/
    config.yaml   # executor, sizing, and the path of site.yaml
    site.yaml     # CPU and GPU partitions, accounts and GPU request syntax
```

## Example profiles

| Profile | Purpose |
| --- | --- |
| `mixed-gres/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gres=gpu:T4:1` |
| `mixed-gpus/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gpus=a100:1` |
| `vera/` | Our team's CPU cluster; its site map has only a CPU allocation, so `cuda` jobs fail before submission |
| `alvis/` | Our team's GPU cluster; `cuda` jobs request `--gres=gpu:T4:1` |

None of these is verified against a live cluster, and they are unlikely to
work for you unchanged.

> **Unverified: whether Alvis accepts CPU-only jobs.** Every graph contains
> CPU-only jobs (dataset generation, transformations, aggregation and
> visualization), so `alvis/site.yaml` maps CPU jobs to the `alvis` partition
> without a GPU request. If Alvis rejects jobs without a GPU, `sbatch` fails
> when such a job is submitted; produce those outputs with a CPU profile
> such as `vera` instead.

## Adapting a profile to your site

1. Copy `mixed-gres/` or `mixed-gpus/` to `extra/workflow/profiles/<site>/`.
2. In `<site>/config.yaml`, point `execution_site_file` at your copy:

   ```yaml
   config:
     execution_site_file: extra/workflow/profiles/<site>/site.yaml
   ```

   The path is relative to the directory you run Snakemake from (the
   repository root); use an absolute path otherwise.
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
4. Size jobs in `<site>/config.yaml` with `default-resources` and
   `set-resources` (`runtime`, `mem_mb`, `cpus_per_task`). Rule names you can
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
5. Check the result with a dry run and read the `resources:` lines:

   ```shell
   uv run snakemake \
       --profile extra/workflow/profiles/config/kdd26 \
       --workflow-profile extra/workflow/profiles/<site> \
       -n -p all
   ```

`execution_site` can instead be given in a config file, but not together with
`execution_site_file`. Do not put the nested `execution_site` mapping in a
profile's `config` section: this Snakemake/plugin version does not preserve
nested profile config in the remote job wrapper.

> **`--config` on the command line replaces the site profile's `config`.**
> Any `--config` drops `execution_site_file`, and the submission then fails
> before any job is submitted. Whenever you pass `--config`, also pass
> `execution_site_file=extra/workflow/profiles/<site>/site.yaml`.

## A site map is required for SLURM

Submitting to SLURM without a site map fails during planning, before any job
is submitted. Local runs and dry runs need none. Without one, a dry run shows
the profile's `default-resources` partition and account, and `cuda` jobs
declare one generic GPU (`gpu=1`), which local runs can bound with
`--resources gpu=<n>`.

## Related documentation

- [Reproducing full results](reproduce_full_results.md) - the single
  full-benchmark command and where to run it
- [Pipeline configuration](../reference/pipeline_configuration.md) - overview of the pipeline
  and its configuration
