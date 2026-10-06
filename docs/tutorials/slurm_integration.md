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
| `vera/`, `alvis/` | Our team's single-allocation CPU and GPU clusters, without a site map |

None of these is verified against a live cluster, and they are unlikely to
work for you unchanged.

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

   The CPU allocation may not request GPUs, and `slurm_extra` must stay empty
   in both: put any other scheduler settings in the profile's ordinary
   executor options. Invalid maps fail before submission.
4. Size jobs in `<site>/config.yaml` with `default-resources` and
   `set-resources` (`runtime`, `mem_mb`, `cpus_per_task`). Rule names you can
   size: `dataset_generation`, `train_classifier`,
   `train_classifier_for_method`, `pretrain_model`, `train_method`,
   `eval_method`, `transform_eval_data`, `merge_eval_perf`,
   `split_by_classifier_type`, `time_df_with_pretrain`,
   `time_df_without_pretrain`, `merge_time`, `plot_eval_perf`,
   `plot_eval_actions` and `plot_time`. The `vera` profile shows sizes we have
   used. Do not set `slurm_partition`, `slurm_account`, `gpu`, `gres`,
   `gpu_model` or `slurm_extra` per rule; the site map owns them, and
   conflicting rule settings are rejected before submission.
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
> Any `--config` drops `execution_site_file` without an error: GPU jobs are
> then submitted with a generic `--gpus=1` to the cluster's default partition
> and account. Whenever you pass `--config`, also pass
> `execution_site_file=extra/workflow/profiles/<site>/site.yaml`.

## Profiles without a site map

Without `execution_site_file`, every job keeps the profile's
`default-resources` partition and account. CPU jobs clear inherited GPU
requests; GPU jobs request one generic GPU (`--gpus=1`) and the profile's
`slurm_extra` is cleared. Whether the partition provides GPUs is up to you,
so use a site map for mixed CPU/GPU runs.

## Related documentation

- [Reproducing full results](reproduce_full_results.md) - the single
  full-benchmark command and where to run it
- [Pipeline explanation](pipeline_explanation.md) - overview of the pipeline
  and its configuration
