# SLURM integration

The pipeline submits jobs through
[Snakemake's SLURM executor plugin](https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html)
(Snakemake 9.12.0, plugin 1.8.0 in `uv.lock`). A cluster's settings live in a
workflow profile, `workflow/profiles/site/<site>/`, passed with
`--workflow-profile`. For the full benchmark command, see
[Run the pipeline](run_the_pipeline.md).

## Portable execution versus site allocation

The benchmark's execution file (`workflow/profiles/execution/default.yaml`) only says
whether each job runs on `cpu` or `cuda`. The site profile maps those two
kinds of execution to SLURM allocations, so the same execution file works on
any cluster:

```text
workflow/profiles/site/<site>/
    config.yaml   # executor, sizing, and the path of site.yaml
    site.yaml     # CPU and GPU partitions, accounts and GPU request syntax
```

## Example profiles

| Profile (under `workflow/profiles/site/`) | Purpose |
| --- | --- |
| `examples/mixed-gres/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gres=gpu:T4:1` |
| `examples/mixed-gpus/` | Illustrative mixed CPU/GPU site requesting GPUs as `--gpus=a100:1` |
| `vera/` | Our team's CPU cluster; its site map has only a CPU allocation, so `cuda` jobs fail before submission |
| `alvis/` | Our team's GPU cluster; `cuda` jobs request `--gres=gpu:T4:1` |
| `arrhenius/` | Our team's mixed cluster, whose jobs run in an image; see [Run jobs in an image](#run-jobs-in-an-image) |

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
2. In `<site>/config.yaml`, point `execution_site_file` at your copy:

   ```yaml
   config:
     execution_site_file: workflow/profiles/site/<site>/site.yaml
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
       --profile workflow/profiles/pipeline/kdd26 \
       --workflow-profile workflow/profiles/site/<site> \
       -n -p all
   ```

`execution_site` can instead be given in a config file, but not together with
`execution_site_file`. Do not put the nested `execution_site` mapping in a
profile's `config` section: this Snakemake/plugin version does not preserve
nested profile config in the remote job wrapper.

> **`--config` on the command line replaces the site profile's `config`.**
> Any `--config` drops `execution_site_file`, and the submission then fails
> before any job is submitted. Whenever you pass `--config`, also pass
> `execution_site_file=workflow/profiles/site/<site>/site.yaml`.

## Run jobs in an image

On a cluster that limits the number of files per project, run every job in
an [image](../../CONTEXT.md) instead of a venv: an image is one file, while a
venv is tens of thousands. The `arrhenius/` profile does this. Its CPU nodes
are x86_64 and its GPU nodes aarch64, so it uses one image per architecture.
The scripts run in the image, while Snakemake runs on the host from a small
environment built beside each image
([ADR 0008](../adr/0008-snakemake-on-the-host-scripts-in-the-image.md)).

1. Build the image and the host environment on a node of each architecture
   your allocations use, from the checkout root. On Arrhenius:

   ```shell
   sbatch -A <cpu account> -p cpu --output=containers/build-%j.log \
       containers/build.sbatch containers
   sbatch -A <gpu account> -p gpu --gpus 1 --output=containers/build-%j.log \
       containers/build.sbatch containers
   ```

   Each job writes `containers/afabench-<arch>.sif` and
   `containers/orchestration-<arch>-<lock hash>/`. Rebuild both after any
   change to `uv.lock`: the pipeline refuses an image built from another
   lock before submitting anything, and a job finds no host environment for
   it.
2. In `<site>/site.yaml`, name each allocation's image, relative to the
   repository root or as an absolute path:

   ```yaml
   execution_site:
     cpu:
       ...
       image: containers/afabench-x86_64.sif
     gpu:
       ...
       image: containers/afabench-aarch64.sif
   ```

   Jobs of a GPU allocation run with `--nv`. A missing image fails before
   submission.
3. In `<site>/config.yaml`, make each job start Snakemake from
   `containers/bin/python`, which picks the host environment of the node's
   architecture:

   ```yaml
   shared-fs-usage: [persistence, input-output, sources, source-cache, storage-local-copies]
   precommand: export PATH=$PWD/containers/bin:$PATH
   ```

4. Run Snakemake with `containers/snakemake.sh` instead of
   `uv run snakemake`, from the checkout root:

   ```shell
   containers/snakemake.sh \
       --profile workflow/profiles/pipeline/kdd26 \
       --workflow-profile workflow/profiles/site/arrhenius -n -p all
   ```

   It prints progress on the login node as usual. To run a single command in
   the image by hand, use `containers/run.sh`.

## A site map is required for SLURM

Submitting to SLURM without a site map fails during planning, before any job
is submitted. Local runs and dry runs need none. Without one, a dry run shows
the profile's `default-resources` partition and account, and `cuda` jobs
declare one generic GPU (`gpu=1`), which local runs can bound with
`--resources gpu=<n>`.

## Related documentation

- [Run the pipeline](run_the_pipeline.md) - the base
  command, how to narrow or change a run, and where to run it
- [Pipeline configuration](../reference/pipeline_configuration.md) - overview of the pipeline
  and its configuration
