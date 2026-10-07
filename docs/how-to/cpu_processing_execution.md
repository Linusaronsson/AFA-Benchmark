# CPU-only dataset and result processing

Dataset generation, evaluation-data transformation, aggregation (including
classifier-type splitting and timing-data combination), and all visualization
rules always resolve to CPU execution. They share the execution-policy/site
allocation mechanism with computational jobs, but their pipeline stages
`dataset_generation`, `transformation`, `aggregation`, and `visualization` are
fixed: they cannot be configured in execution defaults or identity overrides.
Even a deprecated global `device=cuda` cannot change them. Their existing
CPU-native scripts do not receive an invented `device` argument.

## Cluster allocation and sizing

For mixed-device SLURM runs, configure the profile-owned `execution_site.cpu`
partition/account to an authorized CPU allocation, as illustrated by
`extra/workflow/profiles/mixed-gres/site.yaml` and `mixed-gpus/site.yaml`.
Processing rules explicitly clear `gpu`, `gpu_model` and `gres`, including GPU
requests inherited from the profile's default resources, and take `slurm_extra`
from the CPU allocation. A CPU site mapping that requests GPUs, also through its
`slurm_extra`, is invalid, and so is any `slurm_extra` in default resources.
Rule/profile `set-resources` overrides that conflict with the resolved
allocation are rejected during planning, before any submission; configure
allocation through the site map instead. These stages are CPU-only, not
automatically local rules or login-node work.

Threads, `cpus_per_task`, memory and runtime remain independent resource-sizing
settings. Existing profile values and rule-specific sizing are preserved. For
example, the Vera profile's ten CPUs for classifier-type splitting and plots
remain valid; copying those sizing entries into an adapted mixed profile is
appropriate. CPU intent alone does not determine suitable memory or CPU count
for a large dataframe. Every SLURM submission needs a site map with a CPU
allocation, since every graph contains these jobs, even when only GPU methods
are selected; a missing site map or CPU allocation fails before submission.

## One graph and native paths

Supply the existing scientific config files and computational execution choices
as usual. Processing needs no extra portable hardware configuration:

```sh
uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk \
  --workflow-profile extra/workflow/profiles/mixed-gres \
  --configfile benchmark.yaml execution.yaml -n -p all
```

Inspect the plan, then remove `-n` to submit from an authorized controller with
CPU/GPU allocation access and shared input, output and software filesystems.
CPU processing and GPU method jobs belong to the same invocation and retain
all dependencies, native bundle/parquet/plot locations, dataset instance indices,
seeds and plotting formats. No scientific settings are changed. The example
profiles are illustrative, not verified allocations on a live cluster.

`pipeline_no_train.smk` and `pipeline_no_eval.smk` use the same processing rules
for existing upstream outputs. Local CPU smoke-test use remains possible by
omitting the cluster profile and using `--cores` with `smoke_test=True`.

## Observable verification

```sh
uv run pytest test/workflow/test_cpu_processing_execution.py
uv run pytest test/workflow/test_cpu_processing_execution.py -m pipeline
```

Fast checks inspect the real final-target plan and reject conflicting GPU
resource overrides for every processing rule, and `slurm_extra` in default
resources, before submission. Pipeline checks
invoke that same graph with tiny script-boundary fixtures and the shared fake
SLURM boundary, capture actual executor submissions under both GPU request
conventions, and verify every processing rule has CPU partition/account, no GPU
flags, and retained CPU/memory/runtime sizing despite adversarial GPU defaults.
They check script arguments and representative native fixture output paths,
not scientific correctness of stubbed computation. Both pretrained and
non-pretrained timing paths and evaluation-performance/action/timing plots are
included. No live cluster or production computation is used. The pinned plugin
waits between dependency waves; these full-graph tests have an explicit
600-second per-invocation timeout and are excluded from fast QA. `just qa` is
still mandatory.
