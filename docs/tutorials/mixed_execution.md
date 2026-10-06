# Per-method training and evaluation execution

The ordinary `pipeline.smk` invocation can train and evaluate different methods
on CPU and GPU together, without changing scientific configuration, method-owned
scripts, dependencies, or native output paths. This page covers method training
and evaluation; see [classifiers and pretrained models](prerequisite_execution.md),
[CPU-only processing](cpu_processing_execution.md), and the single full-benchmark
command in [Reproducing full results](reproduce_full_results.md).

## Portable execution configuration

Put execution choices in a YAML config file, separate from method options:

```yaml
execution:
  defaults:
    training: cpu
    evaluation: cpu
  methods:
    jafa:
      training: cuda
      evaluation: cpu
    eddi:
      evaluation: cuda
```

Precedence, independently for each selected method and stage:

1. `execution.methods.<method name>.<stage>`
2. `execution.defaults.<stage>`
3. `cpu` (or the deprecated global `device` in invocations without `execution`)

Values are exactly `cpu` and `cuda`. No hardware is inferred from method
implementation, taxonomy, method-selection profile, or GPU availability.
`cuda` declares GPU execution intent and passes `device=cuda` to the existing
script; `cpu` passes `device=cpu`. Evaluation includes the classifier, so declare
its end-to-end execution needs, not only those of the policy. Training and
evaluation need not agree. There is no automatic CPU fallback.

## Site allocation is separate

Adapt `extra/workflow/profiles/mixed-gres/` or `mixed-gpus/`. Both are illustrative,
not verified allocations on any real cluster. The profile passes a scalar
`execution_site_file` path; its `site.yaml` owns partitions, accounts and GPU
request syntax. Paths are relative to the invocation working directory; run from
the repository root, or use an absolute path when adapting the profile.

```yaml
# site.yaml, referenced only by the cluster profile
execution_site:
  cpu:
    slurm_partition: cpu-queue
    slurm_account: cpu-account
  gpu:
    slurm_partition: gpu-queue
    slurm_account: gpu-account
    gres: gpu:T4:1
```

Alternatively, GPU allocation can specify `gpu: 1` and optionally
`gpu_model: a100`, producing `--gpus=a100:1` instead of `--gres=gpu:T4:1`.
Use exactly one positive GPU request convention. CPU allocation may not request
GPUs. Either allocation may add other scheduler flags with `slurm_extra`, but no
GPU requests; `slurm_extra` in `default-resources` is rejected. The portable
execution file must not contain partitions or accounts.

`execution_site` can also be supplied directly in a **config file**, but not
alongside `execution_site_file`. Do not put nested mappings in a profile's
`config` CLI options: Snakemake 9.12.0 / SLURM executor 1.8.0 fail to preserve
those mappings in the remote shell wrapper. The scalar file reference avoids
that boundary issue without changing engine or plugin versions.

Per-job resources distinguish methods within the same `train_method` and
`eval_method` rules. CPU jobs explicitly clear `gpu`, `gpu_model` and `gres`,
even if a profile supplies GPU defaults, and the site map routes partition,
account and `slurm_extra` too. SLURM submission without a site map, or with a
site map lacking the allocation a job resolves to, fails before any job is
submitted; local runs need no site map. CPU count, memory and runtime remain independent profile resource settings.
The examples retain 4000 MB, 8 CPUs and 600 minutes for method jobs; adapt sizing
as needed, rather than changing scientific settings. Existing Vera/Alvis sizing
is unchanged.

Invalid selected-method choices, incompatible site allocations, duplicate device
arguments in `method_specific_params`, and conflicting final rule allocation
resources fail during planning, before **any** job submission. Rule/profile
`set-resources` may still set CPU counts, memory and runtime; do not override the
allocation resources there. Configuration validation cannot prove live cluster
availability or a script's support for a chosen device.

## Invocation and migration

From the repository root, supply the existing scientific config files plus the
new portable execution file. For example, with those scientific definitions
combined in `benchmark.yaml` and the execution mapping in `execution.yaml`:

```sh
uv run snakemake -s extra/workflow/snakefiles/orchestration/pipeline.smk \
  --workflow-profile extra/workflow/profiles/mixed-gres \
  --configfile benchmark.yaml execution.yaml \
  -n -p all_eval_methods
```

Remove `-n` to execute/submit. Use an authorized submit environment with access
to both CPU and GPU allocations and a shared filesystem for inputs, repository,
software environment and outputs. This is one dependency graph, not cross-cluster
dispatch. `pipeline_no_train.smk` uses the same evaluation routing for existing
trained bundles. Local CPU use needs no site profile: omit `--workflow-profile`,
use CPU execution choices, and pass `--cores` and `--config smoke_test=True`.

An explicit global `device` remains supported with a visible deprecation
message in invocations without `execution`. **Any** combination of global
`device` and `execution` is rejected, even if values happen to agree. Remove
`device` from your commands and config files. The former CPU/GPU
method-selection profiles have been removed; their hardware split is declared
in `extra/workflow/conf/execution/`. See
[Reproducing full results](reproduce_full_results.md#migrating-from-the-six-invocation-workflow).

## Boundary verification

```sh
uv run pytest test/workflow/test_method_execution.py
uv run pytest test/workflow/test_method_execution.py -m pipeline
```

The fast tests inspect the real orchestration's planned commands and diagnostics,
including downstream allocation conflicts before submission, and capture the
real executor's first CPU and GPU submissions for both illustrative profiles,
stopping Snakemake before the plugin's first status check. The pipeline-marked
tests run that same orchestration using tiny pre-existing dataset/classifier
fixtures, script-boundary stubs, and fake `sbatch`/`srun`/`sacct` commands. They
capture actual executor submissions for both illustrative profiles, execute real
Snakemake job wrappers, and check script arguments and outputs. No cluster,
training, public network or unrelated toy graph is involved. The pinned plugin
has a fixed 40-second initial status wait despite its similarly named option;
these two fixtures take about four minutes together. `just qa` remains the
mandatory quality gate; run the submission tests explicitly as well.
