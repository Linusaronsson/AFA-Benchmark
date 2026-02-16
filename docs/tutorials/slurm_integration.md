# SLURM Integration

The AFA-Benchmark pipeline uses [Snakemake](https://snakemake.readthedocs.io/) for workflow orchestration. Snakemake provides native support for SLURM (Simple Linux Utility for Resource Management), allowing you to submit jobs to a computing cluster instead of running them locally on your machine.

## Overview

SLURM is a workload manager used on high-performance computing (HPC) clusters. Instead of executing jobs sequentially or in parallel on your local machine, you submit jobs to a queue where they are scheduled and executed on available compute nodes.

Snakemake automatically handles:
- Job submission to SLURM
- Resource allocation (CPU cores, memory, GPU, runtime limits)
- Job scheduling and dependencies
- Parallel execution across many compute nodes

## Quick Start: Using a Profile

The easiest way to use SLURM is with a Snakemake profile. Profiles are located in `extra/workflow/profiles/`. Each profile contains a `config.yaml` file that configures SLURM settings for a specific cluster.

### Available Profiles

- `alvis_han/`, `alvis_rezvan/`, `alvis_valter/` - Profiles for the Alvis cluster
- `vera_rezvan/`, `vera_valter/` - Profiles for the Vera cluster

### Running the Pipeline with a Profile

To run the pipeline using SLURM, add the `--profile` flag to your Snakemake command:

```shell
WANDB_PROJECT=afabench \
  uv run snakemake \
    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    all \
    --profile extra/workflow/profiles/alvis_han \
    --configfile \
      extra/workflow/conf/eval_hard_budgets.yaml \
      extra/workflow/conf/methods.yaml \
      extra/workflow/conf/method_sets.yaml \
      extra/workflow/conf/method_options.yaml \
      extra/workflow/conf/pretrain_mapping.yaml \
      extra/workflow/conf/soft_budget_params.yaml \
      extra/workflow/conf/unmaskers.yaml \
      extra/workflow/conf/classifier_names.yaml \
      extra/workflow/conf/datasets_main.yaml \
    --config \
      eval_dataset_split=val \
      "dataset_instance_indices=[0,1]" \
      smoke_test=false \
      use_wandb=true \
      device=cuda
```

The `--profile` flag replaces the `--jobs` flag you would normally use for local execution. Snakemake reads the profile's `config.yaml` and automatically submits jobs to SLURM.

## Understanding Profile Configuration

Each profile's `config.yaml` contains SLURM settings. Here's an example from `alvis_valter/config.yaml`:

```yaml
executor: slurm
jobs: 9001
latency-wait: 30 # seconds

default-resources:
  runtime: 120 # minutes
  slurm_partition: alvis
  slurm_account: NAISS2025-22-1420
  mem_mb: 0 # Only memory attribute that works
  slurm_extra: "--gres=gpu:T4:1"

set-resources:
  pretrain_model:
    runtime: 600 # 10 hours
  train_method_with_pretrained_model:
    runtime: 600 # 10 hours
  train_method_without_pretrained_model:
    runtime: 600 # 10 hours
  eval_method:
    runtime: 600 # 10 hours
```

### Configuration Parameters

**Top-level settings:**
- `executor: slurm` - Use SLURM as the job executor
- `jobs: N` - Maximum number of jobs to submit to SLURM at once (limits queue size; 9001 is effectively unlimited)
- `latency-wait: 30` - Wait 30 seconds for filesystem to sync between nodes

**default-resources:**
These apply to all jobs unless overridden by `set-resources`:
- `runtime: N` - Job time limit in minutes
- `slurm_partition: name` - SLURM partition (queue) to submit to
- `slurm_account: name` - SLURM account for billing/allocation
- `mem_mb: N` - Memory per job in MB (0 means no explicit limit)
- `cpus_per_task: N` - CPU cores per job
- `slurm_extra: "..."` - Additional SLURM parameters (e.g., GPU requests)

**set-resources:**
Override default resources for specific pipeline steps. Each key corresponds to a Snakemake rule name. For example:
- `pretrain_model` - Pretraining jobs (usually need GPU and longer runtime)
- `train_method_with_pretrained_model` - Training with pretraining (usually most resource-intensive)
- `eval_method` - Evaluation jobs

### Example: GPU Allocation

In `alvis_han/config.yaml`, the default GPU allocation is:
```yaml
slurm_extra: "--gres=gpu:T4:1"  # Request 1 Tesla T4 GPU per job
```

To use a different GPU type, modify this parameter. For example:
```yaml
slurm_extra: "--gres=gpu:A40:1"  # Request 1 A40 GPU per job
```

Different rules can request different GPUs:
```yaml
set-resources:
  pretrain_model:
    slurm_extra: "--gres=gpu:A40:1"  # Use A40 for pretraining
  train_method_with_pretrained_model:
    slurm_extra: "--gres=gpu:T4:1"   # Use T4 for training
```

## Creating Your Own Profile

To create a profile for a new cluster:

1. Create a new directory in `extra/workflow/profiles/`:
   ```bash
   mkdir extra/workflow/profiles/my_cluster
   ```

2. Create `extra/workflow/profiles/my_cluster/config.yaml` with appropriate settings:
   ```yaml
   executor: slurm
   jobs: 9001
   latency-wait: 30
   
   default-resources:
     slurm_partition: my_partition
     slurm_account: my_account
     runtime: 120
     cpus_per_task: 1
     mem_mb: 4000
   
   set-resources:
     pretrain_model:
       runtime: 600
       cpus_per_task: 4
       mem_mb: 32000
     train_method_with_pretrained_model:
       runtime: 600
       cpus_per_task: 4
       mem_mb: 32000
   ```

3. Use your profile with the `--profile` flag:
   ```bash
   uv run snakemake ... --profile extra/workflow/profiles/my_cluster
   ```

## Important Notes

### Resource Estimation

Time limits and memory requirements depend on:
- Dataset size (varies significantly)
- Model complexity
- Number of dataset instances/seeds being run
- Whether pretraining is involved

Start with conservative estimates (generous resource requests) and observe actual usage, then optimize. SLURM will terminate jobs that exceed their allocated runtime.

### Mixed Resources on Different Nodes

If your cluster has heterogeneous hardware, use `set-resources` to allocate different resources to different jobs:

```yaml
set-resources:
  pretrain_model:
    slurm_partition: gpu_partition  # GPU nodes
    slurm_extra: "--gres=gpu:A40:1"
  train_classifier:
    slurm_partition: cpu_partition  # CPU-only nodes
```

### Debugging SLURM Issues

If jobs fail or don't submit:

1. Check if your SLURM account and partition are valid:
   ```bash
   sinfo                          # List available partitions
   sacctmgr show associations     # List your accounts
   ```

2. Check job status:
   ```bash
   squeue -u $USER                # Show your queued/running jobs
   sacct -j <job_id>             # Show job details after completion
   ```

3. Check Snakemake logs for submission errors - Snakemake will print SLURM error messages to the terminal.

### Local Fallback

To test your pipeline locally before submitting to SLURM, run without the `--profile` flag:

```bash
WANDB_PROJECT=afabench \
  uv run snakemake ... --jobs 8
```

This runs 8 jobs in parallel on your local machine without using SLURM.

## Related Documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
- [Snakemake documentation](https://snakemake.readthedocs.io/) - Comprehensive Snakemake guide
- [Snakemake SLURM support](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) - Detailed SLURM configuration options
