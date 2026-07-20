# SLURM integration

The pipeline supports SLURM via [Snakemake's SLURM plugin](https://snakemake.readthedocs.io/en/stable/executing/cluster.html). Profiles are located in `extra/workflow/profiles/`.

## Example profiles

The repository includes two profiles used by our team as examples:

- `alvis/` - GPU cluster
- `vera/` - CPU cluster

These are unlikely to work out of the box for you. See
[Creating your own profile](#creating-your-own-profile) to set one up for your
cluster.

## Running with a profile

Add `--workflow-profile` to the pipeline command instead of `--jobs`. The
example also uses `--profile extra/workflow/profiles/config/gpu_methods` to
load the standard pipeline config files restricted to GPU methods:

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/gpu_methods \
    all \
    --workflow-profile extra/workflow/profiles/alvis \
    --config device=cuda
```

## Creating your own profile

Create a directory `extra/workflow/profiles/<your_cluster>/` containing a
`config.yaml`. Use the existing profiles as a starting point and pass it with
`--workflow-profile`.

**`set-resources` keys must name rules from the workflow you are running.**
Snakemake ignores a key that matches no rule, so a wrong name does not fail; it
downgrades the job to `default-resources` and the job is walltime-killed after
it has already queued. The two workflows use different rule names:

`orchestration/pipeline.smk` (main benchmark):

- `pretrain_model`
- `train_method_with_pretrained_model`
- `train_method_without_pretrained_model`
- `eval_method`

`orchestration/missing_data.smk` (restoration study):

- `pretrain_incomplete_restoration_pvae`, `pretrain_oracle_restoration_pvae`
- `pretrain_missing_data_method`
- `train_missing_data_method_with_pretraining`
- `train_missing_data_method_without_pretraining`
- `eval_missing_data_method`
- plus the cheap `generate_missing_data_dataset`,
  `materialize_missing_training_view`, `restore_missing_training_view`,
  `train_missing_data_shared_classifier`, `train_missing_data_method_classifier`

A profile may list both sets; unmatched keys are harmless. `just test` runs
`test/workflow/test_cluster_profiles.py`, which fails if any key is not a real
rule, or if a network-training rule is left on the default.

Note that the `extra/workflow/profiles/config/missing_data_*` profiles pin
`cores: 4` for local use. Override it on the command line for cluster runs or it
throttles the whole DAG to four concurrent jobs regardless of `jobs:`.

### Available clusters

| Profile | Notes |
|---|---|
| `vera` | Primary. One architecture, 7-day walltime, T4/A40/A100/H100, 155 CPU-only nodes. |
| `arrhenius` | Backup. 72h cap, split `-cpu`/`-gpu` accounts, and **x86_64 CPU nodes with aarch64 Grace Hopper GPU nodes**, so it needs one virtualenv per architecture. Build the aarch64 one from an interactive GPU job; NAISS documents that GPU-side software cannot be built on the login node. See the profile's comments. |
| `alvis` | Where the main-benchmark results in `extra/output/alvis` came from. Uses `--gres` and rejects explicit memory requests. |

On a mixed-architecture cluster, export `AFABENCH_SHELL_PREFIX` so each job
activates the matching environment. `missing_data.smk` applies it via
`shell.prefix`, so no rule needs to know:

```bash
export AFABENCH_SHELL_PREFIX='source "$VENVS"/$(uname -m)/bin/activate; '
```

See the [Snakemake SLURM plugin documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for all available configuration options.

## Related documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
