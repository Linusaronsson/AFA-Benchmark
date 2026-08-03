# SLURM integration

The ordinary pipeline supports per-rule SLURM submission via
[Snakemake's SLURM plugin](https://snakemake.readthedocs.io/en/stable/executing/cluster.html).
Profiles are under `extra/workflow/profiles/`. The missing-data study instead
runs an internal Snakemake scheduler within one allocation; see
`missing_data_experiments.md`.

## Example profiles

The repository includes two profiles used by our team as examples:

- `arrhenius/` - NAISS CPU cluster, ordinary benchmark only
- `vera/` - C3SE cluster, secondary

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
    --workflow-profile extra/workflow/profiles/arrhenius \
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

Do not combine the Arrhenius per-rule profile with a `missing_data_*` config.
Those matrices contain thousands of short processes and run through
`scripts/workflow/submit_missing_data_arrhenius.sh`, which validates the live
association run-minute cap before requesting one allocation.

### Available clusters

| Profile | Notes |
|---|---|
| `arrhenius` | Primary. 72h walltime cap and x86_64 CPU nodes. The per-rule profile is for the ordinary benchmark. Missing-data allocations may instead use aarch64 GH200 nodes through the dedicated runner. |
| `vera` | Secondary. Better hardware fit (one architecture, 7-day walltime, T4/A40 class GPUs) but no durable storage: Alvis and its national Mimer allocations are being decommissioned and Cephyr left NAISS on 2026-07-01, leaving only a 30 GiB home. Use once e-Commons confirms a Chalmers-local allocation. |

Arrhenius has x86_64 CPU nodes and aarch64 GH200 nodes. PyTorch and torchvision
are pinned to 2.11.0 and 0.26.0; PyTorch 2.11 is the first release whose
default Linux aarch64 wheel is CUDA-enabled. Keep one synchronized virtualenv
per architecture on project storage. The missing-data runner selects the
matching environment after Slurm places the allocation and keeps per-rule
temporaries under `$SNIC_TMP`.

For a custom mixed-architecture per-rule workflow, `missing_data.smk` still
honors `AFABENCH_SHELL_PREFIX`:

```bash
export AFABENCH_SHELL_PREFIX='source "$VENVS"/$(uname -m)/bin/activate; '
```

See the [Snakemake SLURM plugin documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for all available configuration options.

## Related documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
