# SLURM integration

The pipeline supports SLURM via [Snakemake's SLURM plugin](https://snakemake.readthedocs.io/en/stable/executing/cluster.html). Profiles are located in `extra/workflow/profiles/`.

## Example profiles

The repository includes two profiles used by our team as examples:

- `arrhenius/` - x86_64 CPU and aarch64 GH200 nodes
- `vera/` - CPU and GPU nodes

These are unlikely to work out of the box for you. See
[Creating your own profile](#creating-your-own-profile) to set one up for your
cluster.

## Running with a profile

```shell
uv run snakemake \
    --profile extra/workflow/profiles/config/gpu_methods \
    all \
    --workflow-profile extra/workflow/profiles/arrhenius \
    --config device=cuda
```

The missing training data study runs in one allocation instead:

```shell
scripts/workflow/submit_missing_data_arrhenius.sh \
    --profile missing_data --cores 32 --mem-mb 64000 --time 72:00:00
```

## Creating your own profile

Create `extra/workflow/profiles/<your_cluster>/config.yaml`, using the existing
profiles as a starting point, and pass it with `--workflow-profile`.
`set-resources` keys must be rule names; Snakemake silently ignores unknown
keys. `just test` checks this.

`orchestration/pipeline.smk`:

- `pretrain_model`
- `train_method_with_pretrained_model`
- `train_method_without_pretrained_model`
- `eval_method`

`orchestration/missing_data.smk`:

- `pretrain_incomplete_restoration_pvae`, `pretrain_oracle_restoration_pvae`
- `pretrain_missing_data_method`
- `train_missing_data_method_with_pretraining`
- `train_missing_data_method_without_pretraining`
- `eval_missing_data_method`

See the [Snakemake SLURM plugin documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for all available configuration options.

## Related documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
