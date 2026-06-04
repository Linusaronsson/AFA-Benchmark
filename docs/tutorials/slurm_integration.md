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
`--workflow-profile`. The pipeline rule names you can set resources for are:

- `pretrain_model`
- `train_method_with_pretrained_model`
- `train_method_without_pretrained_model`
- `eval_method`

See the [Snakemake SLURM plugin documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for all available configuration options.

## Related documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
