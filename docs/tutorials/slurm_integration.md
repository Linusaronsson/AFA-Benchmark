# SLURM Integration

The pipeline supports SLURM via [Snakemake's SLURM plugin](https://snakemake.readthedocs.io/en/stable/executing/cluster.html). Profiles are located in `extra/workflow/profiles/`.

## Example Profiles

The repository includes two profiles used by our team as examples:

- `alvis/` - GPU cluster
- `vera/` - CPU cluster

These are unlikely to work out of the box for you. See [Creating Your Own Profile](#creating-your-own-profile) to set one up for your cluster.

## Running with a Profile

Add `--profile` to the pipeline command instead of `--jobs`:

```shell
uv run snakemake \
    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
    all \
    --profile extra/workflow/profiles/alvis \
    --configfile \
      extra/workflow/conf/eval_hard_budgets/all.yaml \
      extra/workflow/conf/methods/all.yaml \
      extra/workflow/conf/method_sets/all.yaml \
      extra/workflow/conf/method_options/all.yaml \
      extra/workflow/conf/pretrain_mappings/all.yaml \
      extra/workflow/conf/soft_budget_params/all.yaml \
      extra/workflow/conf/unmaskers/all.yaml \
      extra/workflow/conf/classifier_names/all.yaml \
      extra/workflow/conf/datasets/all.yaml \
    --config \
      device=cuda
```

## Creating Your Own Profile

Create a directory `extra/workflow/profiles/<your_cluster>/` containing a `config.yaml`. Use the existing profiles as a starting point. The pipeline rule names you can set resources for are:

- `pretrain_model`
- `train_method_with_pretrained_model`
- `train_method_without_pretrained_model`
- `eval_method`

See the [Snakemake SLURM plugin documentation](https://snakemake.readthedocs.io/en/stable/executing/cluster.html) for all available configuration options.

## Related Documentation

- [Pipeline explanation](pipeline_explanation.md) - Overview of the full pipeline
