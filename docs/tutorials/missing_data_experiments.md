# Missing-training-data experiments

The dedicated workflow in `extra/workflow/snakefiles/orchestration/missing_data.smk` ports the missing-data study onto the current benchmark architecture.
Missingness is limited to immutable training and validation views; evaluation always starts cold on a complete validation or test bundle.
The Snakefile contains dependency logic only: the scientific design lives in `extra/workflow/conf/missing_data/design.yaml`, while runtime scale, hardware, and evaluation split live in separate config files and normal Snakemake profiles.

The full study uses CUBE-NM, a hard budget of 14, and five dataset instances.
Its missingness matrix contains MCAR, MAR, MNAR-logistic, and MNAR-self at probabilities 0.3, 0.5, and 0.7.
MAR uses `p_obs=0.3`; both MNAR mechanisms use `p_params=0.3`.
Mechanism parameters are fit on the training split and reused when sampling the corresponding validation view.

## Baselines and controls

The shared completion ladder contains:

- restricted observations;
- mean completion;
- PVAE (label-conditioned);
- PVAE (label-free);
- PVAE (oracle);
- true completion.

Complete-data runs provide a ceiling/reference.
Zero fill is only an AACO k-NN-search control and is attached to AACO in the method configuration; it is not presented as a general imputation baseline.
The two reweighting controls are also method-specific: doubly robust support correction for AACO and feature-marginal inverse-probability weighting for DIME.

The PVAE fitted to incomplete data respects the fixed factual support mask.
A single joint reconstruction is drawn per row, and factual cells are always preserved.
The oracle PVAE is fitted to complete training data.
Downstream methods receive ordinary immutable dataset bundles, so missingness-specific logic stays at the data boundary.

## Runtime profiles

First run the 128-row, one-instance smoke profile on MPS:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_smoke \
  --cores 4
```

The smoke matrix deliberately covers AACO, DIME, ODIN, both reweighting controls, incomplete and oracle PVAE restoration, evaluation, and summarization.
It writes only to the `smoke` artifact namespace.

For small local results, run two 512-row instances under MCAR and MAR:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_local \
  --cores 4
```

The local profile uses reduced training schedules but does not enable smoke-test shortcuts.
It writes to a separate `local` namespace, so a successful smoke run cannot be mistaken for experimental output.

Run full validation before the final test evaluation:

```console
uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_full_validation \
  --cores 4

uv run snakemake \
  --profile extra/workflow/profiles/config/missing_data_full_test \
  --cores 4
```

The full validation and test profiles share the `full` namespace and identical training settings.
The test profile changes only `eval_dataset_split`, so it reuses the frozen validation-run methods and evaluates them once on the complete test split.

To make a new experiment, add a runtime YAML file and pair it with `design.yaml` in a profile; no Snakefile edit or stage number is needed.
One-off scalar overrides remain possible, for example `--config device=cpu use_wandb=false`, although checked-in runtime configs are preferable for reproducible results.
Cluster resource settings belong in an execution profile and are independent of the experiment definition.

Each namespace produces per-instance metrics, means and standard errors across instances, per-action acquisition rates, and PVAE restoration RMSE under `extra/output/missing_data/summary/`.
