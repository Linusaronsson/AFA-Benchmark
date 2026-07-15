# Missing-training-data experiments

The dedicated workflow in
`extra/workflow/snakefiles/orchestration/missing_data.smk` ports the
missing-data study onto the current benchmark architecture. Missingness is
limited to the training and validation views. Model selection uses the complete
cold-start validation bundle, and stage 4 evaluates the frozen methods once on
the complete cold-start test bundle.

The study is intentionally scoped to CUBE-NM, a hard budget of 14, and five
dataset instances. The fixed missingness matrix contains MCAR, MAR,
MNAR-logistic, and MNAR-self at probabilities 0.3, 0.5, and 0.7. MAR uses
`p_obs=0.3`; both MNAR mechanisms use `p_params=0.3`. Mechanism parameters are
fit on the training split and reused when sampling the corresponding validation
view.

## Baselines and controls

Every method is trained against the same shared ladder:

- restricted observations;
- mean completion;
- PVAE (label-conditioned);
- PVAE (label-free);
- PVAE (oracle);
- true completion.

The complete-data run is added from stage 2 onward. Zero fill is only an AACO
k-NN control and is not presented as a general imputation baseline. The two
reweighting controls are also method-specific: doubly robust support correction
for AACO and feature-marginal IPW for DIME.

The PVAE fitted to incomplete data respects the fixed factual support mask. A
single joint reconstruction is drawn per row, and factual cells are always
preserved. The oracle PVAE is fitted to complete training data. Downstream
methods receive ordinary immutable dataset bundles, so missingness-specific
logic stays at the data boundary.

## Running the stages

Use the stages in order. Stage 1 writes to a separate `smoke` namespace; stages
2 through 4 share the `full` namespace, so later stages extend or reuse earlier
artifacts.

```console
uv run snakemake \
  -s extra/workflow/snakefiles/orchestration/missing_data.smk \
  --cores 4 --config stage=1

uv run snakemake \
  -s extra/workflow/snakefiles/orchestration/missing_data.smk \
  --cores 4 --config stage=2

uv run snakemake \
  -s extra/workflow/snakefiles/orchestration/missing_data.smk \
  --cores 4 --config stage=3

uv run snakemake \
  -s extra/workflow/snakefiles/orchestration/missing_data.smk \
  --cores 4 --config stage=4
```

Stage 1 runs all configured methods and baselines for MCAR at 0.5 on instance 0
with smoke settings. Stage 2 runs all mechanisms at 0.5 over five instances.
Stage 3 expands to all three probabilities. Stage 4 changes only the evaluation
split from validation to test; its training artifact paths are identical to
stage 3, which makes the freeze boundary explicit.

Runtime settings can be overridden without editing the checked-in scientific
matrix, for example `--config stage=1 device=cpu use_wandb=false`. Cluster
resource settings belong in a normal Snakemake workflow profile rather than in
this experiment definition.

Each run produces per-instance metrics, mean and standard error across
instances, per-action acquisition rates, and PVAE restoration RMSE under
`extra/output/missing_data/summary/`.
