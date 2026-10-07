# Compact evaluation dataframes

## Contract

`afabench.evaluation.schemas` defines three Pandera pandas contracts, each
extending the previous one:

- `BatchEvaluationSchema`: one row per episode and zero-based `step`,
  returned by `process_batch`.
- `EvaluationSchema`: the same columns plus `generation_index` and
  `split_index`, returned by `eval_afa_method`.
- `SavedEvaluationSchema`: the same columns plus nullable `eval_seed` and
  `eval_hard_budget`, added by `AFAEvaluator` before saving Parquet.

`episode_id` identifies an episode, not a dataset position. It is batch-local
for an individual `process_batch` call, but unique across all batches in one
`eval_afa_method` result. IDs are local to an evaluation artifact: independently
produced results must remain in separate namespaces when reconstructing
histories. `idx` is no longer emitted.

`generation_index` and `split_index` identify the instance an episode
evaluated, and are the same on every row of an episode
(`docs/adr/0004-instances-carry-their-generation-index.md`). `split_index` is
the instance's position in the evaluated dataset, the eval split's dataset
bundle; `generation_index` is its position in the dataset produced by dataset
generation, before splitting. Both are correct whether or not
`eval_only_n_samples` samples a subset. With the dataset key and dataset
realization, `generation_index` identifies the instance across methods and
splits; for a real-world dataset it is also the instance's row in the source.

`step` starts at zero and counts selections performed before the row's action.
An episode with three selections has steps 0, 1, 2, 3; the last row is stop.
Repeated selections count as separate steps, even if no new features appear.
The initializer's observations are not selections and do not increment `step`.

The other evaluation columns are `action_performed`,
`builtin_predicted_class`, `external_predicted_class`, `true_class`,
`accumulated_cost`, and `forced_stop`. Actions are one-based selection IDs;
zero means stop. Recorded actions include budget/forced-acquisition overrides,
not rejected actions or original policy intentions. Predictions use the
pre-action observations, whereas cost includes the current action.
Missing classifiers produce all-null prediction columns.

Functions return `pandera.typing.DataFrame[EvaluationSchema]`. Runtime
validation rejects missing, unexpected, or duplicate column names and checks
dtypes, nullability, and values without coercing dtypes. Each episode must
have contiguous steps from zero, no duplicate steps, and exactly one stop at
its final step. Row order is immaterial. Validation also runs after batch
concatenation and before saving.

Prediction and metadata fields use `Series[Any]` because pandas represents
all-null columns differently from populated columns. Explicit element-wise
checks enforce scalar integer class IDs and numeric metadata.

## On-demand histories

The producer and saved format do not contain `prev_selections_performed`.
Saving every selection-history prefix costs quadratic space per episode;
scalar episode IDs, steps, and actions require linear space instead.

```python
import pandas as pd
from pandera.typing import DataFrame

from afabench.evaluation.history import reconstruct_selection_history
from afabench.evaluation.schemas import SavedEvaluationSchema

results = DataFrame[SavedEvaluationSchema](pd.read_parquet(path))
histories = reconstruct_selection_history(results)
```

`reconstruct_selection_history` returns a Series aligned to the original row
positions and index, even with reordered rows or duplicate index labels. For
step t it collects the episode's earlier positive actions, subtracting one
from each. It preserves order and duplicates, excluding stop and initial
observations. It validates complete episodes before reconstructing.

This deliberately materializes quadratic-size prefixes only on demand. Use
`step` directly when only the history length is needed. Reconstruct **before**
filtering rows: an individual row or incomplete episode is not self-contained.
Neither histories nor actions alone recover initial masks or the Unmasker's
mapping from selections to features.

## Legacy reading and conversion

Existing Parquet files remain untouched. The evaluation-data transform accepts
both formats: it uses `step` for compact logs and stored history lengths for
legacy logs, including partial legacy tables. Legacy histories can be lists,
Parquet-restored NumPy arrays, or string lists. The plotting output
contract retains `n_selections_performed`, `generation_index` and
`split_index`, and discards episode identity when expanding rows by
classifier. Those tables are not episode logs. Tables transformed from legacy
logs have null `generation_index` and `split_index`. Compact tables written
before those columns existed are rejected rather than backfilled; re-run
their evaluation.

For explicit conversion of a complete, original-order legacy artifact:

```python
from afabench.evaluation.history import (
    convert_legacy_episode_log,
    reconstruct_selection_history,
)

legacy = pd.read_parquet(legacy_path)
compact = convert_legacy_episode_log(legacy, original_order=True)
histories = reconstruct_selection_history(compact)
```

Converted logs have no `generation_index` or `split_index`, which legacy
artifacts never recorded, so they do not satisfy `SavedEvaluationSchema`.

The caller must attest to original producer ordering. Legacy `idx` resets in
each batch; conversion starts a new episode when an index first appears or
reappears after stop. Stored histories must match the preceding actions,
every episode must terminate, and indices/actions/selections must be
nonnegative integers. A stopped index cannot be reused while another episode
in its batch is still active. Inconsistent or incomplete logs fail rather than guess.
The converter cannot prove provenance or detect an entirely omitted episode;
identity cannot be inferred safely from arbitrary reordered legacy tables.
Do not convert classifier-expanded plotting tables or combine source runs.

These contracts cover pandas evaluation results, not subsequent aggregation
or plotting tables or arbitrary dataset frames.
