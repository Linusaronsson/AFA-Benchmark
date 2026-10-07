# Compact evaluation dataframes

## Contract

`afabench.evaluation.schemas` defines Pandera pandas contracts, each
extending `BatchEvaluationSchema` or `EvaluationSchema`:

- `BatchEvaluationSchema`: one row per episode and zero-based `step`,
  returned by `process_batch`.
- `EvaluationSchema`: the same columns plus `generation_index` and
  `split_index`, returned by `eval_afa_method`.
- `SavedEvaluationSchema`: the `EvaluationSchema` columns plus the identity
  columns below, added by `AFAEvaluator` before saving Parquet.
- `PreProvenanceSavedEvaluationSchema`: the `EvaluationSchema` columns plus
  nullable `eval_seed` and `eval_hard_budget`, as the evaluator saved them
  before the identity columns existed. The transform step still accepts
  these tables.

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

Prediction fields use `Series[Any]` because pandas represents all-null
columns differently from populated columns. Explicit element-wise checks
enforce scalar integer class IDs.

## Identity and provenance

A raw evaluation table identifies itself
(`docs/adr/0002-provenance-recorded-in-artifacts.md`), whether the pipeline
or a hand run of `scripts/eval/eval_afa_method.py` wrote it. Its identity
columns are constant per table and use pandas nullable dtypes, which
`pandas.read_parquet` restores without `afabench`:

| Column | dtype | Value |
| --- | --- | --- |
| `afa_method` | `string` | Method name, from the method bundle's record. |
| `dataset` | `string` | Dataset key, from the eval dataset bundle's record. |
| `dataset_realization_index` | `UInt64` | Dataset realization, from the eval dataset bundle's record. |
| `eval_split` | `string` | `train`, `val` or `test`: the eval dataset bundle's own split. |
| `initializer` | `string` | Name of the initializer config the evaluation selected, such as `cold`. Never null. |
| `train_seed` | `UInt64` | Seed of the method bundle's record. |
| `train_hard_budget` | `Float64` | `hard_budget` of the method's training contract. |
| `train_soft_budget_param` | `Float64` | `soft_budget_param` of the method's training contract. |
| `eval_seed` | `UInt64` | The seed the evaluation used. Never null: a null configured seed is drawn once and recorded. |
| `eval_hard_budget` | `Float64` | The evaluation's hard budget, null in the soft-budget setting. |
| `eval_soft_budget_param` | `Float64` | The evaluation's soft-budget parameter, null when not given. |

A value whose source bundle was written without a provenance record is null,
not guessed. The pretraining seed, the Unmasker, the classifier and the rest
of the configuration are in the table's provenance record only. The record
is stored as JSON under the Arrow schema metadata key `afabench.provenance`;
pandas drops it on read, so read it with
`afabench.evaluation.provenance.evaluation_table_provenance(path)`, which
returns null for a table written without one.

The evaluator owns the evaluation seed: it resolves a null seed once, seeds
Python, NumPy and torch with it, and passes the resolved integer to the
method, the Unmasker, the Initializer and the evaluation sampler. The same
seed reproduces the table on CPU.

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

## Plotting-ready tables

`scripts/misc/transform_eval_data_pipeline.py` turns one raw table into one
plotting-ready table. It reads identity from the raw table's columns and only
derives plotting columns: `n_selections_performed`, the melt of the
prediction columns into `classifier` (`builtin` or `external`) and
`predicted_class`, and nullable dtypes. The output keeps `generation_index`,
`split_index` and every identity column, including
`dataset_realization_index` and `eval_split`, and copies the raw table's
provenance record into its own schema metadata. It discards episode
identity when expanding rows by classifier, so these tables are not episode
logs.

Its identity arguments (`--method`, `--dataset`, `--initializer`,
`--train_seed`, `--train_hard_budget`, `--train_soft_budget_param`,
`--eval_soft_budget_param`) are optional checks. An argument that disagrees
with a non-null column raises `IdentityArgumentMismatchError`; `null` is an
explicit null value and disagrees with a non-null column. A column that is
missing or null is filled from its argument, or left null when the argument
is omitted. The Snakemake rule passes its wildcards, so a raw table written
before the identity columns existed is identified from them, except for
`dataset_realization_index` and `eval_split`, which no argument names and
which stay null.

## Legacy reading and conversion

Existing Parquet files remain untouched. The evaluation-data transform accepts
compact tables with and without identity columns, each validated against the
column set it was written with, and legacy logs: it uses `step` for compact
logs and stored history lengths for legacy logs, including partial legacy
tables. Legacy histories can be lists, Parquet-restored NumPy arrays, or
string lists. Tables transformed from legacy logs have null
`generation_index` and `split_index`. Compact tables written before those
columns existed are rejected rather than backfilled; re-run their
evaluation.

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
