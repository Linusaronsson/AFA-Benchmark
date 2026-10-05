# Typed evaluation dataframes

`afabench.evaluation.schemas` defines two Pandera pandas contracts:

- `EvaluationSchema`: one row per sample and acquisition timestep, returned
  by `process_batch` and `eval_afa_method`.
- `SavedEvaluationSchema`: the same columns plus nullable `eval_seed` and
  `eval_hard_budget`, added by `AFAEvaluator` before saving Parquet.

Functions return `pandera.typing.DataFrame[EvaluationSchema]`. Constructing
these typed frames validates required columns, dtypes, nullability, and value
checks at runtime. Results are also validated after batch concatenation and
before saving. Both schemas reject unexpected and duplicate column names.
Validation does not coerce existing dataframe dtypes.

Sample `idx` values are batch-local, not globally unique dataset identifiers.
Selection histories contain zero-based selection IDs, whereas actions are
one-based (zero means stop). Missing classifiers produce all-null prediction
columns. Predictions, when present, are scalar nonnegative integer class IDs.

Prediction and metadata fields use `Series[Any]` because pandas represents
all-null columns differently from populated columns. Explicit element-wise
checks enforce the non-null value types rather than accepting arbitrary data.

For pandas Parquet readers, PyArrow restores history cells as NumPy arrays.
Convert these to lists of Python integers before validating against the
in-memory contract:

```python
import pandas as pd
from pandera.typing import DataFrame

from afabench.evaluation.schemas import SavedEvaluationSchema

frame = pd.read_parquet(path)
frame["prev_selections_performed"] = frame["prev_selections_performed"].map(
    lambda selections: [int(value) for value in selections]
)
results = DataFrame[SavedEvaluationSchema](frame)
```

These contracts cover the pandas evaluation producer and saved results, not
subsequent Polars aggregation/plotting tables or arbitrary dataset frames.
