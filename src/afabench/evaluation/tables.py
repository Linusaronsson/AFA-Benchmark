"""
Load plotting-ready evaluation tables and prepare them for plots.

A plotting-ready table is the output of the transformation stage, merged
over runs and usually split by classifier type
(`docs/reference/evaluation_dataframes.md`). Every visualization job reads
it through `load_plotting_ready_table`, so all plots see the same dtypes.
"""

from pathlib import Path
from typing import cast

import pandas as pd

from afabench.evaluation.schemas import PLOTTING_READY_DTYPES


def _cast_column(series: pd.Series, dtype: str) -> pd.Series:
    """Cast a column, turning values that cannot be cast into nulls."""
    if dtype in {"string", "boolean"}:
        return series.astype(dtype)
    return cast("pd.Series", pd.to_numeric(series, errors="coerce")).astype(
        dtype
    )


def load_plotting_ready_table(path: Path) -> pd.DataFrame:
    """
    Read a plotting-ready evaluation table with nullable pandas dtypes.

    Merging tables whose column types disagree upcasts those columns to
    strings, so each known column is cast back to its dtype and values that
    do not parse become null. Absent columns stay absent.
    """
    table = pd.read_parquet(path, dtype_backend="numpy_nullable")
    for name, dtype in PLOTTING_READY_DTYPES.items():
        if name in table.columns:
            table[name] = _cast_column(table.loc[:, name], dtype)
    return table
