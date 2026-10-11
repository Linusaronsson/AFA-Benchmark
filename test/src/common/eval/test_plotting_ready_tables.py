"""Plotting-ready evaluation tables load and filter the same way for every plot."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from afabench.evaluation.tables import load_plotting_ready_table


def test_loaded_table_has_nullable_dtypes_and_nulls_for_upcast_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_perf.parquet"
    # Merging tables whose column types disagree upcasts columns to strings
    pq.write_table(
        pa.table(
            {
                "action_performed": pa.array([0, 2], pa.int64()),
                "true_class": pa.array([1, 0], pa.int64()),
                "accumulated_cost": pa.array([1.0, 2.0], pa.float64()),
                "forced_stop": pa.array([False, True]),
                "n_selections_performed": pa.array([1, 2], pa.int64()),
                "predicted_class": pa.array([1, None], pa.int64()),
                "afa_method": pa.array(["jafa", "jafa"]),
                "dataset": pa.array(["cube", "cube"]),
                "initializer": pa.array(["cold", "cold"]),
                "train_seed": pa.array([0, 0], pa.int64()),
                "eval_seed": pa.array([0, 0], pa.int64()),
                "eval_hard_budget": pa.array(["2.0", "None"]),
                "eval_soft_budget_param": pa.array([None, None], pa.float64()),
            }
        ),
        path,
    )

    table = load_plotting_ready_table(path)

    assert {name: str(dtype) for name, dtype in table.dtypes.items()} == {
        "action_performed": "UInt64",
        "true_class": "UInt64",
        "accumulated_cost": "Float64",
        "forced_stop": "boolean",
        "n_selections_performed": "UInt64",
        "predicted_class": "UInt64",
        "afa_method": "string",
        "dataset": "string",
        "initializer": "string",
        "train_seed": "UInt64",
        "eval_seed": "UInt64",
        "eval_hard_budget": "Float64",
        "eval_soft_budget_param": "Float64",
    }
    assert table["eval_hard_budget"].tolist() == [2.0, pd.NA]
    assert table["predicted_class"].tolist() == [1, pd.NA]
