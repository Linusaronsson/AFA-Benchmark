"""Plotting-ready evaluation tables load and filter the same way for every plot."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from afabench.evaluation.tables import (
    apply_exclusions,
    combine_soft_budget_params,
    keep_largest_hard_budget,
    keep_smallest_soft_budget_param,
    load_plotting_ready_table,
    metrics_at_stop,
)


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


def budget_rows(
    rows: list[tuple[str, str, float | None, float | None]],
) -> pd.DataFrame:
    """Rows of (afa_method, dataset, eval_hard_budget, soft_budget_param)."""
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["afa_method", "dataset", "eval_hard_budget", "soft_budget_param"]
        ),
    ).astype(
        {
            "afa_method": "string",
            "dataset": "string",
            "eval_hard_budget": "Float64",
            "soft_budget_param": "Float64",
        }
    )


def test_keep_largest_hard_budget_keeps_each_datasets_largest() -> None:
    table = budget_rows(
        [
            ("jafa", "cube", 1.0, None),
            ("gdfs", "cube", 3.0, None),
            ("jafa", "mnist", 2.0, None),
            # Soft-budget rows have no hard budget to compare
            ("jafa", "mnist", None, 0.1),
        ]
    )

    kept = keep_largest_hard_budget(table)

    assert kept.to_numpy().tolist() == [
        ["gdfs", "cube", 3.0, pd.NA],
        ["jafa", "mnist", 2.0, pd.NA],
    ]


def test_keep_smallest_soft_budget_param_per_method_and_dataset() -> None:
    table = budget_rows(
        [
            ("jafa", "cube", None, 0.1),
            ("jafa", "cube", None, 0.01),
            ("gdfs", "cube", None, 0.5),
            ("jafa", "mnist", None, 1.0),
            ("jafa", "mnist", 2.0, None),
        ]
    )

    kept = keep_smallest_soft_budget_param(table)

    assert kept.to_numpy().tolist() == [
        ["jafa", "cube", pd.NA, 0.01],
        ["gdfs", "cube", pd.NA, 0.5],
        ["jafa", "mnist", pd.NA, 1.0],
    ]


def soft_budget_params(
    train: list[float | None], evaluation: list[float | None]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_soft_budget_param": train,
            "eval_soft_budget_param": evaluation,
        }
    ).astype("Float64")


def test_combine_soft_budget_params_takes_whichever_is_set() -> None:
    table = soft_budget_params([0.1, None, None], [None, 0.5, None])

    combined = combine_soft_budget_params(table)

    assert combined.columns.tolist() == ["soft_budget_param"]
    assert combined["soft_budget_param"].tolist() == [0.1, 0.5, pd.NA]


def test_combine_soft_budget_params_rejects_rows_with_both_set() -> None:
    table = soft_budget_params([0.1, 0.2], [None, 0.5])

    with pytest.raises(ValueError, match=r"train_soft_budget_param=0\.2"):
        combine_soft_budget_params(table)


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_metrics_at_stop_score_each_seed_and_budget() -> None:
    table = pd.DataFrame(
        {
            "afa_method": ["jafa"] * 3,
            "dataset": ["cube"] * 3,
            "train_seed": [0, 0, 1],
            "eval_seed": [0, 0, 0],
            "eval_hard_budget": [2.0, 2.0, 2.0],
            "soft_budget_param": [None, None, None],
            "true_class": [1, 0, 1],
            "predicted_class": [1, 1, 1],
            "accumulated_cost": [1.0, 2.0, 2.0],
        }
    ).astype({"soft_budget_param": "Float64"})

    metrics = metrics_at_stop(table)

    assert metrics.to_dict("records") == [
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "train_seed": 0,
            "eval_seed": 0,
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            "accuracy": 0.5,
            # Macro F1 of classes 0 (F1 0) and 1 (F1 2/3)
            "f_score": pytest.approx(1 / 3),
            "avg_accumulated_cost": 1.5,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "train_seed": 1,
            "eval_seed": 0,
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            "accuracy": 1.0,
            "f_score": 1.0,
            "avg_accumulated_cost": 2.0,
        },
    ]


def test_excluded_method_is_kept_only_beside_its_allowed_methods() -> None:
    exclusions = {
        ("odin_model_based", "diabetes"): frozenset(
            [frozenset(["odin_model_based", "odin_model_free"])]
        )
    }
    odin_pair = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("odin_model_free", "diabetes", 1.0, None),
        ]
    )
    with_jafa = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("jafa", "diabetes", 1.0, None),
            ("odin_model_based", "cube", 1.0, None),
        ]
    )

    assert apply_exclusions(odin_pair, exclusions).equals(odin_pair)
    assert apply_exclusions(with_jafa, exclusions)[
        ["afa_method", "dataset"]
    ].to_numpy().tolist() == [
        ["jafa", "diabetes"],
        ["odin_model_based", "cube"],
    ]
