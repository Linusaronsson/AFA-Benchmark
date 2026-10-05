from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.misc.transform_eval_data_pipeline import main


def parquet_schema(path: Path) -> dict[str, pa.DataType]:
    return {field.name: field.type for field in pq.read_schema(path)}


def run_transform(
    monkeypatch: pytest.MonkeyPatch,
    input_path: Path,
    output_path: Path,
    **overrides: str,
) -> None:
    args = {
        "method": "method_a",
        "dataset": "dataset_a",
        "initializer": "cold",
        "train_seed": "2",
        "train_hard_budget": "null",
        "train_soft_budget_param": "0.5",
        "eval_soft_budget_param": "null",
    } | overrides
    argv = [
        "transform_eval_data_pipeline.py",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
    ]
    for name, value in args.items():
        argv += [f"--{name}", value]
    monkeypatch.setattr("sys.argv", argv)
    main()


def test_transform_pivots_classifiers_and_attaches_run_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    output_path = tmp_path / "transformed.parquet"
    # Raw rows as written by evaluation: instance 0 takes one feature and
    # then stops, instance 1 stops immediately. No external classifier was
    # supplied, and evaluation ran under a soft budget, so both are all-null
    # columns.
    pd.DataFrame(
        {
            "prev_selections_performed": [[], [0], []],
            "action_performed": [1, 0, 0],
            "builtin_predicted_class": [2, 1, 0],
            "external_predicted_class": [None, None, None],
            "true_class": [1, 1, 0],
            "accumulated_cost": [1.0, 1.0, 0.0],
            "idx": [0, 0, 1],
            "forced_stop": [False, True, False],
            "eval_seed": [7, 7, 7],
            "eval_hard_budget": [None, None, None],
        }
    ).to_parquet(input_path, index=False)

    run_transform(monkeypatch, input_path, output_path)

    shared = {
        "afa_method": "method_a",
        "dataset": "dataset_a",
        "initializer": "cold",
        "train_seed": 2,
        "train_hard_budget": None,
        "train_soft_budget_param": 0.5,
        "eval_soft_budget_param": None,
    }
    steps = [
        {
            "action_performed": 1,
            "true_class": 1,
            "accumulated_cost": 1.0,
            "forced_stop": False,
            "eval_seed": 7,
            "eval_hard_budget": None,
            "n_selections_performed": 0,
        },
        {
            "action_performed": 0,
            "true_class": 1,
            "accumulated_cost": 1.0,
            "forced_stop": True,
            "eval_seed": 7,
            "eval_hard_budget": None,
            "n_selections_performed": 1,
        },
        {
            "action_performed": 0,
            "true_class": 0,
            "accumulated_cost": 0.0,
            "forced_stop": False,
            "eval_seed": 7,
            "eval_hard_budget": None,
            "n_selections_performed": 0,
        },
    ]
    expected_rows = [
        step
        | {"classifier": "builtin", "predicted_class": prediction}
        | shared
        for step, prediction in zip(steps, [2, 1, 0], strict=True)
    ] + [
        step | {"classifier": "external", "predicted_class": None} | shared
        for step in steps
    ]
    assert pq.read_table(output_path).to_pylist() == expected_rows
    assert parquet_schema(output_path) == {
        "action_performed": pa.uint64(),
        "true_class": pa.uint64(),
        "accumulated_cost": pa.float64(),
        "forced_stop": pa.bool_(),
        "eval_seed": pa.uint64(),
        "eval_hard_budget": pa.float64(),
        "n_selections_performed": pa.uint64(),
        "classifier": pa.string(),
        "predicted_class": pa.uint64(),
        "afa_method": pa.string(),
        "dataset": pa.string(),
        "initializer": pa.string(),
        "train_seed": pa.uint64(),
        "train_hard_budget": pa.float64(),
        "train_soft_budget_param": pa.float64(),
        "eval_soft_budget_param": pa.float64(),
    }


def test_transform_counts_selections_stored_as_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    output_path = tmp_path / "transformed.parquet"
    pd.DataFrame(
        {
            "prev_selections_performed": ["[]", "[3, 1]"],
            "action_performed": [2, 0],
            "builtin_predicted_class": [0, 1],
            "external_predicted_class": [1, 1],
            "true_class": [1, 1],
            "accumulated_cost": [2.0, 2.0],
            "idx": [0, 0],
            "forced_stop": [False, False],
            "eval_seed": [0, 0],
            "eval_hard_budget": [2.0, 2.0],
        }
    ).to_parquet(input_path, index=False)

    run_transform(
        monkeypatch,
        input_path,
        output_path,
        train_seed="null",
        train_hard_budget="2.0",
        train_soft_budget_param="null",
        eval_soft_budget_param="0.1",
    )

    transformed = pq.read_table(output_path).to_pylist()
    assert [row["n_selections_performed"] for row in transformed] == [
        0,
        2,
        0,
        2,
    ]
    assert [row["predicted_class"] for row in transformed] == [0, 1, 1, 1]
    assert {
        (
            row["train_seed"],
            row["train_hard_budget"],
            row["train_soft_budget_param"],
            row["eval_soft_budget_param"],
        )
        for row in transformed
    } == {(None, 2.0, None, 0.1)}
