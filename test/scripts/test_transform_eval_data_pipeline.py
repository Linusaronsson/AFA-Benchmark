from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from afabench.evaluation.provenance import (
    evaluation_table_provenance,
    save_evaluation_table,
)
from afabench.testing.provenance import placeholder_provenance
from scripts.misc.transform_eval_data_pipeline import (
    IdentityArgumentMismatchError,
    main,
)

SNAKEMAKE_ARGUMENTS = {
    "method": "method_a",
    "dataset": "dataset_a",
    "initializer": "cold",
    "train_seed": "2",
    "train_hard_budget": "null",
    "train_soft_budget_param": "0.5",
    "eval_soft_budget_param": "null",
}


def parquet_schema(path: Path) -> dict[str, pa.DataType]:
    return {field.name: field.type for field in pq.read_schema(path)}


def run_transform(
    monkeypatch: pytest.MonkeyPatch,
    input_path: Path,
    output_path: Path,
    arguments: dict[str, str] = SNAKEMAKE_ARGUMENTS,
    **overrides: str,
) -> None:
    args = arguments | overrides
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


@pytest.mark.parametrize("compact", [False, True])
def test_transform_fills_identity_of_a_pre_provenance_table_from_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact: bool,
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    output_path = tmp_path / "transformed.parquet"
    # Raw rows as written by evaluation before ADR 0002 (and, not compact,
    # before episode logs were compact): instance 0 takes one feature and
    # then stops, instance 1 stops immediately. No external classifier was
    # supplied, and evaluation ran under a soft budget, so both are all-null
    # columns.
    frame = pd.DataFrame(
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
    )
    if compact:
        frame = frame.drop(columns=["idx", "prev_selections_performed"])
        frame["episode_id"] = [0, 0, 1]
        frame["generation_index"] = [4, 4, 9]
        frame["split_index"] = [1, 1, 0]
        frame["step"] = [0, 1, 0]
    frame.to_parquet(input_path, index=False)

    run_transform(monkeypatch, input_path, output_path)

    shared = {
        "afa_method": "method_a",
        "dataset": "dataset_a",
        # No argument names these, so they stay unknown
        "dataset_realization_index": None,
        "eval_split": None,
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
    # Legacy logs have no instance identity, compact logs carry it through
    identities = (
        [(4, 1), (4, 1), (9, 0)] if compact else [(None, None)] * len(steps)
    )
    steps = [
        {"generation_index": generation_index, "split_index": split_index}
        | step
        for (generation_index, split_index), step in zip(
            identities, steps, strict=True
        )
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
    assert evaluation_table_provenance(output_path) is None
    assert parquet_schema(output_path) == {
        "generation_index": pa.uint64(),
        "split_index": pa.uint64(),
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
        "dataset_realization_index": pa.uint64(),
        "eval_split": pa.string(),
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


def write_identified_table(path: Path) -> None:
    """Two episodes as the evaluator saves them, with provenance."""
    frame = pd.DataFrame(
        {
            "episode_id": [0, 0, 1],
            "generation_index": [4, 4, 9],
            "split_index": [1, 1, 0],
            "step": [0, 1, 0],
            "action_performed": [1, 0, 0],
            "builtin_predicted_class": [None, None, None],
            "external_predicted_class": [2, 1, 0],
            "true_class": [1, 1, 0],
            "accumulated_cost": [1.0, 1.0, 0.0],
            "forced_stop": [False, True, False],
        }
    )
    identity = {
        "afa_method": ("method_a", "string"),
        "dataset": ("dataset_a", "string"),
        "dataset_realization_index": (3, "UInt64"),
        "eval_split": ("val", "string"),
        "initializer": ("cold", "string"),
        "train_seed": (2, "UInt64"),
        "train_hard_budget": (None, "Float64"),
        "train_soft_budget_param": (0.5, "Float64"),
        "eval_seed": (7, "UInt64"),
        "eval_hard_budget": (None, "Float64"),
        "eval_soft_budget_param": (None, "Float64"),
    }
    for name, (value, dtype) in identity.items():
        frame[name] = pd.Series([value] * 3, dtype=object).astype(dtype)
    save_evaluation_table(
        frame,
        path,
        provenance=placeholder_provenance(
            "evaluation",
            seed=7,
            method_name="method_a",
            dataset_key="dataset_a",
            dataset_realization_index=3,
            split="val",
        ),
    )


@pytest.mark.parametrize(
    "arguments", [{}, SNAKEMAKE_ARGUMENTS], ids=["hand_run", "snakemake"]
)
def test_transform_takes_identity_from_raw_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, str],
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    output_path = tmp_path / "transformed.parquet"
    write_identified_table(input_path)

    run_transform(monkeypatch, input_path, output_path, arguments)

    transformed = pq.read_table(output_path).to_pylist()
    assert len(transformed) == 6
    identity = {
        "afa_method": "method_a",
        "dataset": "dataset_a",
        "dataset_realization_index": 3,
        "eval_split": "val",
        "initializer": "cold",
        "train_seed": 2,
        "train_hard_budget": None,
        "train_soft_budget_param": 0.5,
        "eval_seed": 7,
        "eval_hard_budget": None,
        "eval_soft_budget_param": None,
    }
    for row in transformed:
        assert {name: row[name] for name in identity} == identity
    assert [
        (row["generation_index"], row["classifier"], row["predicted_class"])
        for row in transformed
    ] == [
        (4, "builtin", None),
        (4, "builtin", None),
        (9, "builtin", None),
        (4, "external", 2),
        (4, "external", 1),
        (9, "external", 0),
    ]
    assert evaluation_table_provenance(
        output_path
    ) == evaluation_table_provenance(input_path)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("method", "method_b"),
        ("dataset", "dataset_b"),
        ("initializer", "warm"),
        ("train_seed", "3"),
        ("train_soft_budget_param", "null"),
    ],
)
def test_transform_rejects_an_argument_that_disagrees_with_its_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    value: str,
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    write_identified_table(input_path)

    with pytest.raises(IdentityArgumentMismatchError, match=value):
        run_transform(
            monkeypatch,
            input_path,
            tmp_path / "transformed.parquet",
            {argument: value},
        )


def test_transform_fills_a_null_identity_column_from_its_argument(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A method bundle written before ADR 0002 leaves training identity null
    input_path = tmp_path / "eval_data.parquet"
    write_identified_table(input_path)
    raw = pd.read_parquet(input_path)
    raw["train_seed"] = pd.Series([None] * len(raw), dtype="UInt64")
    save_evaluation_table(
        raw, input_path, provenance=evaluation_table_provenance(input_path)
    )
    output_path = tmp_path / "transformed.parquet"

    run_transform(monkeypatch, input_path, output_path, {"train_seed": "5"})

    assert set(pd.read_parquet(output_path)["train_seed"]) == {5}
