from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.misc import (
    merge_dataframes,
    merge_time_results,
    split_eval_perf_by_classifier,
)


def parquet_schema(path: Path) -> dict[str, pa.DataType]:
    return {field.name: field.type for field in pq.read_schema(path)}


def test_split_by_classifier_writes_one_file_per_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "all.parquet"
    pq.write_table(
        pa.table(
            {
                "classifier": ["builtin", "external", "builtin", "external"],
                "predicted_class": pa.array([1, None, 0, 2], pa.uint64()),
                "eval_hard_budget": pa.array(
                    [None, None, 3.0, 3.0], pa.float64()
                ),
                "afa_method": ["a", "a", "b", "b"],
            }
        ),
        input_path,
    )
    builtin_path = tmp_path / "nested" / "builtin.parquet"
    external_path = tmp_path / "nested" / "external.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "split_eval_perf_by_classifier.py",
            "--input_path",
            str(input_path),
            "--output_builtin",
            str(builtin_path),
            "--output_external",
            str(external_path),
        ],
    )

    split_eval_perf_by_classifier.main()

    assert pq.read_table(builtin_path).to_pylist() == [
        {"predicted_class": 1, "eval_hard_budget": None, "afa_method": "a"},
        {"predicted_class": 0, "eval_hard_budget": 3.0, "afa_method": "b"},
    ]
    assert pq.read_table(external_path).to_pylist() == [
        {"predicted_class": None, "eval_hard_budget": None, "afa_method": "a"},
        {"predicted_class": 2, "eval_hard_budget": 3.0, "afa_method": "b"},
    ]
    expected_schema = {
        "predicted_class": pa.uint64(),
        "eval_hard_budget": pa.float64(),
        "afa_method": pa.string(),
    }
    assert parquet_schema(builtin_path) == expected_schema
    assert parquet_schema(external_path) == expected_schema


def test_split_by_classifier_requires_classifier_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "all.parquet"
    pq.write_table(pa.table({"afa_method": ["a"]}), input_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "split_eval_perf_by_classifier.py",
            "--input_path",
            str(input_path),
            "--output_builtin",
            str(tmp_path / "builtin.parquet"),
            "--output_external",
            str(tmp_path / "external.parquet"),
        ],
    )

    with pytest.raises(ValueError, match="classifier"):
        split_eval_perf_by_classifier.main()


def test_merge_time_results_writes_one_row_with_missing_pretraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path = tmp_path / "train.txt"
    eval_path = tmp_path / "eval.txt"
    train_path.write_text("12,5\n")
    eval_path.write_text("3.25")
    output_path = tmp_path / "nested" / "time.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_time_results.py",
            "--output_path",
            str(output_path),
            "--method",
            "jafa",
            "--dataset",
            "cube",
            "--time_train_path",
            str(train_path),
            "--time_eval_path",
            str(eval_path),
        ],
    )

    merge_time_results.main()

    assert pq.read_table(output_path).to_pylist() == [
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "time_pretrain": None,
            "time_train": 12.5,
            "time_eval": 3.25,
        }
    ]
    assert parquet_schema(output_path) == {
        "afa_method": pa.string(),
        "dataset": pa.string(),
        "time_pretrain": pa.float64(),
        "time_train": pa.float64(),
        "time_eval": pa.float64(),
    }


def run_merge(
    monkeypatch: pytest.MonkeyPatch, inputs: list[Path], output: Path
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "merge_dataframes.py",
            *map(str, inputs),
            "--output",
            str(output),
        ],
    )
    merge_dataframes.main()


def test_merge_takes_union_of_columns_and_fills_missing_with_null(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(
        pa.table(
            {
                "seed": pa.array([0, 1], pa.uint64()),
                "method": pa.array(["a", "a"], pa.large_string()),
            }
        ),
        first,
    )
    pq.write_table(
        pa.table(
            {
                "seed": pa.array([2], pa.uint64()),
                "budget": pa.array([3.0], pa.float64()),
            }
        ),
        second,
    )
    output = tmp_path / "merged.parquet"

    run_merge(monkeypatch, [first, second], output)

    assert pq.read_table(output).to_pylist() == [
        {"seed": 0, "method": "a", "budget": None},
        {"seed": 1, "method": "a", "budget": None},
        {"seed": 2, "method": None, "budget": 3.0},
    ]
    assert pq.read_schema(output).types == [
        pa.uint64(),
        pa.string(),
        pa.float64(),
    ]


def test_merge_promotes_conflicting_column_types_to_string(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pq.write_table(pa.table({"value": pa.array([1], pa.int64())}), first)
    pq.write_table(pa.table({"value": pa.array([2.5], pa.float64())}), second)
    output = tmp_path / "merged.parquet"

    run_merge(monkeypatch, [first, second], output)

    assert pq.read_table(output).to_pylist() == [
        {"value": "1"},
        {"value": "2.5"},
    ]
    assert pq.read_schema(output).types == [pa.string()]


def test_merge_rejects_files_without_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.parquet"
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"value": pa.array([1], pa.int64())}), first)
    pq.write_table(pa.table({"value": pa.array([], pa.int64())}), empty)

    with pytest.raises(ValueError, match="empty"):
        run_merge(monkeypatch, [first, empty], tmp_path / "merged.parquet")
