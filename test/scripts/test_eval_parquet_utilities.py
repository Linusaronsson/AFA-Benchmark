from pathlib import Path

import pandas as pd
import pytest

from scripts.misc import (
    add_dataset_split_col,
    combine_soft_budget_results,
    generate_mock_evaluation_data,
)


def test_add_dataset_split_preserves_parquet_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "nested" / "output.parquet"
    original = pd.DataFrame(
        {"prediction": pd.Series([1, None], dtype="Int64")}
    )
    original.to_parquet(input_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        ["add_dataset_split_col.py", str(input_path), str(output_path), "3"],
    )

    add_dataset_split_col.main()

    expected = original.assign(dataset_split=3)
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), expected)


def test_combine_soft_budget_results_preserves_parquet_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = pd.DataFrame(
        {
            "method": ["a", "b"],
            "training_seed": [1, 2],
            "cost_parameter": [0.1, 0.2],
            "dataset": ["example", "example"],
            "features_chosen": [1, 2],
            "predicted_label_builtin": pd.Series([1, None], dtype="Int64"),
            "predicted_label_external": [0, 1],
            "true_label": [1, 1],
        }
    )
    input_paths = [tmp_path / "a.parquet", tmp_path / "b.parquet"]
    for row, path in enumerate(input_paths):
        original.iloc[[row]].to_parquet(path, index=False)
    output_path = tmp_path / "nested" / "combined.parquet"
    monkeypatch.setattr(
        "sys.argv",
        [
            "combine_soft_budget_results.py",
            *(str(path) for path in input_paths),
            str(output_path),
        ],
    )

    combine_soft_budget_results.main()

    pd.testing.assert_frame_equal(pd.read_parquet(output_path), original)


def test_combine_soft_budget_results_validates_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "invalid.parquet"
    output_path = tmp_path / "combined.parquet"
    pd.DataFrame({"method": ["a"]}).to_parquet(input_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        ["combine_soft_budget_results.py", str(input_path), str(output_path)],
    )

    with pytest.raises(ValueError, match="missing columns"):
        combine_soft_budget_results.main()
    assert not output_path.exists()


def test_mock_evaluation_data_saves_nullable_integers_as_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = generate_mock_evaluation_data.generate_mock_data(
        n_methods=2,
        n_seeds=1,
        n_cost_params=1,
        n_datasets=1,
        n_samples=2,
        n_splits=1,
    )
    monkeypatch.setattr(
        generate_mock_evaluation_data, "generate_mock_data", lambda: original
    )
    monkeypatch.chdir(tmp_path)

    generate_mock_evaluation_data.main()

    saved = pd.read_parquet(tmp_path / "eval_results.parquet")
    pd.testing.assert_frame_equal(saved, original)
    assert saved["predicted_label_builtin"].dtype == pd.Int64Dtype()
    assert saved["predicted_label_builtin"].isna().any()
    assert not (tmp_path / "eval_results.csv").exists()
