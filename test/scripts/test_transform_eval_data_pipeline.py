from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from scripts.misc.transform_eval_data_pipeline import main


def test_transform_eval_data_reads_raw_parquet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "eval_data.parquet"
    output_path = tmp_path / "transformed.parquet"
    pd.DataFrame(
        {
            "prev_selections_performed": [[0], []],
            "action_performed": [1, 0],
            "builtin_predicted_class": [1, 0],
            "external_predicted_class": [0, 1],
            "true_class": [1, 0],
            "accumulated_cost": [1.0, 0.0],
            "idx": [0, 1],
            "forced_stop": [False, False],
            "eval_seed": [2, 2],
            "eval_hard_budget": [3.0, 3.0],
        }
    ).to_parquet(input_path, index=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "transform_eval_data_pipeline.py",
            "--input_path",
            str(input_path),
            "--output_path",
            str(output_path),
            "--method",
            "method_a",
            "--dataset",
            "dataset_a",
            "--initializer",
            "cold",
            "--train_seed",
            "2",
            "--train_hard_budget",
            "3.0",
            "--train_soft_budget_param",
            "null",
            "--eval_soft_budget_param",
            "null",
        ],
    )

    main()

    transformed = pl.read_parquet(output_path)
    assert transformed.select("classifier").to_series().to_list() == [
        "builtin",
        "builtin",
        "external",
        "external",
    ]
    assert transformed.select(
        "n_selections_performed"
    ).to_series().to_list() == [
        1,
        0,
        1,
        0,
    ]
