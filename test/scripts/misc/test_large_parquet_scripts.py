"""Regression tests for streaming parquet workflow scripts."""

import subprocess
from pathlib import Path

import polars as pl


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent


def _create_eval_perf_parquet(path: Path) -> None:
    pl.DataFrame(
        {
            "action_performed": [0, 0, 0, 0],
            "true_class": [0, 1, 0, 1],
            "accumulated_cost": [1.0, 2.0, 1.5, 2.5],
            "forced_stop": [False, False, True, True],
            "eval_seed": [0, 0, 1, 1],
            "eval_hard_budget": [3.0, 3.0, 5.0, 5.0],
            "n_selections_performed": [1, 2, 1, 2],
            "classifier": ["builtin", "external", "builtin", "external"],
            "predicted_class": [0, 1, 1, 1],
            "afa_method": ["aaco_full", "aaco_full", "aaco_dr", "aaco_dr"],
            "dataset": ["actg", "actg", "ckd", "ckd"],
            "initializer": ["cold", "cold", "cold", "cold"],
            "train_seed": [0, 0, 1, 1],
            "train_hard_budget": [3.0, 3.0, 5.0, 5.0],
            "train_soft_budget_param": [None, None, None, None],
            "eval_soft_budget_param": [None, None, None, None],
        }
    ).write_parquet(path)


def test_split_eval_perf_by_classifier(tmp_path: Path) -> None:
    input_path = tmp_path / "eval_perf_all.parquet"
    output_builtin = tmp_path / "builtin.parquet"
    output_external = tmp_path / "external.parquet"
    _create_eval_perf_parquet(input_path)

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/split_eval_perf_by_classifier.py",
            "--input_path",
            str(input_path),
            "--output_builtin",
            str(output_builtin),
            "--output_external",
            str(output_external),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert output_builtin.exists()
    assert output_external.exists()

    builtin_df = pl.read_parquet(output_builtin)
    external_df = pl.read_parquet(output_external)

    assert "classifier" not in builtin_df.columns
    assert "classifier" not in external_df.columns
    assert builtin_df.height == 2
    assert external_df.height == 2
    assert set(builtin_df["dataset"].to_list()) == {"actg", "ckd"}
    assert set(external_df["dataset"].to_list()) == {"actg", "ckd"}


def test_merge_eval_perf_comparison(tmp_path: Path) -> None:
    first_input = tmp_path / "first.parquet"
    second_input = tmp_path / "second.parquet"
    output_path = tmp_path / "merged.parquet"

    pl.DataFrame(
        {
            "dataset": ["actg", "ckd"],
            "afa_method": ["aaco_full", "aaco_dr"],
            "mean_metric": [0.7, 0.8],
        }
    ).write_parquet(first_input)
    pl.DataFrame(
        {
            "dataset": ["actg"],
            "afa_method": ["aaco_zero_fill"],
            "std_metric": [0.05],
        }
    ).write_parquet(second_input)

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_eval_perf_comparison.py",
            "--input",
            "cold",
            str(first_input),
            "--input",
            "mcar_p03",
            str(second_input),
            "--group-column",
            "initializer_group",
            "--output",
            str(output_path),
        ],
        cwd=_project_root(),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert output_path.exists()

    merged_df = pl.read_parquet(output_path)
    assert merged_df.height == 3
    assert "initializer_group" in merged_df.columns
    assert set(merged_df["initializer_group"].to_list()) == {
        "cold",
        "mcar_p03",
    }
    assert "mean_metric" in merged_df.columns
    assert "std_metric" in merged_df.columns
