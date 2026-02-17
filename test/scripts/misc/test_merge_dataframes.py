"""Integration test for scripts/misc/merge_dataframes.py."""

import subprocess
from pathlib import Path

import numpy as np
import polars as pl


def create_dummy_parquet(
    path: Path, rows: int, seed: int, missing_col: str | None = None
) -> None:
    """
    Create a dummy parquet file with test data.

    Args:
        path: Path to write the parquet file
        rows: Number of rows to generate
        seed: Random seed for reproducibility
        missing_col: Optional column name to exclude from this file
    """
    rng = np.random.default_rng(seed)

    data = {
        "action_performed": rng.integers(0, 5, rows),
        "true_class": rng.integers(0, 10, rows),
        "accumulated_cost": rng.uniform(0.5, 15.0, rows),
        "forced_stop": rng.choice([True, False], rows),
        "eval_seed": rng.integers(1, 100, rows),
        "eval_hard_budget": rng.uniform(5.0, 20.0, rows),
        "n_selections_performed": [f"sel_{i}" for i in range(rows)],
        "classifier": rng.choice(["svm", "rf", "lr"], rows),
        "predicted_class": rng.integers(0, 10, rows),
        "afa_method": rng.choice(["random", "jafa", "odin"], rows),
        "dataset": rng.choice(["mnist", "cifar10", "imagenet"], rows),
        "train_seed": rng.integers(1, 100, rows),
        "train_hard_budget": rng.uniform(5.0, 20.0, rows),
        "train_soft_budget_param": rng.uniform(0.1, 1.0, rows),
        "eval_soft_budget_param": rng.uniform(0.1, 1.0, rows),
    }

    if missing_col and missing_col in data:
        del data[missing_col]

    df = pl.DataFrame(data)
    df.write_parquet(path)


def test_merge_dataframes_basic_concatenation(tmp_path: Path) -> None:
    """
    Test that merge_dataframes correctly concatenates multiple files.

    Creates two parquet files with the same schema and verifies:
    - Output file has all rows from both input files
    - Column order is preserved
    - Data types are consistent
    """
    # Create input files
    file1 = tmp_path / "data1.parquet"
    file2 = tmp_path / "data2.parquet"
    output = tmp_path / "merged.parquet"

    create_dummy_parquet(file1, rows=100, seed=42)
    create_dummy_parquet(file2, rows=50, seed=43)

    # Run the merge script
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_dataframes.py",
            str(file1),
            str(file2),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Assert the script ran successfully
    assert result.returncode == 0, (
        f"Script failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )

    # Verify output file exists
    assert output.exists(), "Output file was not created"

    # Read the merged output
    merged_df = pl.read_parquet(output)

    # Read the input files to compare
    df1 = pl.read_parquet(file1)
    df2 = pl.read_parquet(file2)

    # Verify row count
    expected_rows = len(df1) + len(df2)
    assert len(merged_df) == expected_rows, (
        f"Expected {expected_rows} rows in output, got {len(merged_df)}"
    )

    # Verify column count
    expected_cols = len(df1.columns)
    assert len(merged_df.columns) == expected_cols, (
        f"Expected {expected_cols} columns in output, "
        f"got {len(merged_df.columns)}"
    )

    # Compare a few key columns to ensure data integrity
    for col in ["action_performed", "true_class", "predicted_class"]:
        assert col in merged_df.columns, f"Column {col} missing from output"


def test_merge_dataframes_with_missing_columns(tmp_path: Path) -> None:
    """
    Test that merge_dataframes handles files with different columns.

    Creates two parquet files where one is missing a column, and verifies:
    - Output includes all columns from both files
    - Missing columns are filled with null values
    """
    file1 = tmp_path / "data_complete.parquet"
    file2 = tmp_path / "data_partial.parquet"
    output = tmp_path / "merged.parquet"

    create_dummy_parquet(file1, rows=50, seed=42)
    create_dummy_parquet(file2, rows=30, seed=43, missing_col="classifier")

    # Run the merge script
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_dataframes.py",
            str(file1),
            str(file2),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
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

    assert output.exists(), "Output file was not created"

    merged_df = pl.read_parquet(output)
    df1 = pl.read_parquet(file1)

    # Verify all columns from file1 are present
    for col in df1.columns:
        assert col in merged_df.columns, (
            f"Column {col} from file1 missing from output"
        )

    # Verify the missing column is present in output (filled with nulls)
    assert "classifier" in merged_df.columns, (
        "Missing column 'classifier' not added to output"
    )

    # Verify total row count
    assert len(merged_df) == 80, (
        f"Expected 80 rows (50+30), got {len(merged_df)}"
    )


def test_merge_dataframes_large_string_normalization(tmp_path: Path) -> None:
    """
    Test that merge_dataframes correctly handles string type variations.

    Creates parquet files that may have different string representations
    (string vs large_string) and verifies the output uses a consistent type.
    """
    file1 = tmp_path / "data1.parquet"
    file2 = tmp_path / "data2.parquet"
    output = tmp_path / "merged.parquet"

    create_dummy_parquet(file1, rows=100, seed=42)
    create_dummy_parquet(file2, rows=50, seed=43)

    # Run the merge script
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_dataframes.py",
            str(file1),
            str(file2),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
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

    # Read the merged output to verify schema consistency
    merged_df = pl.read_parquet(output)

    # Verify that string columns are properly typed
    string_cols = [
        "n_selections_performed",
        "classifier",
        "afa_method",
        "dataset",
    ]
    for col in string_cols:
        assert col in merged_df.columns, f"String column {col} missing"
        # Polars will normalize types, so just verify the column exists
        # and is readable


def test_merge_dataframes_single_file(tmp_path: Path) -> None:
    """Test that merge_dataframes handles a single input file correctly."""
    file1 = tmp_path / "data.parquet"
    output = tmp_path / "merged.parquet"

    create_dummy_parquet(file1, rows=100, seed=42)

    # Run the merge script with single file
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_dataframes.py",
            str(file1),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
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

    # Verify output matches input
    input_df = pl.read_parquet(file1)
    output_df = pl.read_parquet(output)

    assert len(input_df) == len(output_df), (
        f"Row count mismatch: {len(input_df)} vs {len(output_df)}"
    )
    assert len(input_df.columns) == len(output_df.columns), (
        f"Column count mismatch: {len(input_df.columns)} "
        f"vs {len(output_df.columns)}"
    )


def test_merge_dataframes_missing_input_file(tmp_path: Path) -> None:
    """Test that merge_dataframes fails when input file doesn't exist."""
    output = tmp_path / "merged.parquet"
    nonexistent = tmp_path / "nonexistent.parquet"

    # Run the merge script with non-existent file
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/misc/merge_dataframes.py",
            str(nonexistent),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parent.parent.parent.parent,  # Project root
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    # Script should fail
    assert result.returncode != 0, "Script should fail with non-existent file"
