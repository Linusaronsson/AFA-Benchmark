"""Integration test for scripts/plotting/plot_eval.py."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def test_plot_eval_script_runs_without_crashing(tmp_path: Path) -> None:
    """
    Integration test: verify plot_eval.py runs end-to-end without crashing.

    Creates dummy CSV file with expected data format and runs the script
    via subprocess to ensure it completes successfully.
    """
    # Generate dummy evaluation data
    rng = np.random.default_rng(42)
    n = 500

    # Create hard budget and soft budget data separately to ensure
    # that exactly one is set (not both) as per the script's assertion
    hard_budget_rows = n // 2
    soft_budget_rows = n - hard_budget_rows

    # Hard budget data: train_soft_budget_param is null
    hard_budget_df = pd.DataFrame(
        {
            "afa_method": rng.choice(
                ["random_dummy", "aaco"], hard_budget_rows
            ),
            "dataset": rng.choice(["cube", "afa_context"], hard_budget_rows),
            "predicted_class": rng.integers(0, 5, hard_budget_rows),
            "true_class": rng.integers(0, 5, hard_budget_rows),
            "train_seed": rng.choice([1, 2], hard_budget_rows),
            "eval_seed": rng.choice([1, 2], hard_budget_rows),
            "accumulated_cost": rng.uniform(0.5, 15.0, hard_budget_rows),
            "eval_hard_budget": rng.choice(
                [5.0, 10.0, 15.0], hard_budget_rows
            ),
            "train_soft_budget_param": np.nan,
            "eval_soft_budget_param": np.nan,
            "action_performed": 0,
            "idx": 0,
            "forced_stop": False,
            "selections_performed": 0,
            "train_hard_budget": np.nan,
        }
    )

    # Soft budget data: eval_hard_budget is null
    soft_budget_df = pd.DataFrame(
        {
            "afa_method": rng.choice(
                ["random_dummy", "aaco"], soft_budget_rows
            ),
            "dataset": rng.choice(["cube", "afa_context"], soft_budget_rows),
            "predicted_class": rng.integers(0, 5, soft_budget_rows),
            "true_class": rng.integers(0, 5, soft_budget_rows),
            "train_seed": rng.choice([1, 2], soft_budget_rows),
            "eval_seed": rng.choice([1, 2], soft_budget_rows),
            "accumulated_cost": rng.uniform(0.5, 15.0, soft_budget_rows),
            "eval_hard_budget": np.nan,
            "train_soft_budget_param": rng.choice(
                [0.1, 0.5, 1.0], soft_budget_rows
            ),
            "eval_soft_budget_param": np.nan,
            "action_performed": 0,
            "idx": 0,
            "forced_stop": False,
            "selections_performed": 0,
            "train_hard_budget": np.nan,
        }
    )

    # Combine data
    df = pd.concat([hard_budget_df, soft_budget_df], ignore_index=True)

    # Save dummy CSV
    input_csv = tmp_path / "dummy_eval.csv"
    df.to_csv(input_csv, index=False)

    # Create output directory
    output_dir = tmp_path / "plots"
    output_dir.mkdir()

    # Run the script
    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/plotting/plot_eval.py",
            str(input_csv),
            str(output_dir),
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
