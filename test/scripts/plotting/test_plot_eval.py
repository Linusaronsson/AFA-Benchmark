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

    df = pd.DataFrame(
        {
            "afa_method": rng.choice(["random_dummy", "aaco"], n),
            "classifier": rng.choice(["builtin", "external"], n),
            "dataset": rng.choice(["cube", "afa_context"], n),
            "predicted_class": rng.integers(0, 5, n),
            "true_class": rng.integers(0, 5, n),
            "train_seed": rng.choice([1, 2], n),
            "eval_seed": rng.choice([1, 2], n),
            "accumulated_cost": rng.uniform(0.5, 15.0, n),
            "eval_hard_budget": np.where(
                rng.random(n) > 0.5, rng.choice([5, 10, 15], n), np.nan
            ),
            "train_soft_budget_param": np.where(
                rng.random(n) > 0.5, rng.choice([0.1, 1.0], n), np.nan
            ),
            "action_performed": rng.choice([0, 1, 2], n),
        }
    )

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
