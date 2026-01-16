"""
Plot the time that each method takes to pretrain, train and evaluate, averaged over all datasets and seeds.

Expected input dataframe with columns:
- afa_method (str): Which method was evaluated. For example "shim2018" or "zannone2019".
- dataset (str): Which dataset the method was evaluated on. For example "afa_context" or "mnist".
- time_pretrain (float | NaN): How long the pretraining (if applicable) took in seconds.
- time_train (float | NaN): How long the training (if applicable) took in seconds.
- time_eval (float): How long the evaluation took in seconds.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9


def get_mock_df() -> pd.DataFrame:
    """Generate mock dataframe for testing."""
    methods = ["shim2018", "zannone2019"]
    datasets = ["cube", "afa_context"]
    seeds = list(range(1, 6))

    df = pd.DataFrame(
        [(m, d, s) for m in methods for d in datasets for s in seeds],
        columns=["afa_method", "dataset", "seed"],  # pyright: ignore[reportArgumentType]
    )

    rng = np.random.default_rng(42)
    df["time_pretrain"] = rng.integers(1, 11, size=len(df))
    df["time_train"] = rng.integers(1, 11, size=len(df))
    df["time_eval"] = rng.integers(1, 11, size=len(df))

    return df


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read CSV file with appropriate data types."""
    df = pd.read_csv(path)

    # Ensure columns are categorical where appropriate
    df["afa_method"] = df["afa_method"].astype("category")
    df["dataset"] = df["dataset"].astype("category")

    # Ensure numeric columns
    for col in ["time_pretrain", "time_train", "time_eval"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def parse_args() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the time that each method takes to pretrain, train and evaluate."
    )
    parser.add_argument(
        "output_folder",
        help="Output folder where the plot will be saved",
        type=Path,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input CSV file with timing data. If not provided, uses mock data.",
        default=None,
    )

    args = parser.parse_args()
    return args


def df_operations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clone = df.copy()
    # Calculate total time and group by method
    df_clone["time_total"] = (
        df_clone["time_pretrain"]
        + df_clone["time_train"]
        + df_clone["time_eval"]
    )

    df_summary = df_clone.groupby(
        "afa_method", as_index=False, observed=True
    ).agg(
        {
            "time_pretrain": "mean",
            "time_train": "mean",
            "time_eval": "mean",
            "time_total": ["mean", "std"],
        }
    )
    assert isinstance(df_summary, pd.DataFrame)

    # Flatten column names from multi-level index
    df_summary.columns = [
        "afa_method",
        "time_pretrain",
        "time_train",
        "time_eval",
        "time_total_mean",
        "time_total_sd",
    ]

    # Reshape data for stacked bar chart
    df_stage = df_summary[
        ["afa_method", "time_pretrain", "time_train", "time_eval"]
    ].copy()
    df_stage = df_stage.melt(
        id_vars="afa_method",
        value_vars=["time_pretrain", "time_train", "time_eval"],
        var_name="stage",
        value_name="time",
    )
    df_stage["stage"] = df_stage["stage"].str.replace("time_", "", regex=False)
    return df_summary, df_stage


def create_plot(df_summary: pd.DataFrame, df_stage: pd.DataFrame) -> p9.ggplot:
    plot = (
        p9.ggplot()
        + p9.geom_bar(
            data=df_stage,
            mapping=p9.aes(x="afa_method", y="time", fill="stage"),
            stat="identity",
        )
        + p9.geom_errorbar(
            data=df_summary,
            mapping=p9.aes(
                x="afa_method",
                ymin="time_total_mean - time_total_sd",
                ymax="time_total_mean + time_total_sd",
            ),
        )
    )
    return plot


def main() -> None:
    args = parse_args()
    df = read_csv_safe(args.input) if args.input else get_mock_df()
    df_summary, df_stage = df_operations(df)
    plot = create_plot(df_summary, df_stage)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    plot.save(args.output_folder / "total_time.pdf", verbose=False)


if __name__ == "__main__":
    main()
