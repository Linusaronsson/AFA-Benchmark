"""
Plot the time that each method takes to pretrain, train and evaluate, averaged over all datasets and seeds.

Expected input dataframe with columns:
- afa_method (str): Which method was evaluated. For example "shim2018" or "zannone2019".
- dataset (str): Which dataset the method was evaluated on. For example "afa_context" or "mnist".
- time_pretrain (float | null): How long the pretraining (if applicable) took in seconds.
- time_train (float | null): How long the training (if applicable) took in seconds.
- time_eval (float): How long the evaluation took in seconds.
"""

import argparse
from pathlib import Path

import numpy as np
import plotnine as p9
import polars as pl
from plotnine import (
    aes,
    coord_flip,
    geom_bar,
    geom_errorbar,
    ggplot,
    labs,
    scale_fill_discrete,
    scale_x_discrete,
)

METHOD_NAME_MAPPING = {
    # Dummy
    "random_dummy": "Random dummy",
    "sequential_dummy": "Sequential dummy",
    # RL
    "jafa": "JAFA",
    "odin_model_based": "ODIN-MB",
    "odin_model_free": "ODIN-MF",
    "ol_without_mask": "OL",
    "ol_with_mask": "OL+mask",
    "eddi": "EDDI",
    "dime": "DIME",
    "aaco": "AACO",
    "aaco_nn": "AACO+NN",
}


def get_mock_df() -> pl.DataFrame:
    """Generate mock dataframe for testing."""
    methods = ["shim2018", "zannone2019"]
    datasets = ["cube", "afa_context"]
    seeds = list(range(1, 6))

    rows = [(m, d, s) for m in methods for d in datasets for s in seeds]

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "afa_method": [r[0] for r in rows],
            "dataset": [r[1] for r in rows],
            "seed": [r[2] for r in rows],
            "time_pretrain": rng.integers(1, 11, size=len(rows)).tolist(),
            "time_train": rng.integers(1, 11, size=len(rows)).tolist(),
            "time_eval": rng.integers(1, 11, size=len(rows)).tolist(),
        }
    )

    return df


def read_csv_safe(path: Path) -> pl.DataFrame:
    """Read CSV file with appropriate data types."""
    df = pl.read_csv(
        path,
        schema={
            "afa_method": pl.String,
            "dataset": pl.String,
            "time_pretrain": pl.Float64,
            "time_train": pl.Float64,
            "time_eval": pl.Float64,
        },
        null_values=["null"],
    )

    # Treat null times as 0
    df = df.select(
        "afa_method",
        "dataset",
        pl.col("time_pretrain").fill_null(0).alias("pretrain"),
        pl.col("time_train").fill_null(0).alias("train"),
        pl.col("time_eval").fill_null(0).alias("eval"),
    )

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


def process_df(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_long = df.unpivot(
        on=["pretrain", "train", "eval"],
        index=["afa_method", "dataset"],
        variable_name="stage",
        value_name="time",
    )

    # For the individual stages, we only care about average
    df_stages = df_long.group_by(["afa_method", "stage"]).agg(
        pl.col("time").fill_null(0).mean()
    )

    # For the total time, we also want std
    df_total = (
        df.with_columns(
            total=pl.col("pretrain") + pl.col("train") + pl.col("eval")
        )
        .group_by(["afa_method"])
        .agg(
            total_mean=pl.col("total").mean(),
            total_std=pl.col("total").std(),
        )
    )

    return df_stages, df_total


def create_plot(df_stages: pl.DataFrame, df_total: pl.DataFrame) -> p9.ggplot:
    plot = (
        ggplot()
        + geom_bar(
            data=df_stages,
            mapping=aes(x="afa_method", y="time", fill="stage"),
            stat="identity",
        )
        + geom_errorbar(
            data=df_total,
            mapping=aes(
                x="afa_method",
                ymin="total_mean - total_std",
                ymax="total_mean + total_std",
            ),
        )
        + coord_flip()
        + labs(x="Policy", y="Time (s)", fill="Stage")
        + scale_x_discrete(labels=METHOD_NAME_MAPPING)
        + scale_fill_discrete(
            labels={
                "pretrain": "Pretraining",
                "train": "Training",
                "eval": "Evaluation",
            }
        )
    )
    return plot


def main() -> None:
    args = parse_args()
    df = read_csv_safe(args.input) if args.input else get_mock_df()
    df_stages, df_total = process_df(df)
    plot = create_plot(df_stages, df_total)

    args.output_folder.mkdir(parents=True, exist_ok=True)
    plot.save(args.output_folder / "total_time.pdf", verbose=False)


if __name__ == "__main__":
    main()
