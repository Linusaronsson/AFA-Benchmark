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
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_discrete,
    scale_x_discrete,
)

METHOD_NAME_MAPPING = {
    # Dummy
    "random_dummy": "Random",
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
    # Static
    "cae": "CAE",
    "permutation": "Permutation",
    # Greedy
    "covert2023": "DIME",
    "gadgil2023": "GDFS",
    "ma2018_builtin": "EDDI (builtin)",
    "ma2018_external": "EDDI (external)",
}

DATASET_NAME_MAPPING = {
    "cube": "Cube",
    "cube_nonuniform_costs": "Cube (non-uniform costs)",
    "afa_context": "AFAContext",
    "afa_context_v2": "Cube-NM",
    "afa_context_v2_without_noise": "Noiseless Cube-NM",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "Cube (noiseless)",
    "afa_context_without_noise": "AFAContext (noiseless)",
    "synthetic_mnist_without_noise": "Synthetic MNIST (noiseless)",
    "mnist": "MNIST",
    "actg": "ACTG",
    "bank_marketing": "Bank Marketing",
    "ckd": "CKD",
    "diabetes": "Diabetes",
    "fashion_mnist": "FashionMNIST",
    "miniboone": "MiniBooNE",
    "physionet": "PhysioNet",
    "imagenette": "Imagenette",
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


def read_parquet_safe(path: Path) -> pl.DataFrame:
    """Read CSV file with appropriate data types."""
    df = pl.read_parquet(
        path,
        schema={
            "afa_method": pl.String,
            "dataset": pl.String,
            "time_pretrain": pl.Float64,
            "time_train": pl.Float64,
            "time_eval": pl.Float64,
        },
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
        help="Input parquet file with timing data. If not provided, uses mock data.",
        default=None,
    )

    args = parser.parse_args()
    return args


def common_plot_operations(p: p9.ggplot) -> p9.ggplot:
    return (
        p
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


def get_plots(df: pl.DataFrame) -> tuple[p9.ggplot, p9.ggplot]:
    # Apply name transforms
    df = df.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING),
        afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
    )
    # One plot averaged over datasets
    averaged_plot = ggplot(
        df.group_by(["afa_method", "stage"]).mean()
    ) + geom_bar(
        aes(x="afa_method", y="time", fill="stage"),
        stat="identity",
    )
    averaged_plot = common_plot_operations(averaged_plot)

    # Another one faceted over datasets
    dataset_plot = (
        ggplot(df)
        + geom_bar(
            aes(x="afa_method", y="time", fill="stage"), stat="identity"
        )
        + facet_wrap("dataset", nrow=2, scales="free_x")
    )
    dataset_plot = common_plot_operations(dataset_plot)
    return averaged_plot, dataset_plot


def unpivot(df: pl.DataFrame) -> pl.DataFrame:
    df_long = df.unpivot(
        on=["pretrain", "train", "eval"],
        index=["afa_method", "dataset"],
        variable_name="stage",
        value_name="time",
    )
    return df_long


def main() -> None:
    args = parse_args()
    df = read_parquet_safe(args.input) if args.input else get_mock_df()

    df_long = unpivot(df)

    averaged_plot, dataset_plot = get_plots(df=df_long)

    args.output_folder.mkdir(parents=True, exist_ok=True)
    averaged_plot.save(
        args.output_folder / "average_time.pdf",
        width=10,
        height=3,
        verbose=False,
    )
    dataset_plot.save(
        args.output_folder / "dataset_time.pdf",
        width=20,
        height=5,
        verbose=False,
    )


if __name__ == "__main__":
    main()
