"""
Plot the time that each method takes to pretrain, train and evaluate, averaged over all datasets and seeds.

Expected input dataframe with columns:
- afa_method (str): Which method was evaluated. For example "jafa" or "odin".
- dataset (str): Which dataset the method was evaluated on. For example "cube_nm" or "mnist".
- time_pretrain (float | null): How long the pretraining (if applicable) took in seconds.
- time_train (float | null): How long the training (if applicable) took in seconds.
- time_eval (float): How long the evaluation took in seconds.
"""

from pathlib import Path
from typing import cast

import hydra
import numpy as np
import plotnine as p9
import polars as pl
from omegaconf import OmegaConf
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_brewer,
    scale_x_discrete,
    theme,
)

from afabench.plotting.config import PlottingDisplayConfig, PlotTotalTimeConfig

PLOT_FONT_SIZE = 12


def get_mock_df() -> pl.DataFrame:
    """Generate mock dataframe for testing."""
    methods = ["jafa", "odin"]
    datasets = ["cube", "cube_nm"]
    seeds = list(range(1, 6))

    rows = [(m, d, s) for m in methods for d in datasets for s in seeds]

    rng = np.random.default_rng(42)
    df = pl.DataFrame(
        {
            "afa_method": [r[0] for r in rows],
            "dataset": [r[1] for r in rows],
            "pretrain": rng.integers(1, 11, size=len(rows)).tolist(),
            "train": rng.integers(1, 11, size=len(rows)).tolist(),
            "eval": rng.integers(1, 11, size=len(rows)).tolist(),
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


def common_plot_operations(
    p: p9.ggplot, plotting_config: PlottingDisplayConfig
) -> p9.ggplot:
    return (
        p
        + coord_flip()
        + labs(x="Policy", y="Time (s)", fill="Stage")
        + theme(
            text=element_text(
                family=plotting_config.plot_font_family,
                size=PLOT_FONT_SIZE,
            )
        )
        + scale_x_discrete()
        + scale_fill_brewer(
            type="qual",
            palette=plotting_config.color_palette_name,
            labels={
                "pretrain": "Pretraining",
                "train": "Training",
                "eval": "Evaluation",
            },
            breaks=["pretrain", "train", "eval"],
        )
    )


def filter_common_datasets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter dataframe to only include datasets present in all methods.

    This ensures that the averaging plot is fair - methods that only train on
    a subset of datasets won't be penalized by missing data from larger datasets.
    """
    # Get the set of datasets for each method
    datasets_per_method = df.group_by("afa_method").agg(
        pl.col("dataset").unique().sort()
    )

    # Convert to sets and find intersection
    method_dataset_sets = [
        set(row["dataset"])
        for row in datasets_per_method.iter_rows(named=True)
    ]

    # Find datasets that appear in all methods
    common_datasets = set.intersection(*method_dataset_sets)

    # Filter the dataframe to only include common datasets
    return df.filter(pl.col("dataset").is_in(list(common_datasets)))


def get_plots(
    df: pl.DataFrame, plotting_config: PlottingDisplayConfig
) -> tuple[p9.ggplot, p9.ggplot]:
    # Apply name transforms
    df = df.with_columns(
        dataset=pl.col("dataset").replace(
            plotting_config.dataset_name_mapping
        ),
        afa_method=pl.col("afa_method").replace(
            plotting_config.method_name_mapping
        ),
    )

    # For averaging plot, only use datasets present in all methods
    df_common = filter_common_datasets(df)

    # Get display method order based on METHOD_NAME_MAPPING order.
    # After name transformation, values are already display names.
    method_order = [
        plotting_config.method_name_mapping.get(m, m)
        for m in plotting_config.method_name_mapping
    ]
    # Reverse the order to match the desired display.
    method_order.reverse()

    # Filter to only mapped methods present in the data.
    available_methods = set(df["afa_method"].unique().to_list())
    method_order_filtered = [m for m in method_order if m in available_methods]
    # Include methods that are present in data but not in METHOD_NAME_MAPPING.
    # This avoids enum-cast failures when new/experimental methods appear.
    unknown_methods = sorted(available_methods - set(method_order_filtered))
    method_order_filtered.extend(unknown_methods)

    # Cast to Enum with the correct order
    df_common = df_common.with_columns(
        afa_method=pl.col("afa_method").cast(pl.Enum(method_order_filtered))
    )
    df = df.with_columns(
        afa_method=pl.col("afa_method").cast(pl.Enum(method_order_filtered))
    )

    # One plot averaged over datasets (using only common datasets)
    averaged_plot = ggplot(
        df_common.group_by(["afa_method", "stage"]).mean()
    ) + geom_bar(
        aes(x="afa_method", y="time", fill="stage"),
        stat="identity",
    )
    averaged_plot = common_plot_operations(averaged_plot, plotting_config)

    # Another one faceted over datasets (showing all datasets)
    dataset_plot = (
        ggplot(df)
        + geom_bar(
            aes(x="afa_method", y="time", fill="stage"), stat="identity"
        )
        + facet_wrap("dataset", nrow=2, scales="free_x")
    )
    dataset_plot = common_plot_operations(dataset_plot, plotting_config)
    return averaged_plot, dataset_plot


def unpivot(df: pl.DataFrame) -> pl.DataFrame:
    df_long = df.unpivot(
        on=["pretrain", "train", "eval"],
        index=["afa_method", "dataset"],
        variable_name="stage",
        value_name="time",
    )
    # Convert stage to categorical with correct order for stacking
    # This ensures bars are stacked as: eval (bottom), train, pretrain (top)
    stage_order = ["eval", "train", "pretrain"]
    df_long = df_long.with_columns(pl.col("stage").cast(pl.Enum(stage_order)))
    return df_long


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/plotting/plot_total_time",
    config_name="config",
)
def main(cfg: PlotTotalTimeConfig) -> None:
    cfg = cast("PlotTotalTimeConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotTotalTimeConfig)

    df = read_parquet_safe(Path(cfg.input)) if cfg.input else get_mock_df()

    df_long = unpivot(df)

    averaged_plot, dataset_plot = get_plots(
        df=df_long, plotting_config=cfg.plotting
    )

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.formats:
        averaged_plot.save(
            output_folder / f"average_time.{fmt}",
            width=10,
            height=3,
            verbose=False,
        )
        dataset_plot.save(
            output_folder / f"dataset_time.{fmt}",
            width=20,
            height=5,
            verbose=False,
        )


if __name__ == "__main__":
    main()
