from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import plotnine as p9
import polars as pl
from plotnine import (
    aes,
    facet_wrap,
    geom_errorbar,
    geom_errorbarh,
    geom_line,
    geom_point,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_brewer,
    scale_fill_brewer,
    theme,
)
from sklearn.metrics import accuracy_score, f1_score

from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    DATASET_NAME_MAPPING,
    DATASET_SETS,
    DATASETS_WITH_F_SCORE,
    METHOD_NAME_MAPPING,
    PLOT_HEIGHT,
    PLOT_WIDTH,
)

SUBPLOT_HEIGHT = 2.0

EXCLUSION_MAPPING = {
    # Exclude method from dataset by default, except when methods set matches
    # Structure: (method, dataset) -> frozenset of allowed method combinations
    # If the set of methods in the dataset matches one of the frozensets,
    # the method is NOT excluded. Otherwise, it IS excluded.
    ("odin_model_based", "diabetes"): frozenset(
        [frozenset(["odin_model_based", "odin_model_free"])]
    ),
    ("odin_model_based", "miniboone"): frozenset(
        [frozenset(["odin_model_based", "odin_model_free"])]
    ),
}

# When we plot AACO with other methods on miniboone in the soft-budget case, it becomes unreadable
#
METHOD_SET_MAIN_WITHOUT_AACO = frozenset(
    [
        "jafa",
        "odin_model_based",
        "ol_with_mask",
        "covert2023",
        "ma2018_external",
        "gadgil2023",
        "cae",
    ]
)

Y_RANGE_OVERRIDE = {
    "hard_budget": {},
    "hard_budget_traj": {},
    "soft_budget_lines": {
        ("aaco", METHOD_SET_MAIN_WITHOUT_AACO): {"miniboone": (0.85, 1.0)},
    },
    "soft_budget_2d_errors": {
        ("aaco", METHOD_SET_MAIN_WITHOUT_AACO): {"miniboone": (0.85, 1.0)},
    },
}

DATASETS = DATASET_NAME_MAPPING.keys()

DATASET_NAME_MAPPING_INCLUDING_METRIC = {
    dataset: f"{name} ({'F1' if dataset in DATASETS_WITH_F_SCORE else 'Accuracy'})"
    for dataset, name in DATASET_NAME_MAPPING.items()
}


def create_dummy_data() -> pl.DataFrame:  # noqa: C901
    """Create minimal dummy data for testing both hard and soft budget plots."""
    rows = []

    methods = ["jafa", "odin_model_based"]
    datasets = ["cube_without_noise", "synthetic_mnist_without_noise"]
    train_seeds = [0, 1, 2]
    eval_seeds = [0, 1, 2]
    soft_params = [0.001, 0.01, 0.1]
    hard_budgets = [1.0, 2.0, 3.0]

    # Hard budget data
    for method in methods:
        for dataset in datasets:
            for train_seed in train_seeds:
                for eval_seed in eval_seeds:
                    for budget in hard_budgets:
                        rows.append(  # noqa: PERF401
                            {
                                "action_performed": 0,
                                "true_class": np.random.randint(0, 3),  # noqa: NPY002
                                "accumulated_cost": float(
                                    budget + np.random.normal(0, 0.1)  # noqa: NPY002
                                ),
                                "idx": 0,
                                "forced_stop": False,
                                "eval_seed": eval_seed,
                                "eval_hard_budget": float(budget),
                                "soft_budget_param": None,
                                "selections_performed": 0,
                                "afa_method": method,
                                "dataset": dataset,
                                "train_seed": train_seed,
                                "train_hard_budget": None,
                                "predicted_class": np.random.randint(0, 3),  # noqa: NPY002
                            }
                        )

    # Soft budget data - cost varies significantly with param
    for method in methods:
        for dataset in datasets:
            for train_seed in train_seeds:
                for eval_seed in eval_seeds:
                    for param in soft_params:
                        # Different params should have clearly different costs
                        base_cost = 2.0 + param * 20.0
                        cost = base_cost + np.random.normal(0, 0.5)  # noqa: NPY002
                        rows.append(
                            {
                                "action_performed": 0,
                                "true_class": np.random.randint(0, 3),  # noqa: NPY002
                                "accumulated_cost": float(cost),
                                "idx": 0,
                                "forced_stop": False,
                                "eval_seed": eval_seed,
                                "eval_hard_budget": None,
                                "soft_budget_param": float(param),
                                "selections_performed": 0,
                                "afa_method": method,
                                "dataset": dataset,
                                "train_seed": train_seed,
                                "train_hard_budget": None,
                                "predicted_class": np.random.randint(0, 3),  # noqa: NPY002
                            }
                        )

    return pl.DataFrame(
        rows,
        schema={
            "action_performed": pl.UInt64,
            "true_class": pl.UInt64,
            "accumulated_cost": pl.Float64,
            "idx": pl.UInt64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.UInt64,
            "eval_hard_budget": pl.Float64,
            "soft_budget_param": pl.Float64,
            "selections_performed": pl.UInt64,
            "afa_method": pl.String,
            "dataset": pl.String,
            "train_seed": pl.UInt64,
            "train_hard_budget": pl.Float64,
            "predicted_class": pl.UInt64,
        },
    )


def get_metrics_at_stop_action(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(
        "afa_method",
        "dataset",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "soft_budget_param",
    ).map_groups(
        lambda group_df: pl.DataFrame(
            {
                "afa_method": [group_df["afa_method"].first()],
                "dataset": [group_df["dataset"].first()],
                "train_seed": [group_df["train_seed"].first()],
                "eval_seed": [group_df["eval_seed"].first()],
                "eval_hard_budget": [group_df["eval_hard_budget"].first()],
                "soft_budget_param": [group_df["soft_budget_param"].first()],
                "accuracy": [
                    accuracy_score(
                        group_df["true_class"], group_df["predicted_class"]
                    )
                ],
                "f_score": [
                    f1_score(
                        group_df["true_class"],
                        group_df["predicted_class"],
                        average="macro",
                    )
                ],
                "avg_accumulated_cost": [group_df["accumulated_cost"].mean()],
            },
            schema={
                "afa_method": pl.String,
                "dataset": pl.String,
                "train_seed": pl.Int64,
                "eval_seed": pl.Int64,
                "eval_hard_budget": pl.Float64,
                "soft_budget_param": pl.Float64,
                "accuracy": pl.Float64,
                "f_score": pl.Float64,
                "avg_accumulated_cost": pl.Float64,
            },
        )
    )


def get_metrics_at_every_action(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by(
        "afa_method",
        "dataset",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "soft_budget_param",
        "n_selections_performed",
    ).map_groups(
        lambda group_df: pl.DataFrame(
            {
                "afa_method": [group_df["afa_method"].first()],
                "dataset": [group_df["dataset"].first()],
                "train_seed": [group_df["train_seed"].first()],
                "eval_seed": [group_df["eval_seed"].first()],
                "eval_hard_budget": [group_df["eval_hard_budget"].first()],
                "soft_budget_param": [group_df["soft_budget_param"].first()],
                "n_selections_performed": [
                    group_df["n_selections_performed"].first()
                ],
                "accuracy": [
                    accuracy_score(
                        group_df["true_class"], group_df["predicted_class"]
                    )
                ],
                "f_score": [
                    f1_score(
                        group_df["true_class"],
                        group_df["predicted_class"],
                        average="macro",
                    )
                ],
            },
            schema={
                "afa_method": pl.String,
                "dataset": pl.String,
                "train_seed": pl.Int64,
                "eval_seed": pl.Int64,
                "eval_hard_budget": pl.Float64,
                "soft_budget_param": pl.Float64,
                "n_selections_performed": pl.UInt64,
                "accuracy": pl.Float64,
                "f_score": pl.Float64,
            },
        )
    )


def read_parquet(input_csv_path: Path) -> pl.DataFrame:
    return pl.read_parquet(
        input_csv_path,
        schema={
            "action_performed": pl.UInt64,
            "true_class": pl.UInt64,
            "accumulated_cost": pl.Float64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.UInt64,
            "eval_hard_budget": pl.Float64,
            "n_selections_performed": pl.UInt64,
            "predicted_class": pl.UInt64,
            "afa_method": pl.String,
            "dataset": pl.String,
            "train_seed": pl.UInt64,
            "train_hard_budget": pl.Float64,
            "train_soft_budget_param": pl.Float64,
            "eval_soft_budget_param": pl.Float64,
        },
    )


def get_variance_of_metrics_and_cost(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(
        "afa_method", "dataset", "eval_hard_budget", "soft_budget_param"
    ).agg(
        mean_accuracy=pl.col("accuracy").mean(),
        std_accuracy=pl.col("accuracy").std(),
        mean_f_score=pl.col("f_score").mean(),
        std_f_score=pl.col("f_score").std(),
        mean_avg_accumulated_cost=pl.col("avg_accumulated_cost").mean(),
        std_avg_accumulated_cost=pl.col("avg_accumulated_cost").std(),
    )
    return df


def get_variance_of_metrics(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(
        "afa_method",
        "dataset",
        "eval_hard_budget",
        "soft_budget_param",
        "n_selections_performed",
    ).agg(
        mean_accuracy=pl.col("accuracy").mean(),
        std_accuracy=pl.col("accuracy").std(),
        mean_f_score=pl.col("f_score").mean(),
        std_f_score=pl.col("f_score").std(),
    )
    return df


def apply_exclusions(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply exclusion mappings to filter out methods under specific conditions.

    By default, excludes a method from a dataset. Only includes it if the set
    of methods present in the dataset matches one of the allowed combinations.

    Structure of EXCLUSION_MAPPING:
        (method, dataset) -> frozenset([frozenset(allowed_combinations)])

    Example:
        ("odin_model_based", "diabetes") -> frozenset([
            frozenset(["odin_model_based", "odin_model_free"])
        ])

    This means: Exclude odin_model_based from diabetes, EXCEPT when the
    only methods present are odin_model_based and odin_model_free.

    Args:
        df: Input dataframe with 'afa_method' and 'dataset' columns

    Returns:
        Filtered dataframe with exclusions applied
    """
    # Get unique combinations of method and dataset in the input
    method_dataset_pairs = (
        df.select(["afa_method", "dataset"]).unique().to_dicts()
    )

    # Check each (method, dataset) pair and filter if needed
    for pair in method_dataset_pairs:
        method = pair["afa_method"]
        dataset = pair["dataset"]
        key = (method, dataset)

        # Check if this method-dataset pair has exclusion rules
        if key in EXCLUSION_MAPPING:
            # Get the set of methods present for this dataset
            methods_in_dataset = frozenset(
                df.filter(pl.col("dataset") == dataset)["afa_method"]
                .unique()
                .to_list()
            )

            # Get the allowed method combinations for this exclusion rule
            allowed_combinations = EXCLUSION_MAPPING[key]

            # Only exclude if methods_in_dataset doesn't match any allowed combo
            if methods_in_dataset not in allowed_combinations:
                df = df.filter(
                    ~(
                        (pl.col("afa_method") == method)
                        & (pl.col("dataset") == dataset)
                    )
                )

    return df


def apply_y_range_override(
    plot: p9.ggplot,
    mode: str,
    df: pl.DataFrame | None = None,
) -> p9.ggplot:
    """
    Apply y-axis range overrides to faceted plot subplots.

    This function modifies the y-axis limits of individual facet subplots
    without changing the underlying data. Each subplot's y-axis is
    independently set based on the Y_RANGE_OVERRIDE configuration.

    Note: This function returns a custom plot object that applies
     overrides when .draw() or .save() is called.

    Structure of Y_RANGE_OVERRIDE:
        {
            mode: {
                key: {
                    dataset: (y_min, y_max)
                }
            }
        }
        where key can be:
        - frozenset(methods): applies if data is subset of methods
        - (method, frozenset(methods)): applies if method AND any from frozenset present

    Example:
        Y_RANGE_OVERRIDE = {
            "hard_budget": {
                frozenset(["jafa", "odin_model_based"]): {
                    "miniboone": (0.85, 1.0)
                },
                ("aaco", frozenset(["jafa", "odin_model_based"])): {
                    "miniboone": (0.85, 1.0)
                }
            }
        }

    Args:
        plot: The plotnine ggplot object with faceted subplots
        mode: The plot mode ("hard_budget", "hard_budget_traj",
              "soft_budget_lines", or "soft_budget_2d_errors")
        df: The dataframe used to create the plot (needed to detect methods)

    Returns:
        A wrapped plot object that applies y-range overrides when drawn/saved
    """
    # Check if this mode has any overrides
    if mode not in Y_RANGE_OVERRIDE:
        return plot

    # Get the overrides for this mode
    mode_overrides = Y_RANGE_OVERRIDE[mode]
    if not mode_overrides:
        return plot

    # Wrap the draw method to apply overrides after drawing
    original_draw = plot.draw

    def draw_with_overrides(
        show: bool = False,  # noqa: FBT002
    ) -> matplotlib.figure.Figure:  # noqa: F821
        fig = original_draw(show=show)
        _apply_overrides_to_figure(fig, mode_overrides, df)
        return fig

    plot.draw = draw_with_overrides  # type: ignore[assignment]

    return plot


def _get_applicable_overrides(
    mode_overrides: dict[Any, Any],  # type: ignore[type-arg]
    df: pl.DataFrame | None = None,
) -> dict[Any, Any] | None:
    """
    Determine which method set's overrides apply based on data.

    Supports two key formats:
    1. frozenset: Override applies if data methods are subset of the frozenset
    2. tuple: (method, frozenset) - Override applies if method AND at least
       one method from frozenset are both present in data

    Args:
        mode_overrides: Dictionary mapping method sets to dicts of
            dataset -> (y_min, y_max) tuples
        df: The dataframe used to create the plot (for method detection)

    Returns:
        The applicable overrides dict, or None if no overrides apply
    """
    if df is None or "afa_method" not in df.columns:
        return None

    methods_in_data = frozenset(df["afa_method"].unique().to_list())

    # Find a method set that matches the data
    for method_key, dataset_overrides in mode_overrides.items():
        # Handle tuple format: (method, frozenset)
        if isinstance(method_key, tuple) and len(method_key) == 2:
            required_method, method_set = method_key
            # Check if required_method and at least one from method_set are present
            if (
                required_method in methods_in_data
                and method_set & methods_in_data
            ):
                return dataset_overrides
        # Handle frozenset format
        elif isinstance(method_key, frozenset):
            if methods_in_data.issubset(method_key):
                return dataset_overrides

    return None


def _apply_overrides_to_figure(
    fig: matplotlib.figure.Figure,  # noqa: F821
    mode_overrides: dict[Any, Any],  # type: ignore[type-arg]
    df: pl.DataFrame | None = None,
) -> None:
    """
    Apply y-axis range overrides to a matplotlib figure's axes.

    Args:
        fig: The matplotlib Figure object
        mode_overrides: Dictionary mapping method sets (frozensets or tuples) to
            dicts of dataset -> (y_min, y_max) tuples
        df: The dataframe used to create the plot (for method detection)
    """
    # Extract facet labels (StripText) from the figure
    strip_texts = []
    for artist in fig.get_children():
        if type(artist).__name__ == "StripText":
            label = artist.get_text()
            if label:
                strip_texts.append(label)

    # Determine which method set applies
    applicable_overrides = _get_applicable_overrides(mode_overrides, df)

    # Apply overrides to axes
    if applicable_overrides:
        axes_list = fig.axes
        for i, ax in enumerate(axes_list):
            if i < len(strip_texts):
                facet_label = strip_texts[i]
                if facet_label in applicable_overrides:
                    y_min, y_max = applicable_overrides[facet_label]
                    ax.set_ylim(y_min, y_max)


# def apply_name_mappings(df: pl.DataFrame) -> pl.DataFrame:
#     """Apply display name mappings to methods and datasets."""
#     return df.with_columns(
#         afa_method=pl.col("afa_method").map_elements(
#             lambda x: METHOD_NAME_MAPPING.get(x, x), return_dtype=pl.String
#         ),
#         dataset=pl.col("dataset").map_elements(
#             lambda x: DATASET_NAME_MAPPING.get(x, x), return_dtype=pl.String
#         ),
#     )


def get_plot(
    df: pl.DataFrame,
    x_col: str,
    x_label: str,
    x_error_min: str | None = None,
    x_error_max: str | None = None,
    *,
    use_line: bool = True,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
    mode: str | None = None,
) -> p9.ggplot:
    """
    Create a plot with metrics vs cost/budget.

    Args:
        df: Input dataframe
        x_col: Column name for x-axis
        x_label: Label for x-axis
        x_error_min: Column name for x-axis error minimum
        x_error_max: Column name for x-axis error maximum
        use_line: Whether to use line plot (True) or point plot (False)
        method_order: List of method names in desired legend order
        figure_width: Figure width in inches (default: PLOT_WIDTH)
        figure_height: Figure height in inches (default: PLOT_HEIGHT)
        mode: Plot mode for y-range overrides
              ("hard_budget", "hard_budget_traj", "soft_budget_lines",
               or "soft_budget_2d_errors")
    """
    # Apply name transforms
    processed_df = df.with_columns(
        dataset=pl.col("dataset").replace(
            DATASET_NAME_MAPPING_INCLUDING_METRIC
        ),
        afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
    )

    # Get method order for legend
    if method_order is None:
        method_order = list(METHOD_NAME_MAPPING.keys())

    # Get display names in order, filtering to only methods in the data
    available_methods = processed_df["afa_method"].unique().to_list()
    ordered_display_names = [
        METHOD_NAME_MAPPING.get(orig, orig)
        for orig in method_order
        if METHOD_NAME_MAPPING.get(orig, orig) in available_methods
    ]

    # Use provided dimensions or defaults
    if figure_width is None:
        figure_width = PLOT_WIDTH
    if figure_height is None:
        figure_height = PLOT_HEIGHT

    plot = (
        ggplot(
            processed_df,
            aes(
                x=x_col,
                y="mean_metric",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + facet_wrap(
            "dataset",
            scales="free",
            ncol=4,
        )
        + labs(color="Policy", fill="Policy", x=x_label, y="Metric")
        + theme(figure_size=(figure_width, figure_height))
        + scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=ordered_display_names,
        )
        + scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=ordered_display_names,
        )
    )

    if use_line:
        plot += geom_point()
        plot += geom_ribbon(
            aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
        )
        plot += geom_line()
    else:
        plot += geom_point()
        plot += geom_errorbar(
            aes(ymin="low_metric", ymax="high_metric"), width=0
        )
        if x_error_min and x_error_max:
            plot += geom_errorbarh(
                aes(xmin=x_error_min, xmax=x_error_max),
                height=0,
            )

    # Apply y-range overrides to subplot axes after plot is created
    if mode is not None:
        plot = apply_y_range_override(plot, mode, df)

    return plot


def get_normal_hard_budget_plot(
    df: pl.DataFrame,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    return get_plot(
        df,
        x_col="eval_hard_budget",
        x_label="Hard budget",
        use_line=True,
        method_order=method_order,
        figure_width=figure_width,
        figure_height=figure_height,
        mode="hard_budget",
    )


def get_traj_hard_budget_plot(
    df: pl.DataFrame,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    return get_plot(
        df,
        x_col="n_selections_performed",
        x_label="Number of selections",
        use_line=True,
        method_order=method_order,
        figure_width=figure_width,
        figure_height=figure_height,
        mode="hard_budget_traj",
    )


def get_soft_budget_plot(
    df: pl.DataFrame,
    mode: str,
    method_order: list[str] | None = None,
    figure_width: float | None = None,
    figure_height: float | None = None,
) -> p9.ggplot:
    if mode == "2d_errors":
        return get_plot(
            df,
            x_col="mean_avg_accumulated_cost",
            x_label="Average cost accumulated per episode",
            x_error_min="low_avg_accumulated_cost",
            x_error_max="high_avg_accumulated_cost",
            use_line=False,
            method_order=method_order,
            figure_width=figure_width,
            figure_height=figure_height,
            mode="soft_budget_2d_errors",
        )
    if mode == "lines":
        return get_plot(
            df,
            x_col="mean_avg_accumulated_cost",
            x_label="Average cost accumulated per episode",
            x_error_min=None,
            x_error_max=None,
            use_line=True,
            method_order=method_order,
            figure_width=figure_width,
            figure_height=figure_height,
            mode="soft_budget_lines",
        )
    msg = f"Expected either '2d_errors' or 'lines' as mode, got '{mode}'"
    raise ValueError(msg)


def calculate_figure_dimensions(
    num_datasets: int,
    ncol: int = 4,
    subplot_height: float = 3.0,
) -> tuple[float, float]:
    """
    Calculate figure dimensions based on number of subplots.

    Args:
        num_datasets: Number of datasets (subplots)
        ncol: Number of columns in facet layout
        subplot_height: Height of each individual subplot in inches

    Returns:
        Tuple of (width, height) for the figure
    """
    num_rows = (num_datasets + ncol - 1) // ncol
    figure_width = PLOT_WIDTH
    figure_height = subplot_height * num_rows
    return figure_width, figure_height


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from merged evaluation results.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input CSV file with evaluation results.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where plots will be saved",
    )
    return parser.parse_args()


def add_metric_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        mean_metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("mean_f_score"))
        .otherwise(pl.col("mean_accuracy")),
        std_metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("std_f_score"))
        .otherwise(pl.col("std_accuracy")),
    )


def assert_only_one_soft_budget_param_type(df: pl.DataFrame) -> pl.DataFrame:
    assert (
        df["train_soft_budget_param"].is_null()
        | df["eval_soft_budget_param"].is_null()
    ).all(), (
        "Both train_soft_budget_param and eval_soft_budget_param cannot be set. Choose one."
    )
    df = df.with_columns(
        soft_budget_param=pl.coalesce(
            "train_soft_budget_param", "eval_soft_budget_param"
        )
    ).drop(["train_soft_budget_param", "eval_soft_budget_param"])
    return df


def process_df_only_stop_action(df: pl.DataFrame) -> pl.DataFrame:
    df_only_stop_action = df.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_class").is_not_null()
    )

    metric_df_only_stop_action = get_metrics_at_stop_action(
        df_only_stop_action
    )

    # Variance of metrics across seeds
    var_metric_df_only_stop_action = get_variance_of_metrics_and_cost(
        metric_df_only_stop_action
    )

    # Datasets use different metrics
    var_metric_df_only_stop_action = add_metric_column(
        var_metric_df_only_stop_action
    )

    # Add "low" and "high" versions of metrics to enable plotting of ranges
    var_metric_df_only_stop_action = (
        var_metric_df_only_stop_action.with_columns(
            low_metric=pl.col("mean_metric") - pl.col("std_metric"),
            high_metric=pl.col("mean_metric") + pl.col("std_metric"),
            low_avg_accumulated_cost=pl.col("mean_avg_accumulated_cost")
            - pl.col("std_avg_accumulated_cost"),
            high_avg_accumulated_cost=pl.col("mean_avg_accumulated_cost")
            + pl.col("std_avg_accumulated_cost"),
        )
    )

    return var_metric_df_only_stop_action


def filter_only_largest_budget(df: pl.DataFrame) -> pl.DataFrame:
    """For each dataset, only keep the largest evaluation budget."""
    return df.filter(
        pl.col("eval_hard_budget")
        == pl.col("eval_hard_budget").max().over("dataset")
    )


def process_df_every_action(df: pl.DataFrame) -> pl.DataFrame:
    # Just like in process_df_only_stop_action, filter out null predictions
    df = df.filter(pl.col("predicted_class").is_not_null())

    # When considering performance up to some budget, we only look at the case when the largest budget is used
    df = filter_only_largest_budget(df)

    metric_df_every_action = get_metrics_at_every_action(df)

    # Variance of metrics across seeds
    var_metric_df_every_action = get_variance_of_metrics(
        metric_df_every_action
    )

    # Datasets use different metrics
    var_metric_df_every_action = add_metric_column(var_metric_df_every_action)

    # Add "low" and "high" versions of metrics to enable plotting of ranges
    var_metric_df_every_action = var_metric_df_every_action.with_columns(
        low_metric=pl.col("mean_metric") - pl.col("std_metric"),
        high_metric=pl.col("mean_metric") + pl.col("std_metric"),
    )

    return var_metric_df_every_action


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    df = read_parquet(args.input)

    df = assert_only_one_soft_budget_param_type(df)
    df_stop_action = process_df_only_stop_action(df)
    df_traj = process_df_every_action(df)

    # One set of plots per dataset set
    for dataset_set_name, dataset_set in DATASET_SETS.items():
        subfolder = args.output_folder / dataset_set_name
        subfolder.mkdir(parents=True, exist_ok=True)

        df_stop_action_filtered = df_stop_action.filter(
            pl.col("dataset").is_in(dataset_set)
        )

        df_stop_action_hard_budget = df_stop_action_filtered.filter(
            pl.col("eval_hard_budget").is_null().not_()
        )
        df_stop_action_hard_budget = apply_exclusions(
            df_stop_action_hard_budget
        )
        if not df_stop_action_hard_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_stop_action_hard_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets, subplot_height=SUBPLOT_HEIGHT
            )
            normal_hard_budget_plot = get_normal_hard_budget_plot(
                df_stop_action_hard_budget,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            normal_hard_budget_plot.save(
                subfolder / "hard_budget_normal.pdf",
                width=fig_width,
                height=fig_height,
            )

        df_traj_filtered = df_traj.filter(pl.col("dataset").is_in(dataset_set))
        df_traj_hard_budget = df_traj_filtered.filter(
            pl.col("eval_hard_budget").is_null().not_()
        )
        df_traj_hard_budget = apply_exclusions(df_traj_hard_budget)
        if not df_traj_hard_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_traj_hard_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets, subplot_height=SUBPLOT_HEIGHT
            )
            traj_hard_budget_plot = get_traj_hard_budget_plot(
                df_traj_hard_budget,
                figure_width=fig_width,
                figure_height=fig_height,
            )
            traj_hard_budget_plot.save(
                subfolder / "hard_budget_traj.pdf",
                width=fig_width,
                height=fig_height,
            )

        df_stop_action_soft_budget = df_stop_action_filtered.filter(
            pl.col("soft_budget_param").is_null().not_()
        )
        df_stop_action_soft_budget = apply_exclusions(
            df_stop_action_soft_budget
        )
        if not df_stop_action_soft_budget.is_empty():
            # Calculate figure dimensions based on number of unique datasets
            num_datasets = df_stop_action_soft_budget["dataset"].n_unique()
            fig_width, fig_height = calculate_figure_dimensions(
                num_datasets, subplot_height=SUBPLOT_HEIGHT
            )
            soft_budget_plot_2d_errors = get_soft_budget_plot(
                df_stop_action_soft_budget,
                mode="2d_errors",
                figure_width=fig_width,
                figure_height=fig_height,
            )
            soft_budget_plot_lines = get_soft_budget_plot(
                df_stop_action_soft_budget,
                mode="lines",
                figure_width=fig_width,
                figure_height=fig_height,
            )
            soft_budget_plot_2d_errors.save(
                subfolder / "soft_budget_2d_errors.pdf",
                width=fig_width,
                height=fig_height,
            )
            soft_budget_plot_lines.save(
                subfolder / "soft_budget_lines.pdf",
                width=fig_width,
                height=fig_height,
            )


if __name__ == "__main__":
    main()
