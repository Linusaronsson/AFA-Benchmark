from __future__ import annotations

import argparse
from pathlib import Path

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
)
from sklearn.metrics import accuracy_score, f1_score

# Datasets not listed here will use accuracy
DATASETS_WITH_F_SCORE = [
    "cube_without_noise",
    "synthetic_mnist_without_noise",
    "cube",
    "synthetic_mnist",
]

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

DATASET_NAME_MAPPING = {
    "cube": "Cube",
    "afa_context": "AFAContext",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "Cube (noiseless)",
    "afa_context_without_noise": "AFAContext (noiseless)",
    "synthetic_mnist_without_noise": "Synthetic MNIST (noiseless)",
    "MNIST": "MNIST",
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
            "action_performed": pl.Int64,
            "true_class": pl.Int64,
            "accumulated_cost": pl.Float64,
            "idx": pl.Int64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.Int64,
            "eval_hard_budget": pl.Float64,
            "soft_budget_param": pl.Float64,
            "selections_performed": pl.Int64,
            "afa_method": pl.String,
            "dataset": pl.String,
            "train_seed": pl.Int64,
            "train_hard_budget": pl.Float64,
            "predicted_class": pl.Int64,
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


def read_csv(input_csv_path: Path) -> pl.DataFrame:
    return pl.read_csv(
        input_csv_path,
        schema_overrides={
            "action_performed": pl.Int64,
            "true_class": pl.Int64,
            "accumulated_cost": pl.Float64,
            "idx": pl.Int64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.Int64,
            "eval_hard_budget": pl.Float64,
            "train_soft_budget_param": pl.Float64,
            "eval_soft_budget_param": pl.Float64,
            "selections_performed": pl.Int64,
            "afa_method": pl.String,
            "dataset": pl.String,
            "train_seed": pl.Int64,
            "train_hard_budget": pl.Float64,
            "predicted_class": pl.Int64,
        },
        null_values=["null"],
    )


def get_variance_of_metrics(df: pl.DataFrame) -> pl.DataFrame:
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


def apply_name_mappings(df: pl.DataFrame) -> pl.DataFrame:
    """Apply display name mappings to methods and datasets."""
    return df.with_columns(
        afa_method=pl.col("afa_method").map_elements(
            lambda x: METHOD_NAME_MAPPING.get(x, x), return_dtype=pl.String
        ),
        dataset=pl.col("dataset").map_elements(
            lambda x: DATASET_NAME_MAPPING.get(x, x), return_dtype=pl.String
        ),
    )


def get_plot(
    df: pl.DataFrame,
    x_col: str,
    x_label: str,
    x_error_min: str | None = None,
    x_error_max: str | None = None,
    *,
    use_line: bool = True,
) -> p9.ggplot:
    """Create a plot with metrics vs cost/budget."""
    plot = (
        ggplot(
            df,
            aes(
                x=x_col,
                y="mean_metric",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + facet_wrap("dataset", scales="free", ncol=4)
        + labs(color="Method", fill="Method", x=x_label, y="Metric")
        + p9.theme(figure_size=(10, 8))
    )

    if use_line:
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

    return plot


def get_hard_budget_plot(df: pl.DataFrame) -> p9.ggplot:
    return get_plot(
        df,
        x_col="eval_hard_budget",
        x_label="Hard budget",
        use_line=True,
    )


def get_soft_budget_plot(df: pl.DataFrame) -> p9.ggplot:
    return get_plot(
        df,
        x_col="mean_avg_accumulated_cost",
        x_label="Cost per episode",
        x_error_min="low_avg_accumulated_cost",
        x_error_max="high_avg_accumulated_cost",
        use_line=False,
    )


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


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    df = read_csv(args.input)

    # Either train_soft_budget_param is set, or eval_soft_budget_param is set, but not both. This means that one of them must always be null
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
    )

    # Only consider performance at stop action
    df = df.filter(pl.col("action_performed") == 0)

    metric_df = get_metrics_at_stop_action(df)

    # Variance of metrics across seeds
    var_metric_df = get_variance_of_metrics(metric_df)

    # Datasets use different metrics
    var_metric_df = add_metric_column(var_metric_df)

    # Add "low" and "high" versions of metrics to enable plotting of ranges
    var_metric_df = var_metric_df.with_columns(
        low_metric=pl.col("mean_metric") - pl.col("std_metric"),
        high_metric=pl.col("mean_metric") + pl.col("std_metric"),
        low_avg_accumulated_cost=pl.col("mean_avg_accumulated_cost")
        - pl.col("std_avg_accumulated_cost"),
        high_avg_accumulated_cost=pl.col("mean_avg_accumulated_cost")
        + pl.col("std_avg_accumulated_cost"),
    )

    var_metric_df = apply_name_mappings(var_metric_df)

    df_hard_budget = var_metric_df.filter(
        pl.col("eval_hard_budget").is_null().not_()
    )
    hard_budget_plot = get_hard_budget_plot(df_hard_budget)
    hard_budget_plot.save(args.output_folder / "hard_budget.pdf")

    df_soft_budget = var_metric_df.filter(
        pl.col("soft_budget_param").is_null().not_()
    )
    soft_budget_plot = get_soft_budget_plot(df_soft_budget)
    soft_budget_plot.save(args.output_folder / "soft_budget.pdf")


if __name__ == "__main__":
    main()
