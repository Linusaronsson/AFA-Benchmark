from __future__ import annotations

import argparse
import re
from pathlib import Path

import plotnine as p9
import polars as pl
from sklearn.metrics import accuracy_score, f1_score

from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    DATASET_NAME_MAPPING,
    DATASETS_WITH_F_SCORE,
    METHOD_NAME_MAPPING,
)

SUBPLOT_HEIGHT = 2.8

INITIALIZER_NAME_MAPPING = {
    "missingness": "Missingness",
    "missingness_all_observed": "All observed",
    "cube_nm_ar": "CUBE-NM-AR",
    "cold": "Cold start",
}

VARIANT_NAME_MAPPING = {
    "marf": "MARF",
    "madt": "MADT",
    "magbt": "MAGBT",
    "malasso": "MALasso",
}
TRAIN_INITIALIZER_NAME_MAPPING = {
    "cold": "Full train support",
    "cube_nm_ar": "Rescue-censored train",
}
MECHANISM_INITIALIZER_MAPPING = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar_logistic": "MNAR logistic",
}


def _format_rate_initializer_label(initializer: str) -> str | None:
    match = re.fullmatch(r"(mcar|mar|mnar_logistic)_p(\d+)", initializer)
    if match is None:
        return None

    mechanism, rate_code = match.groups()
    mechanism_label = MECHANISM_INITIALIZER_MAPPING.get(
        mechanism, mechanism.upper()
    )
    # Initializer suffixes use compact tenths notation:
    # p01 -> 0.1, p03 -> 0.3, p05 -> 0.5, p07 -> 0.7.
    rate = float(rate_code) / (10 ** max(len(rate_code) - 1, 0))
    return f"{mechanism_label} {round(rate * 100):.0f}%"


def _format_initializer_label(initializer: str) -> str:
    composite_match = re.fullmatch(
        r"train_initializer-(.+)\+eval_initializer-(.+)",
        initializer,
    )
    formatted_label: str
    if composite_match is not None:
        train_initializer, eval_initializer = composite_match.groups()
        train_label = TRAIN_INITIALIZER_NAME_MAPPING.get(
            train_initializer,
            _format_initializer_label(train_initializer),
        )
        if eval_initializer == "cold":
            formatted_label = train_label
        else:
            eval_label = _format_initializer_label(eval_initializer)
            formatted_label = f"{train_label} / eval {eval_label}"
    else:
        rate_label = _format_rate_initializer_label(initializer)
        if rate_label is not None:
            formatted_label = rate_label
        elif initializer in INITIALIZER_NAME_MAPPING:
            formatted_label = INITIALIZER_NAME_MAPPING[initializer]
        elif initializer.startswith("missingness_all_observed_"):
            variant = initializer.removeprefix("missingness_all_observed_")
            variant_label = VARIANT_NAME_MAPPING.get(variant, variant.upper())
            formatted_label = f"{variant_label} all observed"
        elif initializer.startswith("missingness_"):
            variant = initializer.removeprefix("missingness_")
            variant_label = VARIANT_NAME_MAPPING.get(variant, variant.upper())
            formatted_label = f"{variant_label} missingness"
        else:
            formatted_label = initializer

    return formatted_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation performance grouped by a comparison column, "
            "using already merged eval parquet from the workflow."
        )
    )
    parser.add_argument("input_path", type=Path, help="Input parquet path.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--group-column",
        type=str,
        default="initializer",
        help="Column used for the comparison axis.",
    )
    parser.add_argument(
        "--group-label",
        type=str,
        default="Initializer",
        help="Axis and legend label for the comparison column.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="initializer_comparison",
        help="Stem used for output filenames.",
    )
    parser.add_argument(
        "--budget-mode",
        type=str,
        default="hard",
        choices=["hard", "soft", "all"],
        help="Which budget regime to include.",
    )
    parser.add_argument(
        "--keep-largest-hard-budget",
        action="store_true",
        help=(
            "When budget-mode=hard, keep only the largest hard budget per "
            "dataset."
        ),
    )
    parser.add_argument(
        "--eval-hard-budget",
        type=float,
        default=None,
        help=(
            "Optional hard budget to keep. When set, this takes precedence "
            "over --keep-largest-hard-budget."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of methods to keep.",
    )
    return parser.parse_args()


def _format_group_value(value: str, group_column: str) -> str:
    if group_column == "initializer":
        return _format_initializer_label(value)
    return value


def _compute_metrics_per_run(
    data: pl.DataFrame,
    *,
    group_column: str,
) -> pl.DataFrame:
    group_cols = [
        "afa_method",
        "dataset",
        group_column,
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "train_hard_budget",
        "train_soft_budget_param",
        "eval_soft_budget_param",
    ]

    return data.group_by(*group_cols).map_groups(
        lambda group_df: pl.DataFrame(
            {
                "afa_method": [group_df["afa_method"].first()],
                "dataset": [group_df["dataset"].first()],
                group_column: [group_df[group_column].first()],
                "train_seed": [group_df["train_seed"].first()],
                "eval_seed": [group_df["eval_seed"].first()],
                "eval_hard_budget": [group_df["eval_hard_budget"].first()],
                "train_hard_budget": [group_df["train_hard_budget"].first()],
                "train_soft_budget_param": [
                    group_df["train_soft_budget_param"].first()
                ],
                "eval_soft_budget_param": [
                    group_df["eval_soft_budget_param"].first()
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
                group_column: pl.String,
                "train_seed": pl.UInt64,
                "eval_seed": pl.UInt64,
                "eval_hard_budget": pl.Float64,
                "train_hard_budget": pl.Float64,
                "train_soft_budget_param": pl.Float64,
                "eval_soft_budget_param": pl.Float64,
                "accuracy": pl.Float64,
                "f_score": pl.Float64,
            },
        )
    )


def _aggregate_across_seeds(
    data: pl.DataFrame,
    *,
    group_column: str,
) -> pl.DataFrame:
    data = data.with_columns(
        metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("f_score"))
        .otherwise(pl.col("accuracy"))
    )
    return (
        data.group_by("dataset", "afa_method", group_column)
        .agg(
            mean_metric=pl.col("metric").mean(),
            std_metric=pl.col("metric").std(),
            n_runs=pl.len(),
        )
        .with_columns(
            std_metric=pl.col("std_metric").fill_null(0.0),
            low_metric=pl.col("mean_metric")
            - pl.col("std_metric").fill_null(0.0),
            high_metric=pl.col("mean_metric")
            + pl.col("std_metric").fill_null(0.0),
        )
    )


def _filter_budget_mode(
    data: pl.DataFrame,
    mode: str,
    keep_largest_hard_budget: bool,
    eval_hard_budget: float | None = None,
) -> pl.DataFrame:
    if mode == "hard":
        data = data.filter(pl.col("eval_hard_budget").is_not_null())
        if eval_hard_budget is not None:
            data = data.filter(pl.col("eval_hard_budget") == eval_hard_budget)
        elif keep_largest_hard_budget:
            data = data.filter(
                pl.col("eval_hard_budget")
                == pl.col("eval_hard_budget").max().over("dataset")
            )
        return data
    if mode == "soft":
        return data.filter(
            pl.col("train_soft_budget_param").is_not_null()
            | pl.col("eval_soft_budget_param").is_not_null()
        )
    return data


def _make_plot(
    summary_data: pl.DataFrame,
    *,
    group_column: str,
    group_label: str,
) -> p9.ggplot:
    plot_data = summary_data.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING),
        afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
        **{
            group_column: pl.col(group_column).map_elements(
                lambda value: _format_group_value(value, group_column),
                return_dtype=pl.String,
            )
        },
    )
    n_datasets = max(plot_data["dataset"].n_unique(), 1)
    n_rows = (n_datasets + 1) // 2
    figure_height = SUBPLOT_HEIGHT * n_rows

    return (
        p9.ggplot(
            plot_data,
            p9.aes(
                x=group_column,
                y="mean_metric",
                color=group_column,
                shape="afa_method",
            ),
        )
        + p9.geom_point(size=2.8)
        + p9.geom_errorbar(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            width=0.08,
        )
        + p9.facet_wrap("dataset", scales="free_y", ncol=2)
        + p9.scale_color_brewer(type="qual", palette=COLOR_PALETTE_NAME)
        + p9.labs(
            x=group_label,
            y="Metric",
            color=group_label,
            shape="Policy",
        )
        + p9.theme(figure_size=(11.0, figure_height))
        # + p9.theme(
        #     axis_text_x=p9.element_blank(),
        #     axis_ticks_major_x=p9.element_blank(),
        #     axis_title_x=p9.element_blank(),  # optional
        # )
        + p9.theme(axis_text_x=p9.element_text(rotation=45, ha="right"))
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pl.read_parquet(args.input_path)
    required_cols = {
        "action_performed",
        "predicted_class",
        "true_class",
        "dataset",
        "afa_method",
        args.group_column,
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "train_hard_budget",
        "train_soft_budget_param",
        "eval_soft_budget_param",
    }
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        msg = f"Input parquet missing required columns: {sorted(missing_cols)}"
        raise ValueError(msg)

    if args.methods:
        data = data.filter(pl.col("afa_method").is_in(args.methods))

    data = data.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_class").is_not_null()
    )
    available_hard_budgets = None
    if args.budget_mode in {"hard", "all"}:
        available_hard_budgets = (
            data.filter(pl.col("eval_hard_budget").is_not_null())
            .group_by("dataset")
            .agg(
                budgets=pl.col("eval_hard_budget")
                .unique()
                .sort()
                .cast(pl.List(pl.Float64))
            )
            .sort("dataset")
        )
    data = _filter_budget_mode(
        data,
        args.budget_mode,
        args.keep_largest_hard_budget,
        args.eval_hard_budget,
    )
    if data.is_empty():
        msg = "No rows left after filtering; cannot generate plot."
        if (
            args.budget_mode == "hard"
            and args.eval_hard_budget is not None
            and available_hard_budgets is not None
            and not available_hard_budgets.is_empty()
        ):
            budget_summary = ", ".join(
                f"{row['dataset']}={row['budgets']}"
                for row in available_hard_budgets.to_dicts()
            )
            msg = (
                "No rows left after filtering for "
                f"eval_hard_budget={args.eval_hard_budget}. "
                f"Available hard budgets by dataset: {budget_summary}."
            )
        raise ValueError(msg)

    metrics_per_run = _compute_metrics_per_run(
        data,
        group_column=args.group_column,
    )
    summary_data = _aggregate_across_seeds(
        metrics_per_run,
        group_column=args.group_column,
    )
    if summary_data.is_empty():
        msg = "No grouped rows produced; cannot generate plot."
        raise ValueError(msg)

    plot = _make_plot(
        summary_data,
        group_column=args.group_column,
        group_label=args.group_label,
    )
    plot_height = max(
        6.0,
        SUBPLOT_HEIGHT * ((summary_data["dataset"].n_unique() + 1) // 2),
    )
    for suffix in ("pdf", "svg", "png"):
        plot.save(
            args.output_dir / f"{args.output_stem}.{suffix}",
            width=15.0,
            height=plot_height,
        )
    summary_data.write_csv(args.output_dir / f"{args.output_stem}.csv")


if __name__ == "__main__":
    main()
