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
    PLOT_WIDTH,
)

SUBPLOT_HEIGHT = 2.8

MECHANISM_LABELS = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar_logistic": "MNAR logistic",
}


def _parse_mechanism_and_rate(
    initializer: str,
) -> tuple[str, float] | None:
    # Handles both raw (mcar_p03) and composite
    # (train_initializer-mcar_p03+eval_initializer-cold) forms.
    composite = re.fullmatch(
        r"train_initializer-(.+)\+eval_initializer-(.+)",
        initializer,
    )
    train_init = composite.group(1) if composite else initializer

    match = re.fullmatch(r"(mcar|mar|mnar_logistic)_p(\d+)", train_init)
    if match is None:
        return None

    mechanism, rate_code = match.groups()
    rate = float(rate_code) / (10 ** max(len(rate_code) - 1, 0))
    return mechanism, rate


def _is_cold_start(initializer: str) -> bool:
    composite = re.fullmatch(
        r"train_initializer-(.+)\+eval_initializer-(.+)",
        initializer,
    )
    train_init = composite.group(1) if composite else initializer
    return train_init == "cold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation performance grouped by a comparison "
            "column, using already merged eval parquet from the "
            "workflow."
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
            "When budget-mode=hard, keep only the largest hard "
            "budget per dataset."
        ),
    )
    parser.add_argument(
        "--eval-hard-budget",
        type=float,
        default=None,
        help=(
            "Optional hard budget to keep. When set, this takes "
            "precedence over --keep-largest-hard-budget."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of methods to keep.",
    )
    parser.add_argument(
        "--save-individual-subplots",
        action="store_true",
        help="Also save one figure per dataset per mechanism.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Output formats (default: pdf svg png).",
    )
    return parser.parse_args()


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
                        group_df["true_class"],
                        group_df["predicted_class"],
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
            sem_metric=pl.col("std_metric").fill_null(0.0)
            / pl.col("n_runs").clip(1).sqrt(),
        )
        .with_columns(
            low_metric=pl.col("mean_metric") - pl.col("sem_metric"),
            high_metric=pl.col("mean_metric") + pl.col("sem_metric"),
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


def _enrich_with_mechanism_and_rate(
    summary: pl.DataFrame,
    group_column: str,
) -> pl.DataFrame:
    # Cold-start rows get mechanism="cold" and rate=0.0.
    mechanisms: list[str | None] = []
    rates: list[float | None] = []
    for value in summary[group_column].to_list():
        if _is_cold_start(value):
            mechanisms.append("cold")
            rates.append(0.0)
        else:
            parsed = _parse_mechanism_and_rate(value)
            if parsed is not None:
                mechanisms.append(parsed[0])
                rates.append(parsed[1])
            else:
                mechanisms.append(None)
                rates.append(None)
    return summary.with_columns(
        mechanism=pl.Series(mechanisms, dtype=pl.String),
        rate=pl.Series(rates, dtype=pl.Float64),
    )


DATASET_NAME_MAPPING_WITH_METRIC = {
    ds: (f"{name} ({'F1' if ds in DATASETS_WITH_F_SCORE else 'Acc.'})")
    for ds, name in DATASET_NAME_MAPPING.items()
}


def _make_mechanism_plot(
    mechanism_data: pl.DataFrame,
    baseline_data: pl.DataFrame,
    *,
    mechanism_label: str,
) -> p9.ggplot:
    # mechanism_data: rows with varying rate.
    # baseline_data: cold-start reference (horizontal dashed lines).
    plot_df = mechanism_data.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING_WITH_METRIC),
        afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
    )

    method_order = list(METHOD_NAME_MAPPING.keys())
    available = plot_df["afa_method"].unique().to_list()
    ordered = [
        METHOD_NAME_MAPPING.get(m, m)
        for m in method_order
        if METHOD_NAME_MAPPING.get(m, m) in available
    ]

    n_datasets = max(plot_df["dataset"].n_unique(), 1)
    n_rows = (n_datasets + 1) // 2
    figure_height = max(6.0, SUBPLOT_HEIGHT * n_rows)
    figure_width = min(PLOT_WIDTH, 11.0)

    plot = (
        p9.ggplot(
            plot_df,
            p9.aes(
                x="rate",
                y="mean_metric",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=2.5)
        + p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
        )
        + p9.facet_wrap("dataset", scales="free_y", ncol=2)
        + p9.scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=ordered,
        )
        + p9.scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=ordered,
        )
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:.0%}" for x in xs],
        )
        + p9.labs(
            x=f"Training missingness rate ({mechanism_label})",
            y="Metric",
            color="Policy",
            fill="Policy",
        )
        + p9.theme(figure_size=(figure_width, figure_height))
    )

    # Cold-start baselines as horizontal dashed lines.
    if not baseline_data.is_empty():
        bl = baseline_data.with_columns(
            dataset=pl.col("dataset").replace(
                DATASET_NAME_MAPPING_WITH_METRIC
            ),
            afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
        )
        plot += p9.geom_hline(
            data=bl,
            mapping=p9.aes(
                yintercept="mean_metric",
                color="afa_method",
            ),
            linetype="dashed",
            alpha=0.5,
            size=0.6,
        )

    return plot


def _save_plot(
    plot: p9.ggplot,
    output_path: Path,
    *,
    width: float,
    height: float,
    formats: list[str],
) -> None:
    for fmt in formats:
        plot.save(
            output_path.with_suffix(f".{fmt}"),
            width=width,
            height=height,
        )


def _load_and_prepare(
    args: argparse.Namespace,
) -> pl.DataFrame:
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
        msg = "No rows left after filtering."
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
                f"Available: {budget_summary}."
            )
        raise ValueError(msg)
    return data


def _save_individual_subplots(
    mech_data: pl.DataFrame,
    baseline: pl.DataFrame,
    *,
    mech_label: str,
    subplot_dir: Path,
    figure_width: float,
    formats: list[str],
) -> None:
    subplot_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(mech_data["dataset"].unique().to_list()):
        ds_data = mech_data.filter(pl.col("dataset") == dataset)
        ds_baseline = baseline.filter(pl.col("dataset") == dataset)
        ds_plot = _make_mechanism_plot(
            ds_data,
            ds_baseline,
            mechanism_label=mech_label,
        )
        display_name = DATASET_NAME_MAPPING.get(dataset, dataset)
        ds_stem = (display_name or dataset).replace(" ", "_").lower()
        _save_plot(
            ds_plot,
            subplot_dir / ds_stem,
            width=figure_width * 0.55,
            height=SUBPLOT_HEIGHT * 1.4,
            formats=formats,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_and_prepare(args)

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

    summary_data.write_csv(args.output_dir / f"{args.output_stem}.csv")

    enriched = _enrich_with_mechanism_and_rate(summary_data, args.group_column)

    baseline = enriched.filter(pl.col("mechanism") == "cold")
    mechanism_rows = enriched.filter(
        (pl.col("mechanism") != "cold") & pl.col("mechanism").is_not_null()
    )

    if mechanism_rows.is_empty():
        print("No mechanism-specific rows found; nothing to plot.")
        return

    mechanisms = sorted(mechanism_rows["mechanism"].unique().to_list())
    figure_width = min(PLOT_WIDTH, 11.0)

    for mechanism in mechanisms:
        assert isinstance(mechanism, str)
        mech_label: str = MECHANISM_LABELS.get(mechanism, mechanism.upper())
        mech_data = mechanism_rows.filter(pl.col("mechanism") == mechanism)
        if mech_data.is_empty():
            continue

        n_datasets = max(mech_data["dataset"].n_unique(), 1)
        n_rows = (n_datasets + 1) // 2
        figure_height = max(6.0, SUBPLOT_HEIGHT * n_rows)

        plot = _make_mechanism_plot(
            mech_data,
            baseline,
            mechanism_label=mech_label,
        )
        stem = f"{args.output_stem}_{mechanism}"
        _save_plot(
            plot,
            args.output_dir / stem,
            width=figure_width,
            height=figure_height,
            formats=args.formats,
        )

        if args.save_individual_subplots:
            _save_individual_subplots(
                mech_data,
                baseline,
                mech_label=mech_label,
                subplot_dir=args.output_dir / f"{stem}_subplots",
                figure_width=figure_width,
                formats=args.formats,
            )


if __name__ == "__main__":
    main()
