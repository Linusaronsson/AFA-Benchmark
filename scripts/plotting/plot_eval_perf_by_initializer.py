from __future__ import annotations

import argparse
import re
from pathlib import Path

import plotnine as p9
import polars as pl

from afabench.common.naming import LEGACY_DATASET_KEY_ALIASES
from afabench.common.parquet import (
    collect_streaming,
    require_columns,
    scan_parquet,
)
from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    DATASET_NAME_MAPPING,
    DATASETS_WITH_F_SCORE,
    METHOD_NAME_MAPPING,
    PLOT_WIDTH,
)

SUBPLOT_HEIGHT = 2.8
DATASET_FACET_COLS = 4

MECHANISM_LABELS = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar": "MNAR",
    "mnar_logistic": "MNAR logistic",
}

PUBLICATION_METHOD_SPECS: dict[str, dict[str, str | None]] = {
    "gadgil2023": {
        "baseline_label": "DIME baseline",
        "curve_label": "DIME + block-only",
    },
    "gadgil2023_ipw_feature_marginal": {
        "baseline_label": None,
        "curve_label": "DIME + IPW",
    },
    "aaco_full": {
        "baseline_label": "AACO baseline",
        "curve_label": None,
    },
    "aaco_zero_fill": {
        "baseline_label": None,
        "curve_label": "AACO + zero-fill",
    },
    "aaco_mask_aware": {
        "baseline_label": None,
        "curve_label": "AACO + mask-aware",
    },
    "aaco_dr": {
        "baseline_label": None,
        "curve_label": "AACO + DR",
    },
    "ol_with_mask": {
        "baseline_label": "OL-MFRL baseline",
        "curve_label": "OL-MFRL + block-only",
    },
}

CURVE_LABEL_ORDER = [
    "DIME + block-only",
    "DIME + IPW",
    "AACO + zero-fill",
    "AACO + mask-aware",
    "AACO + DR",
    "OL-MFRL + block-only",
]

BASELINE_LABEL_ORDER = [
    "DIME baseline",
    "AACO baseline",
    "OL-MFRL baseline",
]

BASELINE_LINETYPES = {
    "DIME baseline": "dotted",
    "AACO baseline": "dashed",
    "OL-MFRL baseline": "dashdot",
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

    match = re.fullmatch(
        r"(?:xor_)?(mcar|mar|mnar|mnar_logistic)_p(\d+)",
        train_init,
    )
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


def _collect_lazy(frame: pl.LazyFrame) -> pl.DataFrame:
    return collect_streaming(frame)


def _publication_method_spec(
    afa_method: str,
) -> dict[str, str | None]:
    default_label = METHOD_NAME_MAPPING.get(afa_method, afa_method)
    return PUBLICATION_METHOD_SPECS.get(
        afa_method,
        {
            "baseline_label": None,
            "curve_label": default_label,
        },
    )


def _compute_metrics_per_run(
    data: pl.LazyFrame,
    *,
    group_column: str,
) -> pl.LazyFrame:
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

    confusion = data.group_by(
        [*group_cols, "true_class", "predicted_class"]
    ).agg(n=pl.len())

    accuracy = (
        confusion.group_by(group_cols)
        .agg(
            total=pl.col("n").sum(),
            correct=pl.when(pl.col("true_class") == pl.col("predicted_class"))
            .then(pl.col("n"))
            .otherwise(0)
            .sum(),
        )
        .with_columns(
            accuracy=pl.col("correct").cast(pl.Float64)
            / pl.col("total").clip(lower_bound=1).cast(pl.Float64)
        )
        .select([*group_cols, "accuracy"])
    )

    true_totals = (
        confusion.group_by([*group_cols, "true_class"])
        .agg(true_total=pl.col("n").sum().cast(pl.UInt64))
        .rename({"true_class": "label"})
        .with_columns(pred_total=pl.lit(0, dtype=pl.UInt64))
        .select([*group_cols, "label", "true_total", "pred_total"])
    )
    pred_totals = (
        confusion.group_by([*group_cols, "predicted_class"])
        .agg(pred_total=pl.col("n").sum().cast(pl.UInt64))
        .rename({"predicted_class": "label"})
        .with_columns(true_total=pl.lit(0, dtype=pl.UInt64))
        .select([*group_cols, "label", "true_total", "pred_total"])
    )
    label_totals = (
        pl.concat([true_totals, pred_totals])
        .group_by([*group_cols, "label"])
        .agg(
            true_total=pl.col("true_total").sum(),
            pred_total=pl.col("pred_total").sum(),
        )
    )
    tp_by_label = (
        confusion.filter(pl.col("true_class") == pl.col("predicted_class"))
        .select([*group_cols, pl.col("true_class").alias("label"), "n"])
        .rename({"n": "tp"})
        .with_columns(tp=pl.col("tp").cast(pl.UInt64))
    )

    f_score = (
        label_totals.join(
            tp_by_label,
            on=[*group_cols, "label"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(
            tp=pl.col("tp").fill_null(0),
            denom=(pl.col("true_total") + pl.col("pred_total")).cast(
                pl.Float64
            ),
        )
        .with_columns(
            f1_label=pl.when(pl.col("denom") > 0)
            .then(2.0 * pl.col("tp").cast(pl.Float64) / pl.col("denom"))
            .otherwise(0.0)
        )
        .group_by(group_cols)
        .agg(f_score=pl.col("f1_label").mean())
    )

    return accuracy.join(
        f_score,
        on=group_cols,
        how="left",
        nulls_equal=True,
    )


def _aggregate_across_seeds(
    data: pl.LazyFrame,
    *,
    group_column: str,
) -> pl.LazyFrame:
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
    data: pl.LazyFrame,
    mode: str,
    keep_largest_hard_budget: bool,
    eval_hard_budget: float | None = None,
) -> pl.LazyFrame:
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


def _enrich_with_publication_roles(
    summary: pl.DataFrame,
) -> pl.DataFrame:
    baseline_labels: list[str | None] = []
    curve_labels: list[str | None] = []
    for afa_method in summary["afa_method"].to_list():
        spec = _publication_method_spec(afa_method)
        baseline_labels.append(spec["baseline_label"])
        curve_labels.append(spec["curve_label"])
    return summary.with_columns(
        baseline_label=pl.Series(baseline_labels, dtype=pl.String),
        curve_label=pl.Series(curve_labels, dtype=pl.String),
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
    # mechanism_data: mitigation rows with varying training missingness rate.
    # baseline_data: full-support reference methods shown as hlines.
    plot_df = mechanism_data.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING_WITH_METRIC),
    )

    available_curve_labels = plot_df["curve_label"].unique().to_list()
    curve_breaks = [
        label for label in CURVE_LABEL_ORDER if label in available_curve_labels
    ]

    n_datasets = max(plot_df["dataset"].n_unique(), 1)
    n_rows = (n_datasets + DATASET_FACET_COLS - 1) // DATASET_FACET_COLS
    figure_height = max(6.0, SUBPLOT_HEIGHT * n_rows)
    figure_width = PLOT_WIDTH

    plot = (
        p9.ggplot(
            plot_df,
            p9.aes(
                x="rate",
                y="mean_metric",
                color="curve_label",
                fill="curve_label",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=2.5)
        + p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
            show_legend=False,
        )
        + p9.facet_wrap(
            "dataset",
            scales="free_y",
            ncol=DATASET_FACET_COLS,
        )
        + p9.scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=curve_breaks,
        )
        + p9.scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=curve_breaks,
        )
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:.0%}" for x in xs],
        )
        + p9.labs(
            x=f"Training missingness rate ({mechanism_label})",
            y="Metric",
            color="Mitigations",
            fill="Mitigations",
            linetype="Baselines",
        )
        + p9.guides(
            fill="none",
            color=p9.guide_legend(order=1),
            linetype=p9.guide_legend(order=2),
        )
        + p9.theme(figure_size=(figure_width, figure_height))
    )

    if not baseline_data.is_empty():
        bl = baseline_data.with_columns(
            dataset=pl.col("dataset").replace(
                DATASET_NAME_MAPPING_WITH_METRIC
            ),
        )
        available_baseline_labels = bl["baseline_label"].unique().to_list()
        baseline_breaks = [
            label
            for label in BASELINE_LABEL_ORDER
            if label in available_baseline_labels
        ]
        plot += p9.geom_hline(
            data=bl,
            mapping=p9.aes(
                yintercept="mean_metric",
                linetype="baseline_label",
            ),
            color="black",
            alpha=0.9,
            size=0.8,
        )
        plot += p9.scale_linetype_manual(
            values=BASELINE_LINETYPES,
            breaks=baseline_breaks,
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
) -> pl.LazyFrame:
    data = scan_parquet(args.input_path)
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
    require_columns(data, required_cols)

    data = data.select(sorted(required_cols))
    data = data.with_columns(
        dataset=pl.col("dataset").replace(LEGACY_DATASET_KEY_ALIASES)
    )

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
        available_hard_budgets = _collect_lazy(available_hard_budgets)
    data = _filter_budget_mode(
        data,
        args.budget_mode,
        args.keep_largest_hard_budget,
        args.eval_hard_budget,
    )
    remaining_rows = _collect_lazy(data.select(pl.len())).item()
    if remaining_rows == 0:
        msg = "No rows left after filtering."
        if (
            args.budget_mode == "hard"
            and args.eval_hard_budget is not None
            and available_hard_budgets is not None
            and available_hard_budgets.height > 0
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
    summary_data = _collect_lazy(
        _aggregate_across_seeds(
            metrics_per_run,
            group_column=args.group_column,
        )
    )
    if summary_data.height == 0:
        msg = "No grouped rows produced; cannot generate plot."
        raise ValueError(msg)

    enriched = _enrich_with_mechanism_and_rate(summary_data, args.group_column)
    enriched = _enrich_with_publication_roles(enriched)
    enriched.write_csv(args.output_dir / f"{args.output_stem}.csv")

    baseline = enriched.filter(
        (pl.col("mechanism") == "cold")
        & pl.col("baseline_label").is_not_null()
    )
    mechanism_rows = enriched.filter(
        (pl.col("mechanism") != "cold")
        & pl.col("mechanism").is_not_null()
        & pl.col("curve_label").is_not_null()
    )

    if mechanism_rows.is_empty():
        print("No mechanism-specific rows found; nothing to plot.")
        return

    mechanisms = sorted(mechanism_rows["mechanism"].unique().to_list())
    figure_width = PLOT_WIDTH

    for mechanism in mechanisms:
        assert isinstance(mechanism, str)
        mech_label: str = MECHANISM_LABELS.get(mechanism, mechanism.upper())
        mech_data = mechanism_rows.filter(pl.col("mechanism") == mechanism)
        if mech_data.is_empty():
            continue

        n_datasets = max(mech_data["dataset"].n_unique(), 1)
        n_rows = (n_datasets + DATASET_FACET_COLS - 1) // DATASET_FACET_COLS
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
