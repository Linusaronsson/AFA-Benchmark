# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Plot stop-shield effectiveness: accuracy and safety vs delta.

Reads aggregate_summary.csv files produced by plot_cube_nm_ar_results.py
under `stop_shield-*` directories and generates a two-panel plot with
`delta=0` used for the unshielded reference.

Usage:
    python scripts/analysis/plot_stop_shield_effectiveness.py \
        extra/output/plot_results/cube_nm_ar/eval_split-val/\
train_initializer-comparison+eval_initializer-cold/budget_mode-soft \
        extra/output/analysis/stop_shield
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D

from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    METHOD_NAME_MAPPING,
    PLOT_HEIGHT,
    PLOT_WIDTH,
)

DEFAULT_FORMATS = ("pdf", "svg")

REPRESENTATIVE_SOFT_BUDGETS: dict[str, float] = {
    "aaco_full": 1.2,
    "aaco_zero_fill": 1.2,
    "aaco_mask_aware": 1.2,
    "aaco_dr": 1.2,
    "gadgil2023": 0.01,
    "gadgil2023_ipw_feature_marginal": 0.01,
    "ol_with_mask": 0.01,
}

TRAIN_INITIALIZER_LABELS: dict[str, str] = {
    "cold": "Full train support",
    "cube_nm_ar": "Rescue-censored train",
    "cube_nm_ar_mar": "MAR control",
    "mcar_p03": "MCAR 30%",
    "mar_p03": "MAR 30%",
    "mnar_logistic_p03": "MNAR logistic 30%",
}

TRAIN_INITIALIZER_LINESTYLES: dict[str, str | tuple[int, tuple[int, ...]]] = {
    "cold": "-",
    "cube_nm_ar": "--",
    "cube_nm_ar_mar": "-.",
    "mcar_p03": ":",
    "mar_p03": (0, (3, 1, 1, 1)),
    "mnar_logistic_p03": (0, (5, 1)),
}


def _palette_colors() -> tuple[str, ...]:
    cmap = plt.get_cmap(COLOR_PALETTE_NAME)
    raw = getattr(cmap, "colors", None)
    if raw is None:
        raw = [cmap(x) for x in np.linspace(0.0, 1.0, 8)]
    return tuple(to_hex(color) for color in raw)


def _apply_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--train-initializers", nargs="+", default=None)
    parser.add_argument("--risk-budget", type=float, default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
    )
    return parser.parse_args()


def _load_summaries(input_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    unshielded_path = input_root / "stop_shield-none" / "aggregate_summary.csv"
    if unshielded_path.exists():
        summary_df = pd.read_csv(unshielded_path)
        summary_df["stop_shield_delta"] = 0.0
        rows.append(summary_df)

    pattern = re.compile(r"stop_shield-([^/]+)")
    for path in sorted(input_root.glob("stop_shield-*/aggregate_summary.csv")):
        match = pattern.search(str(path.parent))
        if match is None or match.group(1) == "none":
            continue
        stop_shield_delta = float(match.group(1))
        summary_df = pd.read_csv(path)
        summary_df["stop_shield_delta"] = stop_shield_delta
        rows.append(summary_df)

    if not rows:
        msg = f"No aggregate_summary.csv files found under {input_root}."
        raise FileNotFoundError(msg)

    return pd.concat(rows, ignore_index=True)


def _filter_representative_soft_budgets(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    mask = summary.apply(
        lambda row: (
            row["method"] in REPRESENTATIVE_SOFT_BUDGETS
            and row["eval_soft_budget_param"]
            == REPRESENTATIVE_SOFT_BUDGETS[row["method"]]
        ),
        axis=1,
    )
    return pd.DataFrame(summary[mask]).copy()


def _build_method_styles(
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    colors = _palette_colors()
    markers = ["o", "s", "^", "D", "P", "h", "X", "v"]
    styles: dict[str, dict[str, Any]] = {}
    for idx, method in enumerate(methods):
        styles[method] = {
            "label": METHOD_NAME_MAPPING.get(method, method),
            "color": colors[idx % len(colors)],
            "marker": markers[idx % len(markers)],
        }
    return styles


def _prepare_plot_summary(
    summary: pd.DataFrame,
    *,
    methods: list[str] | None,
    train_initializers: list[str] | None,
) -> pd.DataFrame:
    if methods is not None:
        summary = cast(
            "pd.DataFrame",
            summary.loc[summary["method"].isin(methods)].copy(),
        )
    if train_initializers is not None:
        summary = cast(
            "pd.DataFrame",
            summary.loc[
                summary["train_initializer"].isin(train_initializers)
            ].copy(),
        )
    if summary.empty:
        return pd.DataFrame()

    representative = _filter_representative_soft_budgets(summary)
    if representative.empty:
        return pd.DataFrame()

    group_cols = ["method", "train_initializer", "stop_shield_delta"]
    return (
        representative.groupby(group_cols, as_index=False)
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            accuracy_std=("accuracy_std", "mean"),
            risky_unsafe_stop_rate_mean=(
                "risky_unsafe_stop_rate_mean",
                "mean",
            ),
            risky_unsafe_stop_rate_std=(
                "risky_unsafe_stop_rate_std",
                "mean",
            ),
        )
        .sort_values("stop_shield_delta")
    )


def plot_accuracy_vs_delta(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    methods: list[str] | None,
    train_initializers: list[str] | None,
    risk_budget: float | None,
    formats: list[str],
) -> None:
    prepared = _prepare_plot_summary(
        summary,
        methods=methods,
        train_initializers=train_initializers,
    )
    if prepared.empty:
        print("No data for accuracy_vs_delta plot.")
        return

    prepared.to_csv(
        output_path.parent / "stop_shield_effectiveness_summary.csv",
        index=False,
    )

    all_methods = (
        methods if methods is not None else sorted(prepared["method"].unique())
    )
    styles = _build_method_styles(all_methods)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PLOT_WIDTH * 0.68, PLOT_HEIGHT * 0.72),
        squeeze=False,
    )

    panels = [
        (
            "accuracy_mean",
            "accuracy_std",
            "Accuracy",
            (0.0, 1.0),
        ),
        (
            "risky_unsafe_stop_rate_mean",
            "risky_unsafe_stop_rate_std",
            "Risky Unsafe-Stop Rate",
            (0.0, 1.0),
        ),
    ]

    seen_methods: set[str] = set()
    seen_inits: set[str] = set()

    for ax, (mean_col, std_col, ylabel, ylim) in zip(
        axes[0], panels, strict=True
    ):
        for group_key, grp in prepared.groupby(
            ["method", "train_initializer"]
        ):
            method, train_initializer = cast(
                "tuple[str, str]",
                group_key,
            )
            style = styles.get(
                method,
                {"label": method, "color": "gray", "marker": "o"},
            )
            linestyle = TRAIN_INITIALIZER_LINESTYLES.get(
                train_initializer,
                "-",
            )
            x = grp["stop_shield_delta"].to_numpy()
            y = grp[mean_col].to_numpy()
            yerr = grp[std_col].to_numpy()

            ax.plot(
                x,
                y,
                color=style["color"],
                linestyle=linestyle,
                marker=style["marker"],
                markersize=5,
            )
            if grp.shape[0] > 1:
                ax.fill_between(
                    x,
                    np.clip(y - yerr, *ylim),
                    np.clip(y + yerr, *ylim),
                    color=style["color"],
                    alpha=0.10,
                    linewidth=0,
                )

            seen_methods.add(method)
            seen_inits.add(train_initializer)

        ax.set_xlabel(r"$\delta$ (0 = unshielded)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.18)

        if risk_budget is not None and "unsafe" in mean_col:
            ax.axhline(
                risk_budget,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
            )

    method_handles = [
        Line2D(
            [],
            [],
            color=styles[method]["color"],
            marker=styles[method]["marker"],
            linestyle="-",
            label=styles[method]["label"],
        )
        for method in all_methods
        if method in seen_methods and method in styles
    ]
    init_handles = [
        Line2D(
            [],
            [],
            color="#4C4C4C",
            linestyle=TRAIN_INITIALIZER_LINESTYLES.get(
                train_initializer,
                "-",
            ),
            linewidth=2.2,
            label=TRAIN_INITIALIZER_LABELS.get(
                train_initializer,
                train_initializer.replace("_", " "),
            ),
        )
        for train_initializer in sorted(seen_inits)
    ]

    all_handles = method_handles + init_handles
    if all_handles:
        fig.legend(
            handles=all_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(4, len(all_handles)),
            fontsize=8,
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    for fmt in formats:
        fig.savefig(
            output_path.with_suffix(f".{fmt}"),
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    _apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = _load_summaries(args.input_root)
    except FileNotFoundError as exc:
        print(exc)
        return

    print(
        f"Loaded {len(summary)} rows across "
        f"{summary['stop_shield_delta'].nunique()} delta values."
    )

    plot_accuracy_vs_delta(
        summary,
        args.output_dir / "accuracy_vs_delta",
        methods=args.methods,
        train_initializers=args.train_initializers,
        risk_budget=args.risk_budget,
        formats=args.formats,
    )

    print("Done.")


if __name__ == "__main__":
    main()
