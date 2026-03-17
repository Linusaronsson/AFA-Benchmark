# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Plot dualized-stop effectiveness: accuracy and safety vs lambda.

Reads aggregate_summary.csv files produced by
plot_cube_nm_ar_results.py across dual_lambda sweep directories and
generates:
  1. Accuracy + unsafe-stop-rate vs dual_lambda (two-panel).
  2. Pareto frontier: accuracy vs unsafe-stop-rate.

Usage:
    python scripts/analysis/plot_dualized_stop_effectiveness.py \
        extra/output/plot_results/cube_nm_ar/eval_split-val \
        extra/output/analysis/dualized_stop \
        --budget-mode hard \
        --train-initializers cold cube_nm_ar
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

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

TRAIN_INITIALIZER_LABELS: dict[str, str] = {
    "cold": "Full train support",
    "cube_nm_ar": "Rescue-censored train",
    "mcar_p03": "MCAR 30%",
    "mar_p03": "MAR 30%",
    "mnar_logistic_p03": "MNAR logistic 30%",
}

TRAIN_INITIALIZER_LINESTYLES: dict[
    str, str | tuple[int, tuple[int, ...]]
] = {
    "cold": "-",
    "cube_nm_ar": "--",
    "mcar_p03": ":",
    "mar_p03": "-.",
    "mnar_logistic_p03": (0, (3, 1, 1, 1)),
}


def _palette_colors() -> tuple[str, ...]:
    cmap = plt.get_cmap(COLOR_PALETTE_NAME)
    raw = getattr(cmap, "colors", None)
    if raw is None:
        raw = [cmap(x) for x in np.linspace(0.0, 1.0, 8)]
    return tuple(to_hex(c) for c in raw)


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
    parser.add_argument(
        "--budget-mode",
        choices=["hard", "soft"],
        default="hard",
    )
    parser.add_argument("--risk-budget", type=float, default=None)
    parser.add_argument(
        "--train-initializers",
        nargs="+",
        default=["cold", "cube_nm_ar"],
    )
    parser.add_argument("--eval-initializers", nargs="+", default=["cold"])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
    )
    return parser.parse_args()


def _load_summaries(input_root: Path) -> pd.DataFrame:
    """Load aggregate_summary.csv from each dual_lambda-* directory."""
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"dual_lambda-([^/]+)")

    for path in sorted(
        input_root.glob("dual_lambda-*/aggregate_summary.csv")
    ):
        match = pattern.search(str(path.parent))
        if match is None:
            continue
        dual_lambda = float(match.group(1))
        df = pd.read_csv(path)
        df["dual_lambda"] = dual_lambda
        rows.append(df)

    if not rows:
        msg = (
            "No aggregate_summary.csv files found under "
            f"{input_root}."
        )
        raise FileNotFoundError(msg)
    return pd.concat(rows, ignore_index=True)


def _budget_column(budget_mode: str) -> str:
    if budget_mode == "hard":
        return "eval_hard_budget"
    return "eval_soft_budget_param"


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


def plot_accuracy_vs_lambda(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    methods: list[str] | None,
    risk_budget: float | None,
    formats: list[str],
) -> None:
    """Two-panel: accuracy and unsafe-stop-rate vs dual_lambda."""
    budget_col = _budget_column(budget_mode)
    if methods is not None:
        summary = summary[summary["method"].isin(methods)]
    if summary.empty:
        print("No data for accuracy_vs_lambda plot.")
        return

    # Average across budgets for a single curve per series.
    group_cols = ["method", "train_initializer", "dual_lambda"]
    agg = (
        summary.groupby(group_cols, as_index=False)
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
        .sort_values("dual_lambda")
    )

    all_methods = sorted(agg["method"].unique())
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

    legend_entries: list[dict[str, Any]] = []
    seen_methods: set[str] = set()
    seen_inits: set[str] = set()

    for ax, (mean_col, std_col, ylabel, ylim) in zip(
        axes[0], panels, strict=True
    ):
        for (method, train_init), grp in agg.groupby(
            ["method", "train_initializer"]
        ):
            style = styles.get(
                method,
                {"label": method, "color": "gray", "marker": "o"},
            )
            linestyle = TRAIN_INITIALIZER_LINESTYLES.get(
                train_init, "-"
            )
            x = grp["dual_lambda"].to_numpy()
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
            seen_inits.add(train_init)

        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.18)

        if risk_budget is not None and "unsafe" in mean_col:
            ax.axhline(
                risk_budget,
                color="red",
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                label=f"Risk budget = {risk_budget}",
            )

    # Build legend.
    method_handles = [
        Line2D(
            [],
            [],
            color=styles[m]["color"],
            marker=styles[m]["marker"],
            linestyle="-",
            label=styles[m]["label"],
        )
        for m in sorted(seen_methods)
        if m in styles
    ]
    init_handles = [
        Line2D(
            [],
            [],
            color="#4C4C4C",
            linestyle=TRAIN_INITIALIZER_LINESTYLES.get(i, "-"),
            linewidth=2.2,
            label=TRAIN_INITIALIZER_LABELS.get(
                i, i.replace("_", " ")
            ),
        )
        for i in sorted(seen_inits)
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


def plot_pareto_frontier(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    methods: list[str] | None,
    risk_budget: float | None,
    formats: list[str],
) -> None:
    """Pareto: accuracy vs unsafe-stop-rate, annotated with lambda."""
    if methods is not None:
        summary = summary[summary["method"].isin(methods)]
    if summary.empty:
        print("No data for Pareto frontier plot.")
        return

    group_cols = ["method", "train_initializer", "dual_lambda"]
    agg = (
        summary.groupby(group_cols, as_index=False)
        .agg(
            accuracy_mean=("accuracy_mean", "mean"),
            risky_unsafe_stop_rate_mean=(
                "risky_unsafe_stop_rate_mean",
                "mean",
            ),
        )
        .sort_values("dual_lambda")
    )

    all_methods = sorted(agg["method"].unique())
    styles = _build_method_styles(all_methods)

    fig, ax = plt.subplots(
        figsize=(PLOT_WIDTH * 0.42, PLOT_HEIGHT * 0.72)
    )

    for (method, train_init), grp in agg.groupby(
        ["method", "train_initializer"]
    ):
        style = styles.get(
            method,
            {"label": method, "color": "gray", "marker": "o"},
        )
        linestyle = TRAIN_INITIALIZER_LINESTYLES.get(
            train_init, "-"
        )
        init_label = TRAIN_INITIALIZER_LABELS.get(
            train_init, train_init
        )
        x = grp["risky_unsafe_stop_rate_mean"].to_numpy()
        y = grp["accuracy_mean"].to_numpy()
        lambdas = grp["dual_lambda"].to_numpy()

        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=linestyle,
            marker=style["marker"],
            markersize=5,
            label=f"{style['label']} ({init_label})",
        )
        # Annotate points with lambda values.
        for xi, yi, lam in zip(x, y, lambdas, strict=True):
            ax.annotate(
                f"{lam:.1g}",
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                color=style["color"],
                alpha=0.8,
            )

    if risk_budget is not None:
        ax.axvline(
            risk_budget,
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_xlabel("Risky Unsafe-Stop Rate")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.18)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()

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

    all_summaries: list[pd.DataFrame] = []
    for train_init in args.train_initializers:
        for eval_init in args.eval_initializers:
            init_dir = (
                f"train_initializer-{train_init}"
                f"+eval_initializer-{eval_init}"
            )
            init_root = (
                args.input_root
                / init_dir
                / f"budget_mode-{args.budget_mode}"
            )
            if not init_root.exists():
                print(f"Warning: {init_root} not found, skipping.")
                continue
            try:
                df = _load_summaries(init_root)
            except FileNotFoundError as exc:
                print(f"Warning: {exc}")
                continue
            df["train_initializer"] = train_init
            df["eval_initializer"] = eval_init
            all_summaries.append(df)

    if not all_summaries:
        print(
            "No dualized stop results found. "
            "Run the pipeline with dual_lambdas configured first."
        )
        return

    combined = pd.concat(all_summaries, ignore_index=True)
    print(
        f"Loaded {len(combined)} rows across "
        f"{combined['dual_lambda'].nunique()} lambda values."
    )

    plot_accuracy_vs_lambda(
        combined,
        args.output_dir / "accuracy_vs_lambda",
        budget_mode=args.budget_mode,
        methods=args.methods,
        risk_budget=args.risk_budget,
        formats=args.formats,
    )
    plot_pareto_frontier(
        combined,
        args.output_dir / "pareto_frontier",
        budget_mode=args.budget_mode,
        methods=args.methods,
        risk_budget=args.risk_budget,
        formats=args.formats,
    )

    print("Done.")


if __name__ == "__main__":
    main()
