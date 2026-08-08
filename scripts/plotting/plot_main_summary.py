"""Render the paper's main figure, direct learning against generative restoration."""

# Panel (a) is what a practitioner gets, one dumbbell per dataset and method in
# that dataset's own primary metric with the complete-data ceiling ticked. The
# comparison is within a row, so mixing accuracy and macro-F1 across rows is
# sound. Panel (b) is why, the restoration gain against the missingness damage
# over every dataset, method, mechanism and rate.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    INDUCED_MECHANISMS,
    MECHANISM_LABELS,
    MECHANISM_MARKERS,
    METHOD_COLORS,
    METHOD_LABELS,
    PRIMARY_METHODS,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}

# Current induced-missingness confirmatory namespaces. The factual-native arm
# has neither an induced MCAR 0.5 cell nor a counterfactual complete-data
# ceiling, so it belongs in a separate panel rather than this figure.
SOURCES = {
    "core_group_missingness_v2": ["cube_nm", "cube"],
    "induced_nonuniform_missingness_v2": [
        "cube_nonuniform_costs",
        "heart_disease",
    ],
    "induced_real_missingness_v2": [
        "actg",
        "diabetes",
        "nhanes_mortality",
    ],
}

DATASET_LABELS = {
    "cube_nm": "CUBE-NM",
    "cube": "CUBE",
    "cube_nonuniform_costs": "CUBE non-uniform cost",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES mortality",
}

DIRECT = "restricted"
GENERATIVE = "pvae_label_conditioned"
PANEL_MECHANISM, PANEL_RATE = "mcar", 0.5


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Typed column access, since a bare lookup widens to include ndarray."""
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Typed boolean selection, for the same reason."""
    return cast("pd.DataFrame", frame[mask])


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in ACCURACY_DATASETS else "f_score"


def _largest_budget(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    subset = _rows(frame, _column(frame, "dataset") == dataset)
    budget = _column(subset, "eval_hard_budget")
    return _rows(subset, budget == budget.max())


def collect_panel_a(summary_root: Path) -> pd.DataFrame:
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _largest_budget(frame, dataset)
            metric = primary_metric(dataset)
            for method in PRIMARY_METHODS:
                per_method = _rows(
                    per_dataset, _column(per_dataset, "method") == method
                )
                if per_method.empty:
                    continue

                def score(
                    strategy: str,
                    per_method: pd.DataFrame = per_method,
                    metric: str = metric,
                ) -> float:
                    cells = _rows(
                        per_method, _column(per_method, "strategy") == strategy
                    )
                    if strategy != "complete":
                        cells = _rows(
                            cells,
                            (_column(cells, "mechanism") == PANEL_MECHANISM)
                            & (_column(cells, "p") == PANEL_RATE),
                        )
                    values = _column(cells, metric).dropna()
                    return (
                        float(values.mean()) if len(values) else float("nan")
                    )

                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "direct": score(DIRECT),
                        "generative": score(GENERATIVE),
                        "ceiling": score("complete"),
                    }
                )
    return pd.DataFrame(rows)


def collect_panel_b(summary_root: Path) -> pd.DataFrame:
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _largest_budget(frame, dataset)
            metric = primary_metric(dataset)
            for method in PRIMARY_METHODS:
                per_method = _rows(
                    per_dataset, _column(per_dataset, "method") == method
                )
                if per_method.empty:
                    continue
                complete = _rows(
                    per_method, _column(per_method, "strategy") == "complete"
                )[metric].mean()
                grouped = cast("Any", per_method.groupby(["mechanism", "p"]))
                for (mechanism, rate), cell in grouped:
                    if mechanism == "none":
                        continue
                    direct = _rows(cell, _column(cell, "strategy") == DIRECT)[
                        metric
                    ].mean()
                    generative = _rows(
                        cell, _column(cell, "strategy") == GENERATIVE
                    )[metric].mean()
                    triple = [
                        float(complete),
                        float(direct),
                        float(generative),
                    ]
                    if np.isnan(triple).any():
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "mechanism": mechanism,
                            "p": rate,
                            "damage": complete - direct,
                            "gain": generative - direct,
                        }
                    )
    return pd.DataFrame(rows)


def _draw_panel_a(axis_a: Axes, panel_a: pd.DataFrame) -> None:
    # Datasets ordered by the largest move any method makes on them.
    moves = panel_a.assign(
        move=(
            _column(panel_a, "generative") - _column(panel_a, "direct")
        ).abs()
    )
    ranked = cast(
        "pd.Series", moves.groupby("dataset")["move"].max()
    ).sort_values(ascending=False)
    order = [str(dataset) for dataset in ranked.index]
    offsets = {
        "aaco": 0.30,
        "dime": 0.10,
        "ol_with_mask": -0.10,
        "ol_full_state": -0.30,
    }

    for row_index, dataset in enumerate(order):
        base = len(order) - row_index
        for method, offset in offsets.items():
            cell = panel_a[
                (panel_a["dataset"] == dataset) & (panel_a["method"] == method)
            ]
            if cell.empty:
                continue
            record = cast("Any", cell.iloc[0])
            y = base + offset
            color = METHOD_COLORS[method]
            axis_a.plot(
                [record["direct"], record["generative"]],
                [y, y],
                color=color,
                linewidth=1.4,
                solid_capstyle="round",
                zorder=2,
            )
            axis_a.scatter(
                [record["direct"]],
                [y],
                s=16,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            axis_a.scatter(
                [record["generative"]], [y], s=20, color=color, zorder=4
            )
            if not np.isnan(record["ceiling"]):
                axis_a.plot(
                    [record["ceiling"], record["ceiling"]],
                    [y - 0.16, y + 0.16],
                    color=INK_MUTED,
                    linewidth=0.9,
                    zorder=5,
                )

    axis_a.set_yticks([len(order) - i for i in range(len(order))])
    axis_a.set_yticklabels(
        [DATASET_LABELS[d] for d in order],
        fontsize=7.5,
        color=INK,
    )
    axis_a.set_xlabel("Primary metric", color=INK_MUTED, fontsize=8)
    axis_a.set_title(
        "(a) direct learning $\\rightarrow$ generative",
        fontsize=8.5,
        color=INK,
        pad=5,
    )
    axis_a.grid(True, axis="x", color=GRID, linewidth=0.4, alpha=0.7)
    axis_a.set_axisbelow(True)
    axis_a.set_ylim(0.4, len(order) + 0.7)
    for spine in ("top", "right", "left"):
        axis_a.spines[spine].set_visible(False)
    axis_a.tick_params(axis="y", length=0)


def _draw_panel_b(axis_b: Axes, panel_b: pd.DataFrame) -> None:
    lo = min(panel_b["damage"].min(), panel_b["gain"].min()) - 0.03
    hi = max(panel_b["damage"].max(), panel_b["gain"].max()) + 0.03
    axis_b.plot(
        [lo, hi],
        [lo, hi],
        color=GRID,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        zorder=1,
    )
    axis_b.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
    axis_b.axvline(0.0, color=GRID, linewidth=0.8, zorder=1)
    # Colour carries the method, marker carries the mechanism, so a reader can
    # ask either "which method" or "which mechanism" of the same cloud.
    for method in PRIMARY_METHODS:
        color = METHOD_COLORS[method]
        for mechanism in INDUCED_MECHANISMS:
            subset = _rows(
                panel_b,
                (_column(panel_b, "method") == method)
                & (_column(panel_b, "mechanism") == mechanism),
            )
            if subset.empty:
                continue
            axis_b.scatter(
                subset["damage"],
                subset["gain"],
                s=16,
                marker=MECHANISM_MARKERS[mechanism],
                facecolor=color,
                edgecolor=SURFACE,
                linewidth=0.4,
                alpha=0.85,
                zorder=3,
            )
    lines = [
        f"{METHOD_LABELS[m]} $r={_correlation(panel_b, m):.2f}$"
        for m in PRIMARY_METHODS
    ]
    axis_b.text(
        0.04,
        0.96,
        "\n".join(lines),
        transform=axis_b.transAxes,
        fontsize=7,
        color=INK_MUTED,
        va="top",
        linespacing=1.5,
    )
    axis_b.set_xlim(lo, hi)
    axis_b.set_ylim(lo, hi)
    axis_b.set_aspect("equal")
    # Square data box, so anchor it to the top of its cell to keep the two
    # panel titles on one line.
    axis_b.set_anchor("C")
    axis_b.set_xlabel("Missingness damage $D_r$", color=INK_MUTED, fontsize=8)
    axis_b.set_ylabel("Restoration gain $R_r$", color=INK_MUTED, fontsize=8)
    axis_b.set_title(
        "(b) the gain tracks the damage", fontsize=8.5, color=INK, pad=5
    )
    axis_b.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    axis_b.set_axisbelow(True)
    for spine in ("top", "right"):
        axis_b.spines[spine].set_visible(False)


def _correlation(panel_b: pd.DataFrame, method: str) -> float:
    subset = _rows(panel_b, _column(panel_b, "method") == method)
    return float(_column(subset, "damage").corr(_column(subset, "gain")))


def plot(panel_a: pd.DataFrame, panel_b: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )

    # Panel (a) grows a band per dataset, so the figure has to grow with it or
    # the four method rows inside each band collide.
    n_datasets = panel_a["dataset"].nunique()
    height = max(3.9, 1.55 + 0.52 * n_datasets)
    legend_fraction = 1.05 / height
    figure = plt.figure(figsize=(7.1, height))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.25, 1.0],
        wspace=0.28,
        left=0.21,
        right=0.97,
        top=1.0 - 0.28 / height,
        bottom=legend_fraction,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b = figure.add_subplot(grid[0, 1])

    _draw_panel_a(axis_a, panel_a)
    _draw_panel_b(axis_b, panel_b)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            label=METHOD_LABELS[method],
        )
        for method in PRIMARY_METHODS
    ]
    handles += [
        Line2D(
            [],
            [],
            marker=MECHANISM_MARKERS[mechanism],
            linestyle="none",
            markersize=4.0,
            markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE,
            markeredgewidth=0.4,
            label=MECHANISM_LABELS[mechanism],
        )
        for mechanism in INDUCED_MECHANISMS
    ]
    handles += [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED,
            markeredgewidth=1.0,
            label="Direct learning",
        ),
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linewidth=0.9,
            label="Complete-data ceiling",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=7.5,
        labelcolor=INK_MUTED,
        bbox_to_anchor=(0.5, 0.005),
        columnspacing=1.4,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/missing_data/analysis_figures/main_summary.pdf"
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=(
            "Write the per-cell damage and restoration gain behind panel (b). "
            "Defaults to <output>.cells.csv."
        ),
    )
    arguments = parser.parse_args()

    panel_a = collect_panel_a(arguments.summary_root)
    panel_b = collect_panel_b(arguments.summary_root)
    if panel_a.empty or panel_b.empty:
        message = "no cells collected"
        raise SystemExit(message)
    plot(panel_a, panel_b, arguments.output)

    table = arguments.table or arguments.output.with_suffix(".cells.csv")
    panel_b.to_csv(table, index=False)

    print(f"panel (a) rows: {len(panel_a)}   panel (b) cells: {len(panel_b)}")
    print(
        panel_a.assign(move=panel_a["generative"] - panel_a["direct"])
        .round(3)
        .to_string(index=False)
    )
    for method in PRIMARY_METHODS:
        subset = _rows(panel_b, _column(panel_b, "method") == method)
        print(
            f"  {METHOD_LABELS[method]:5s} n={len(subset):3d} "
            f"r={_correlation(panel_b, method):+.3f}"
        )
    print(f"wrote {arguments.output} and {table}")


if __name__ == "__main__":
    main()
