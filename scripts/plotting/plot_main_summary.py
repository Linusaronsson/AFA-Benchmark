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
import numpy.typing as npt
import pandas as pd
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    METHOD_COLORS,
    METHOD_LABELS,
    PRIMARY_METHODS,
    TEXT_WIDTH_IN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
WEDGE = "#f0efec"
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


BOOTSTRAP_DRAWS = 2000
SEED = 0


def _interval(
    values: npt.NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """
    Percentile bootstrap over the dataset instances.

    Five instances is few, which is the regime stratified bootstrap intervals
    exist for; a normal-theory standard error would be assuming more than the
    data supports.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = values[
        rng.integers(0, len(values), (BOOTSTRAP_DRAWS, len(values)))
    ]
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def collect_panel_a(summary_root: Path) -> pd.DataFrame:
    """
    Distance below each method's own complete-data ceiling.

    Reported as a gap rather than a raw score so every dataset shares a scale on
    which zero means the same thing. In raw units ACTG175 spans 0.58 to 0.85 and
    Diabetes 0.60 to 0.69, so one axis compresses both and an 0.02 move reads
    differently per row.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _largest_budget(frame, dataset)
            metric = primary_metric(dataset)
            # summarize_missing_data.py writes score - complete, so negate it.
            gap_column = f"{metric}_gap_to_complete"
            for method in PRIMARY_METHODS:
                per_method = _rows(
                    per_dataset, _column(per_dataset, "method") == method
                )
                if per_method.empty:
                    continue

                def gaps(
                    strategy: str,
                    per_method: pd.DataFrame = per_method,
                    gap_column: str = gap_column,
                ) -> npt.NDArray[np.float64]:
                    cells = _rows(
                        per_method, _column(per_method, "strategy") == strategy
                    )
                    cells = _rows(
                        cells,
                        (_column(cells, "mechanism") == PANEL_MECHANISM)
                        & (_column(cells, "p") == PANEL_RATE),
                    )
                    return -_column(cells, gap_column).dropna().to_numpy()

                direct, generative = gaps(DIRECT), gaps(GENERATIVE)
                if not len(direct) or not len(generative):
                    continue
                direct_lo, direct_hi = _interval(direct, rng)
                generative_lo, generative_hi = _interval(generative, rng)
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "metric": metric,
                        "direct": float(direct.mean()),
                        "direct_lo": direct_lo,
                        "direct_hi": direct_hi,
                        "generative": float(generative.mean()),
                        "generative_lo": generative_lo,
                        "generative_hi": generative_hi,
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
            _column(panel_a, "direct") - _column(panel_a, "generative")
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
            for key, capsize in (("direct", 0.0), ("generative", 0.0)):
                low, high = record[f"{key}_lo"], record[f"{key}_hi"]
                if np.isnan(low):
                    continue
                axis_a.plot(
                    [low, high],
                    [y, y],
                    color=color,
                    linewidth=3.0,
                    alpha=0.22,
                    solid_capstyle="butt",
                    zorder=1 + capsize,
                )
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

    # Zero is the complete-data ceiling for every dataset, which is the whole
    # point of plotting a gap: one scale on which zero means the same thing.
    axis_a.axvline(0.0, color=INK_MUTED, linewidth=0.9, zorder=5)
    axis_a.set_yticks([len(order) - i for i in range(len(order))])
    axis_a.set_yticklabels(
        [DATASET_LABELS[d] for d in order],
        fontsize=7.5,
        color=INK,
    )
    # Ceiling on the right, so the arrow of improvement points rightward.
    axis_a.invert_xaxis()
    axis_a.set_xlabel(
        "Gap below complete-data ceiling", color=INK_MUTED, fontsize=8
    )
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
    # Name the two reference lines, so a point's height is read directly rather
    # than decoded from the definitions of D and R. Everything between them is
    # partial recovery; below zero restoration made the method worse.
    axis_b.fill_between(
        [0.0, hi],
        [0.0, 0.0],
        [0.0, hi],
        color=WEDGE,
        linewidth=0,
        zorder=0,
    )
    axis_b.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=0.9, zorder=2)
    axis_b.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=2)
    axis_b.axvline(0.0, color=GRID, linewidth=0.8, zorder=1)
    axis_b.annotate(
        "full recovery, $R=D$",
        (hi, hi),
        textcoords="offset points",
        xytext=(-3, -9),
        fontsize=6.5,
        color=INK_MUTED,
        ha="right",
    )
    axis_b.annotate(
        "nothing recovered, $R=0$",
        (hi, 0.0),
        textcoords="offset points",
        xytext=(-3, -15),
        fontsize=6.5,
        color=INK_MUTED,
        ha="right",
        va="top",
    )
    # Colour alone carries the method here. Mechanism moved to the
    # identification figure, which took this panel from 16 classes to 4.
    for method in PRIMARY_METHODS:
        subset = _rows(panel_b, _column(panel_b, "method") == method)
        if subset.empty:
            continue
        axis_b.scatter(
            subset["damage"],
            subset["gain"],
            s=17,
            marker="o",
            facecolor=METHOD_COLORS[method],
            edgecolor=SURFACE,
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
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
    height = max(3.4, 1.45 + 0.46 * n_datasets)
    legend_fraction = 1.05 / height
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, height))
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
            linewidth=3.0,
            alpha=0.22,
            solid_capstyle="butt",
            label="95% bootstrap CI",
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
