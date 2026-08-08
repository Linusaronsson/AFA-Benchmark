"""
When training missingness matters, and how much of it comes back.

Two things decide whether restoration is worth anything, and neither is visible
in the direct-versus-generative comparison itself.

Panel (a), the mechanism decides whether the completion is identified at all. We
plot the oracle generator's advantage over the honest one in reconstruction
error on mechanism-missing entries. That is a property of the generator, not of
any method, and it needs no division, which matters because a recovered share
`R / D` is meaningless on the many cells where `D` is near zero. Self-masking
MNAR is the mechanism `prop:mnar` says is not identified, and it is the one that
separates.

Panel (b), the dataset decides how much damage there is in the first place. We
plot the damage against the two route-structure descriptors of
`tab:route-structure`. Read them together with `V_static`, since a small route
sensitivity means "every route is equally good" on a saturated dataset and "no
fixed route is any good" on one that needs adaptive acquisition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
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

DATASET_LABELS = {
    "cube": "CUBE",
    "cube_nm": "CUBE-NM",
    "cube_nonuniform_costs": "CUBE-NUC",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES",
    "ckd": "CKD",
    "physionet": "PhysioNet",
}
# Panel (b) fixes the rate rather than averaging over it, because damage grows
# with the rate and mixing rates would blur the very spread being explained.
STRUCTURE_RATE = 0.7
# Three mechanisms leave the completion identified and one does not, so the
# encoding says that rather than giving four arbitrary hues.
IDENTIFIED_INK = "#8a8985"
UNIDENTIFIED = "#D55E00"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def identification_gap(quality: pd.DataFrame) -> pd.DataFrame:
    """
    Oracle minus honest generator in reconstruction error, per mechanism.

    ``rmse_improvement`` is a property of the generator pair, so it repeats
    across the methods trained on the same restored view. Deduplicate first or
    every dataset is counted four times.
    """
    generators = quality.drop_duplicates(
        ["dataset", "mechanism", "p", "instance"]
    )
    return cast(
        "pd.DataFrame",
        generators.groupby(["mechanism", "p"], as_index=False).agg(
            gap=("rmse_improvement", "mean"),
            sem=("rmse_improvement", "sem"),
            n=("rmse_improvement", "size"),
        ),
    )


def structure_points(
    cells: pd.DataFrame, structure: pd.DataFrame
) -> pd.DataFrame:
    """Damage per dataset and method at one rate, joined to route structure."""
    at_rate = _rows(cells, _column(cells, "p") == STRUCTURE_RATE)
    damage = cast(
        "pd.DataFrame",
        at_rate.groupby(["dataset", "method"], as_index=False)[
            "damage"
        ].mean(),
    ).rename(columns={"damage": "D"})
    return damage.merge(structure, on="dataset", how="inner")


def _draw_panel_a(axis: Axes, gaps: pd.DataFrame) -> None:
    rates = sorted({float(value) for value in gaps["p"]})
    for mechanism in INDUCED_MECHANISMS:
        subset = _rows(
            gaps, _column(gaps, "mechanism") == mechanism
        ).sort_values("p")
        if subset.empty:
            continue
        identified = mechanism != "mnar_self"
        color = IDENTIFIED_INK if identified else UNIDENTIFIED
        axis.errorbar(
            subset["p"],
            subset["gap"],
            yerr=subset["sem"],
            marker=MECHANISM_MARKERS[mechanism],
            markersize=4.0,
            linewidth=1.0 if identified else 1.5,
            linestyle=(0, (4, 2)) if identified else "solid",
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            label=MECHANISM_LABELS[mechanism],
            zorder=3 if identified else 4,
        )
    axis.set_xticks(rates)
    axis.set_xlabel("Training missingness rate", fontsize=8)
    axis.set_ylabel("Oracle $-$ honest reconstruction error", fontsize=8)
    axis.set_title(
        "(a) identification decides the generator", fontsize=8.5, pad=5
    )
    axis.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False, fontsize=6.5, loc="upper left", handlelength=2.4
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _draw_panel_b(axes: list[Axes], points: pd.DataFrame) -> None:
    for axis, (column, label) in zip(
        axes,
        [
            ("rho_top", r"$\rho_{\mathrm{top}}$"),
            ("route_sensitivity", r"$\Delta_{\mathrm{route}}$"),
        ],
        strict=True,
    ):
        axis.axhline(0.0, color=GRID, linewidth=0.8, zorder=1)
        for method in PRIMARY_METHODS:
            subset = _rows(points, _column(points, "method") == method)
            if subset.empty:
                continue
            axis.scatter(
                subset[column],
                subset["D"],
                s=18,
                marker="o",
                facecolor=METHOD_COLORS[method],
                edgecolor=SURFACE,
                linewidth=0.4,
                alpha=0.9,
                zorder=3,
            )
        span = float(points[column].max() - points[column].min()) or 1.0
        midpoint = float(points[column].mean())
        for dataset, group in points.groupby("dataset"):
            x = float(group[column].iloc[0])
            # Labels lean away from the nearer axis edge, so neither the y-axis
            # of this panel nor the one beside it is overwritten.
            leans_left = x > midpoint
            axis.annotate(
                DATASET_LABELS.get(str(dataset), str(dataset)),
                (x, float(group["D"].max())),
                textcoords="offset points",
                xytext=(-4 if leans_left else 4, 4),
                fontsize=5.5,
                color=INK_MUTED,
                ha="right" if leans_left else "left",
            )
        axis.set_xlim(
            points[column].min() - 0.22 * span,
            points[column].max() + 0.22 * span,
        )
        axis.set_xlabel(label, fontsize=8)
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel(f"Damage $D$ at $p={STRUCTURE_RATE:g}$", fontsize=8)
    axes[1].tick_params(labelleft=False)


def plot(
    gaps: pd.DataFrame,
    points: pd.DataFrame,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    figure = plt.figure(figsize=(7.1, 3.3))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.20, 1.0, 1.0],
        wspace=0.30,
        left=0.095,
        right=0.985,
        top=0.87,
        bottom=0.29,
    )
    axis_a = figure.add_subplot(grid[0, 0])
    axis_b1 = figure.add_subplot(grid[0, 1])
    axis_b2 = figure.add_subplot(grid[0, 2], sharey=axis_b1)

    _draw_panel_a(axis_a, gaps)
    _draw_panel_b([axis_b1, axis_b2], points)
    # One title centred over the pair, since panel (b) is two axes.
    box1 = axis_b1.get_position()
    box2 = axis_b2.get_position()
    figure.text(
        (box1.x0 + box2.x1) / 2,
        0.925,
        "(b) the dataset decides how much there is",
        fontsize=8.5,
        color=INK,
        ha="center",
    )

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=SURFACE,
            markeredgewidth=0.4,
            label=METHOD_LABELS[method],
        )
        for method in PRIMARY_METHODS
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
        labelcolor=INK_MUTED,
        bbox_to_anchor=(0.5, 0.0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def load_structure(path: Path) -> pd.DataFrame:
    """Accept either the aggregated table or route_redundancy.py's own output."""
    frame = pd.read_csv(path)
    if "top_route_correctness_correlation" not in frame.columns:
        return frame
    renamed = frame.rename(
        columns={
            "top_route_correctness_correlation": "rho_top",
            "static_reference_score": "v_static",
        }
    )
    return cast(
        "pd.DataFrame",
        renamed.groupby("dataset", as_index=False)[
            ["v_static", "route_sensitivity", "rho_top"]
        ].mean(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/main_summary.cells.csv"
        ),
        help="CSV written by plot_main_summary.py --table",
    )
    parser.add_argument(
        "--route-structure",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/route_structure.csv"
        ),
        help=(
            "Aggregated route structure, or route_redundancy_<ns>.csv directly."
        ),
    )
    parser.add_argument(
        "--generator-quality",
        type=Path,
        default=Path(
            "extra/output/missing_data/analysis/"
            "generator_quality_induced_real_missingness_v2_val.csv"
        ),
        help="CSV written by analyze_missing_data_mechanisms.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/paper/experiments/results/when.pdf"),
    )
    arguments = parser.parse_args()

    cells = pd.read_csv(arguments.cells)
    structure = load_structure(arguments.route_structure)
    gaps = identification_gap(pd.read_csv(arguments.generator_quality))
    points = structure_points(cells, structure)
    if gaps.empty or points.empty:
        message = "no cells collected"
        raise SystemExit(message)
    plot(gaps, points, arguments.output)

    print(gaps.round(4).to_string(index=False))
    print()
    print(
        points[["dataset", "method", "D", "rho_top", "route_sensitivity"]]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
