"""
When training missingness matters, and how much of it comes back.

Two things decide whether restoration is worth anything, and neither is visible
in the direct-versus-generative comparison itself. They are two separate
questions, so this writes two separate figures rather than one two-panel float
whose halves shared nothing but a caption.

The identification figure: the mechanism decides whether the completion is
identified at all. We plot the oracle generator's advantage over the honest one
in reconstruction error on mechanism-missing entries. That is a property of the
generator, not of any method, and it needs no division, which matters because a
recovered share `R / D` is meaningless on the many cells where `D` is near zero.
Self-masking MNAR is the mechanism `prop:mnar` says is not identified, and it is
the one that separates.

The structure figure: the dataset decides how much damage there is in the first
place. Datasets are ordered by `rho_top`, so the claim is the ordering itself,
and the dataset names are axis ticks rather than annotations scattered over the
points, which is what used to collide. Read `rho_top` together with `V_static`
from `tab:route-structure`, since a small route sensitivity means "every route
is equally good" on a saturated dataset and "no fixed route is any good" on one
that needs adaptive acquisition.
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
    DATASET_LABELS_SHORT,
    INDUCED_MECHANISMS,
    MECHANISM_COLORS,
    MECHANISM_LABELS,
    MECHANISM_MARKERS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    PRIMARY_METHODS,
    TEXT_WIDTH_IN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

# The structure figure fixes the rate rather than averaging over it, because
# damage grows with the rate and mixing rates would blur the very spread being
# explained.
STRUCTURE_RATE = 0.7


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


def _draw_identification(axis: Axes, gaps: pd.DataFrame) -> None:
    rates = sorted({float(value) for value in gaps["p"]})
    for mechanism in INDUCED_MECHANISMS:
        subset = _rows(
            gaps, _column(gaps, "mechanism") == mechanism
        ).sort_values("p")
        if subset.empty:
            continue
        # The mechanisms are ordered by how far identification degrades, so they
        # take the sequential ramp rather than four hues. Self-masking MNAR is
        # the darkest and the only solid line because it is the one `prop:mnar`
        # says is not identified. The old encoding drew it in AACO's vermillion,
        # which made a mechanism look like a method.
        unidentified = mechanism == "mnar_self"
        color = MECHANISM_COLORS[mechanism]
        axis.errorbar(
            subset["p"],
            subset["gap"],
            yerr=subset["sem"],
            marker=MECHANISM_MARKERS[mechanism],
            markersize=4.0,
            linewidth=1.6 if unidentified else 1.0,
            linestyle="solid" if unidentified else (0, (4, 2)),
            color=color,
            ecolor=color,
            elinewidth=0.8,
            capsize=2.0,
            label=MECHANISM_LABELS[mechanism],
            zorder=4 if unidentified else 3,
        )
    axis.set_xticks(rates)
    axis.set_xlabel("Missingness rate $p$", fontsize=8)
    axis.set_ylabel("Oracle $-$ honest error", fontsize=8)
    axis.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False, fontsize=6.5, loc="upper left", handlelength=2.4
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _draw_structure(axis: Axes, points: pd.DataFrame) -> None:
    """
    Damage per dataset, datasets ordered by how interchangeable their routes are.

    A dot plot rather than damage against a continuous `rho_top`, because the
    claim is an ordering and the dataset is the unit. Names become axis ticks,
    which is what removes the overlapping annotations the scatter needed, and
    the four methods sit on one row so a reader compares them where they differ
    rather than across two sub-panels.
    """
    order = cast(
        "pd.Series",
        points.groupby("dataset")["rho_top"].first(),
    ).sort_values(ascending=False)
    datasets = [str(name) for name in order.index]
    axis.axvline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
    for row, dataset in enumerate(datasets):
        group = _rows(points, _column(points, "dataset") == dataset)
        # A hairline through each row's methods, so the spread within a dataset
        # reads as one object before the individual methods are picked out.
        axis.plot(
            [float(group["D"].min()), float(group["D"].max())],
            [row, row],
            color=GRID,
            linewidth=2.6,
            solid_capstyle="round",
            zorder=2,
        )
        for method in PRIMARY_METHODS:
            cell = _rows(group, _column(group, "method") == method)
            if cell.empty:
                continue
            axis.scatter(
                cell["D"],
                [row] * len(cell),
                s=22,
                marker=METHOD_MARKERS[method],
                facecolor=METHOD_COLORS[method],
                edgecolor=SURFACE,
                linewidth=0.4,
                zorder=3,
            )
    axis.set_yticks(range(len(datasets)))
    axis.set_yticklabels(
        [DATASET_LABELS_SHORT.get(name, name) for name in datasets],
        fontsize=7,
    )
    axis.set_ylim(-0.7, len(datasets) - 0.3)
    axis.set_xlabel(
        f"Missingness damage $D_r$ at $p={STRUCTURE_RATE:g}$", fontsize=8
    )
    # The ordering variable belongs on the figure, since the ordering is the
    # claim. A right-hand column keeps it out of the data area.
    right = axis.twinx()
    right.set_ylim(axis.get_ylim())
    right.set_yticks(range(len(datasets)))
    right.set_yticklabels(
        [f"{float(order[name]):.2f}" for name in datasets], fontsize=6.5
    )
    # Head the column rather than labelling its middle, where a centred label
    # lands on top of the centre row's own value.
    right.annotate(
        r"$\rho_{\mathrm{top}}$",
        (1.0, 1.0),
        xycoords="axes fraction",
        textcoords="offset points",
        xytext=(26, 5),
        fontsize=8,
        color=INK_MUTED,
        ha="right",
    )
    right.tick_params(length=0, colors=INK_MUTED)
    for spine in ("top", "right", "left"):
        right.spines[spine].set_visible(False)
    axis.grid(True, axis="x", color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot(
    gaps: pd.DataFrame,
    points: pd.DataFrame | None,
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
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    figure.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.16)
    _draw_identification(axis, gaps)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)

    if points is None:
        return

    # The structure claim is its own figure. Two questions that share nothing
    # but a caption do not belong in one float.
    structure_output = output.with_name(
        f"{output.stem}_structure{output.suffix}"
    )
    rows = points["dataset"].nunique()
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 1.05 + 0.30 * rows))
    axis = figure.add_axes((0.16, 0.30 - 0.012 * rows, 0.76, 0.66))
    _draw_structure(axis, points)
    handles = [
        Line2D(
            [],
            [],
            marker=METHOD_MARKERS[method],
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
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.2,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.savefig(structure_output)
    figure.savefig(structure_output.with_suffix(".png"), dpi=200)


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
        "--with-structure",
        action="store_true",
        help=(
            "Add the damage-against-route-structure panel. Needs more than the "
            "three real datasets to say anything, so it is off until "
            "core_group_missingness_v2 lands CUBE and CUBE-NM."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/paper/experiments/results/when.pdf"),
    )
    arguments = parser.parse_args()

    cells = pd.read_csv(arguments.cells)
    gaps = identification_gap(pd.read_csv(arguments.generator_quality))
    points = None
    if arguments.with_structure:
        points = structure_points(
            cells, load_structure(arguments.route_structure)
        )
        if points.empty:
            message = "no structure points collected"
            raise SystemExit(message)
    if gaps.empty:
        message = "no generator cells collected"
        raise SystemExit(message)
    plot(gaps, points, arguments.output)

    print(gaps.round(4).to_string(index=False))
    if points is not None:
        print()
        print(
            points[["dataset", "method", "D", "rho_top", "route_sensitivity"]]
            .round(4)
            .to_string(index=False)
        )
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
