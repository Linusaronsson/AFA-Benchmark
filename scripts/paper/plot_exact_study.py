"""Plot the exact finite-state control study used in the paper."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    GRID,
    INK,
    INK_MUTED,
    TEXT_WIDTH_IN,
    apply_paper_style,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes


class Curve(NamedTuple):
    arm: str
    d: int
    p_miss: float


PANELS = (
    ("mask_local", "(a) Filtering"),
    ("mask_agnostic", "(b) Aliasing"),
    ("generative", "(c) Generative restoration"),
)
COLORS = {0.3: "#b7a3d4", 0.5: "#7752a5", 0.7: "#37225c"}
MARKERS = {6: "o", 8: "s", 10: "D"}


def read_means(
    path: Path,
) -> dict[Curve, tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]]:
    """Read replicate-level results and average each plotted cell."""
    cells: dict[tuple[str, int, float, int], list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cells[
                (
                    row["arm"],
                    int(row["d"]),
                    float(row["p_miss"]),
                    int(row["n"]),
                )
            ].append(float(row["regret"]))

    curves: dict[Curve, list[tuple[int, float]]] = defaultdict(list)
    for (arm, d, p_miss, n), values in cells.items():
        curves[Curve(arm, d, p_miss)].append((n, float(np.mean(values))))

    return {
        key: (
            np.asarray([n for n, _ in sorted(points)], dtype=np.int64),
            np.asarray([mean for _, mean in sorted(points)], dtype=np.float64),
        )
        for key, points in curves.items()
    }


def render(input_path: Path, output_stem: Path) -> None:
    curves = read_means(input_path)
    apply_paper_style()
    figure, axes_object = plt.subplots(
        1,
        3,
        figsize=(TEXT_WIDTH_IN, 2.12),
        sharex=True,
        sharey=True,
    )
    axes = cast("Sequence[Axes]", axes_object)

    for axis, (arm, title) in zip(axes, PANELS, strict=True):
        for p_miss, color in COLORS.items():
            for d, marker in MARKERS.items():
                key = Curve(arm, d, p_miss)
                x, y = curves[key]
                axis.plot(
                    x,
                    y,
                    color=color,
                    linewidth=0.9,
                    marker=marker,
                    markersize=2.8,
                    markeredgewidth=0.45,
                    markevery=2,
                )
        if arm == "mask_agnostic":
            axis.axhline(0.25, color=INK_MUTED, linewidth=0.7, linestyle="--")
            axis.text(
                8.5e4,
                0.263,
                "$p=0.7$ limit",
                ha="right",
                va="bottom",
                fontsize=6.5,
            )
        axis.set_title(title, fontsize=8.5, color=INK, pad=4)
        axis.set_xscale("log")
        axis.set_xlim(9, 1.2e5)
        axis.set_ylim(-0.015, 0.52)
        axis.set_yticks((0.0, 0.25, 0.5))
        axis.grid(axis="y", color=GRID, linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, labelsize=7)

    axes[0].set_ylabel("Evaluation regret")
    figure.supxlabel(
        "Training data instances", x=0.48, y=0.16, fontsize=8, color=INK_MUTED
    )

    handles = [
        Line2D([0], [0], color=color, linewidth=1.2, label=f"$p={p_miss:g}$")
        for p_miss, color in COLORS.items()
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color=INK,
            marker=marker,
            linestyle="none",
            markersize=4,
            label=f"$d={d}$",
        )
        for d, marker in MARKERS.items()
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(
        left=0.09, right=0.995, top=0.86, bottom=0.31, wspace=0.11
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight"
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("extra/output/paper/experiments/results/exact_study.csv"),
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("extra/output/paper/experiments/results/exact_study_raw"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(args.input, args.output_stem)
    print(f"wrote {args.output_stem}.pdf and {args.output_stem}.png")


if __name__ == "__main__":
    main()
