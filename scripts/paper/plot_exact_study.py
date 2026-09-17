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
    TEXT_WIDTH_IN,
    apply_paper_style,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class Curve(NamedTuple):
    arm: str
    d: int
    p_miss: float
    horizon: int
    action: int = -1
    budget: int = 2


class Estimate(NamedTuple):
    n: npt.NDArray[np.int64]
    mean: npt.NDArray[np.float64]
    ci95: npt.NDArray[np.float64]


PANELS = (
    ("mask_local", "(a) Filtering"),
    ("mask_agnostic", "(b) Aliasing"),
    ("generative", "(c) Generative Restoration"),
)
COLORS = {0.3: "#9b80c6", 0.5: "#6b479c", 0.7: "#331f57"}

DIMENSION_COLORS = ("#0072B2", "#D55E00", "#009E73")
BUDGET_COLORS = ("#882255", "#7570B3", "#8C6D00")

MYOPIC_ALPHA = 0.4
MARKERS = {6: "o", 8: "s", 10: "D"}
BUDGET_MARKERS = {2: "o", 3: "s", 4: "D"}

MAIN_RATE = 0.5
POLICIES = (
    (r"Full-horizon ($\widehat Q_b$)", "-", False),
    (r"Myopic ($\widehat Q_1$)", "--", True),
)


def read_means(path: Path, *, values: bool = False) -> dict[Curve, Estimate]:
    """Aggregate replicates; intervals are pointwise normal 95% mean CIs."""
    metric = "accuracy" if values else "regret"
    cells: dict[tuple[Curve, int], list[float]] = defaultdict(list)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"arm", "d", "p_miss", "n", "rep", "horizon", metric}
        if values:
            required.add("action")
        if not required.issubset(reader.fieldnames or ()):
            message = (
                f"{path} lacks paired-horizon columns; regenerate with "
                "scripts/paper/exact_study.py."
            )
            raise ValueError(message)
        seen = set()
        for row in reader:
            key = Curve(
                row["arm"],
                int(row["d"]),
                float(row["p_miss"]),
                int(row["horizon"]),
                int(row["action"]) if values else -1,
                int(row.get("budget", 2)),
            )
            n, rep, value = int(row["n"]), int(row["rep"]), float(row[metric])
            if (
                key.horizon not in (1, key.budget)
                or n <= 0
                or not np.isfinite(value)
            ):
                message = f"Invalid study result in {path}: {row}"
                raise ValueError(message)
            identity = (key, n, rep)
            if identity in seen:
                message = f"Duplicate study replicate in {path}: {identity}"
                raise ValueError(message)
            seen.add(identity)
            cells[key, n].append(value)

    # Both horizons must use the same replicate cells, not separate runs.
    for key, n, rep in seen:
        other = key._replace(horizon=key.budget if key.horizon == 1 else 1)
        if (other, n, rep) not in seen:
            message = f"Unpaired horizon in {path}: {(key, n, rep)}"
            raise ValueError(message)

    curves: dict[Curve, list[tuple[int, float, float]]] = defaultdict(list)
    for (key, n), samples in cells.items():
        ci = (
            1.96 * float(np.std(samples, ddof=1)) / np.sqrt(len(samples))
            if len(samples) > 1
            else float("nan")
        )
        curves[key].append((n, float(np.mean(samples)), float(ci)))
    return {
        key: Estimate(
            np.asarray([n for n, _, _ in sorted(points)], dtype=np.int64),
            np.asarray([mean for _, mean, _ in sorted(points)]),
            np.asarray([ci for _, _, ci in sorted(points)]),
        )
        for key, points in curves.items()
    }


def _figure() -> tuple[Figure, Sequence[Sequence[Axes]]]:
    apply_paper_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(TEXT_WIDTH_IN, 3.75),
        sharex=True,
        sharey=True,
    )
    return figure, cast("Sequence[Sequence[Axes]]", axes)


def _style_axis(axis: Axes, title: str, horizon: int) -> None:
    if horizon == 1:
        axis.set_title(title, fontsize=7.5, color=INK, pad=5)
    axis.set_xscale("log")
    axis.grid(axis="y", color=GRID, linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(length=2.5, labelsize=7)


def _save(figure: Figure, output_stem: Path, handles: list[Line2D]) -> None:
    figure.supxlabel(
        "Training Instances", x=0.54, y=0.10, fontsize=8, color=INK
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        handlelength=1.4,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.98,
        top=0.91,
        bottom=0.21,
        wspace=0.12,
        hspace=0.20,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"))
    figure.savefig(output_stem.with_suffix(".png"), dpi=240)
    plt.close(figure)


def _rate_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, linewidth=1.2, label=f"$p={p:g}$")
        for p, color in COLORS.items()
    ]


def render(
    input_path: Path, output_stem: Path, *, vary_budget: bool = False
) -> None:
    curves = read_means(input_path)
    dimensions = sorted(
        {key.budget if vary_budget else key.d for key in curves}
    )
    markers = BUDGET_MARKERS if vary_budget else MARKERS
    figure, axes = _figure()
    for horizon, row in enumerate(axes, start=1):
        for axis, (arm, title) in zip(row, PANELS, strict=True):
            for p_miss, color in COLORS.items():
                for d in dimensions:
                    key = (
                        Curve(
                            arm, 10, p_miss, 1 if horizon == 1 else d, budget=d
                        )
                        if vary_budget
                        else Curve(arm, d, p_miss, horizon)
                    )
                    x, y, _ = curves[key]
                    axis.plot(
                        x,
                        y,
                        color=color,
                        linewidth=0.9,
                        marker=markers[d],
                        markersize=2.8,
                        markeredgewidth=0.45,
                        markevery=2,
                    )
            if arm == "mask_agnostic" and horizon == 2:
                axis.axhline(0.25, color=INK, linewidth=0.7, linestyle=":")
                label = "$p = 0.7$ Limit"
                axis.text(
                    0.97,
                    0.54,
                    label,
                    transform=axis.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=6,
                )
            _style_axis(axis, title, horizon)
            axis.set_ylim(-0.015, 0.52)
            axis.set_yticks((0.0, 0.25, 0.5))
        name = (
            "Myopic"
            if horizon == 1
            else ("Full-horizon" if vary_budget else "Two-step")
        )
        label = "b" if vary_budget and horizon != 1 else str(horizon)
        row[0].set_ylabel(
            f"{name} ($Q_{label}$)\nEvaluation Regret", fontsize=8
        )

    handles = _rate_handles() + [
        Line2D(
            [0],
            [0],
            color=INK,
            marker=markers[d],
            linestyle="none",
            markersize=4,
            label=f"${'b' if vary_budget else 'd'}={d}$",
        )
        for d in dimensions
    ]
    _save(figure, output_stem, handles)


def render_combined(
    input_path: Path, budget_input: Path, output_stem: Path
) -> None:
    figure, axes = _figure()
    figure.set_size_inches(TEXT_WIDTH_IN, 3.9)
    sweeps = (
        (
            read_means(input_path),
            DIMENSION_COLORS,
            (6, 8, 10),
            "d",
            r"Dimension sweep ($b=2$)",
        ),
        (
            read_means(budget_input),
            BUDGET_COLORS,
            (2, 3, 4),
            "b",
            r"Budget sweep ($d=10$)",
        ),
    )
    for row_index, (curves, palette, values, parameter, label) in enumerate(
        sweeps
    ):
        row = axes[row_index]
        for axis, (arm, title) in zip(row, PANELS, strict=True):
            for color, value in zip(palette, values, strict=True):
                budget = 2 if parameter == "d" else value
                dimension = value if parameter == "d" else 10
                for _, linestyle, myopic in POLICIES:
                    key = Curve(
                        arm,
                        dimension,
                        MAIN_RATE,
                        1 if myopic else budget,
                        budget=budget,
                    )
                    x, y, _ = curves[key]
                    axis.plot(
                        x,
                        y,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.2,
                        alpha=MYOPIC_ALPHA if myopic else 1.0,
                        zorder=2 if myopic else 3,
                    )
            _style_axis(axis, title, row_index + 1)
            axis.set_ylim(-0.015, 0.52)
            axis.set_yticks((0.0, 0.25, 0.5))
        row[0].set_ylabel(f"{label}\nEvaluation Regret", fontsize=8)
    handles = [
        Line2D([0], [0], color=color, linewidth=1.4, label=f"${name}={value}$")
        for palette, values, name in (
            (DIMENSION_COLORS, (6, 8, 10), "d"),
            (BUDGET_COLORS, (2, 3, 4), "b"),
        )
        for color, value in zip(palette, values, strict=True)
    ] + [
        Line2D(
            [0],
            [0],
            color=INK,
            linestyle=linestyle,
            linewidth=1.2,
            alpha=MYOPIC_ALPHA if myopic else 1.0,
            label=label,
        )
        for label, linestyle, myopic in POLICIES
    ]
    _save(figure, output_stem, handles)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--vary-budget", action="store_true")
    layout.add_argument(
        "--budget-input",
        type=Path,
        help="Combine the dimension input with this budget sweep CSV.",
    )
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
    if args.budget_input:
        render_combined(args.input, args.budget_input, args.output_stem)
    else:
        render(args.input, args.output_stem, vary_budget=args.vary_budget)
    print(f"wrote figures to {args.output_stem.parent}")


if __name__ == "__main__":
    main()
