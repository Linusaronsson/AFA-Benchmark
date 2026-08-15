"""
Why direct learning appeared to help OL: the Q-function's state was short.

`PROJECT.md` carried this as an open question. OL trained on the restricted view
used to score *above* its own complete-data ceiling, so restoration could only
look harmful, and that failed the preregistered concordance on CUBE-NM.

The v2 matrices run two OL states against each other. `ol_with_mask` sees the
acquired values and the acquired-feature mask; `ol_full_state` additionally sees
the currently legal continuation-action mask in Q. Only the second one is told
which acquisitions the training episode may still make, which is exactly what
training missingness removes.

We plot the difference between them rather than both levels, because the
difference is the claim. Zero means the two states agree, so the dilemma opening
under missingness and closing again after restoration is the shape of the
figure. The complete-data difference is drawn as a reference: it sits at zero on
every dataset, which is what rules out a capacity or optimisation handicap and
leaves missingness as the only thing separating them.
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
    DATASET_LABELS,
    GRID,
    INK_MUTED,
    LEGEND_STRIP_IN,
    MECHANISM_LABELS,
    METHOD_COLORS,
    SURFACE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}
SHORT, FULL = "ol_with_mask", "ol_full_state"
DIRECT = "restricted"
GENERATIVE = "pvae_label_conditioned"

# Same namespace map as the main figure, so the two never disagree about which
# datasets the induced arm contains.
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
        "miniboone",
    ],
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in ACCURACY_DATASETS else "f_score"


def collect(summary_root: Path, mechanism: str) -> pd.DataFrame:
    """Measure full-state minus acquired-mask, per dataset and rate."""
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _rows(frame, _column(frame, "dataset") == dataset)
            if per_dataset.empty:
                continue
            budget = _column(per_dataset, "eval_hard_budget").max()
            per_dataset = _rows(
                per_dataset, _column(per_dataset, "eval_hard_budget") == budget
            )
            metric = primary_metric(dataset)

            def score(
                method: str,
                strategy: str,
                rate: float | None,
                per_dataset: pd.DataFrame = per_dataset,
                metric: str = metric,
                mechanism: str = mechanism,
            ) -> float:
                cell = _rows(
                    per_dataset,
                    (_column(per_dataset, "method") == method)
                    & (_column(per_dataset, "strategy") == strategy),
                )
                if rate is not None:
                    cell = _rows(
                        cell,
                        (_column(cell, "mechanism") == mechanism)
                        & (_column(cell, "p") == rate),
                    )
                return float(_column(cell, metric).mean())

            complete = score(FULL, "complete", None) - score(
                SHORT, "complete", None
            )
            rates = sorted(
                {
                    float(value)
                    for value in _column(per_dataset, "p")
                    if float(value) > 0.0
                }
            )
            rows.extend(
                {
                    "dataset": dataset,
                    "p": rate,
                    "complete": complete,
                    "direct": score(FULL, DIRECT, rate)
                    - score(SHORT, DIRECT, rate),
                    "restored": score(FULL, GENERATIVE, rate)
                    - score(SHORT, GENERATIVE, rate),
                }
                for rate in rates
            )
    return pd.DataFrame(rows)


# The quantity is the full state's disadvantage, so it takes that method's
# colour; fill separates the training view, as in every other figure.
STATE_COLOR = METHOD_COLORS[FULL]
SERIES = (
    ("direct", "Direct learning", (0, (4, 2)), SURFACE),
    ("restored", "Episode-start restoration", "solid", STATE_COLOR),
)


def _draw(axis: Axes, per_dataset: pd.DataFrame, dataset: str) -> None:
    ordered = per_dataset.sort_values("p")
    rates = [float(value) for value in ordered["p"]]
    # Zero is "the two states agree", which is where complete data puts them.
    axis.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=2)
    axis.plot(
        rates,
        ordered["complete"],
        color=GRID,
        linewidth=2.4,
        solid_capstyle="round",
        zorder=1,
    )
    for column, _, linestyle, facecolor in SERIES:
        axis.plot(
            rates,
            ordered[column],
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            linestyle=linestyle,
            color=STATE_COLOR,
            markerfacecolor=facecolor,
            markeredgecolor=STATE_COLOR,
            markeredgewidth=1.1,
            zorder=3,
        )
    axis.set_xticks(rates)
    axis.set_xticklabels([f"{rate:g}" for rate in rates], fontsize=7)
    axis.set_xlim(min(rates) - 0.07, max(rates) + 0.07)
    axis.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=8)
    axis.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot(frame: pd.DataFrame, mechanism: str, output: Path) -> None:
    apply_paper_style()
    # Datasets ordered by how far the two states separate, so the panels that
    # carry the finding come first.
    spread = cast(
        "pd.Series",
        frame.groupby("dataset")["direct"].apply(lambda s: s.abs().max()),
    ).sort_values(ascending=False)
    datasets = [str(name) for name in spread.index]
    columns = 4 if len(datasets) > 6 else min(3, len(datasets))
    rows = -(-len(datasets) // columns)
    height = 1.35 + 1.55 * rows
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, height),
        squeeze=False,
        sharey=True,
    )
    for index, dataset in enumerate(datasets):
        _draw(
            axes[index // columns][index % columns],
            _rows(frame, _column(frame, "dataset") == dataset),
            dataset,
        )
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.supxlabel(
        f"{MECHANISM_LABELS.get(mechanism, mechanism)} missingness rate $p$",
        fontsize=8,
        y=LEGEND_STRIP_IN * 0.65 / height,
    )

    figure.supylabel("$Q(x_S,S,m) - Q(x_S,S)$", fontsize=8, x=0.015)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=4.0,
            linewidth=1.5,
            linestyle=linestyle,
            color=STATE_COLOR,
            markerfacecolor=facecolor,
            markeredgecolor=STATE_COLOR,
            markeredgewidth=1.1,
            label=label,
        )
        for _, label, linestyle, facecolor in SERIES
    ]
    handles.append(
        Line2D([], [], color=GRID, linewidth=2.4, label="Complete data")
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.4,
        handlelength=2.0,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.92,
        bottom=LEGEND_STRIP_IN / height,
        hspace=0.45,
        wspace=0.30,
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
    parser.add_argument("--mechanism", default="mcar")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/ol_state.pdf",
        ),
    )
    arguments = parser.parse_args()

    frame = collect(arguments.summary_root, arguments.mechanism)
    if frame.empty:
        message = "no cells collected"
        raise SystemExit(message)
    plot(frame, arguments.mechanism, arguments.output)
    print(frame.round(3).to_string(index=False))
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
    main()
