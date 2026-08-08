"""
Why direct learning appeared to help OL: the Q-function's state was short.

`PROJECT.md` carried this as an open question. OL trained on the restricted view
used to score *above* its own complete-data ceiling, so restoration could only
look harmful, and that failed the preregistered concordance on CUBE-NM.

The v2 matrices run two OL states against each other. `ol_with_mask` sees the
acquired values and the acquired-feature mask; `ol_full_state` additionally
sees the currently legal continuation-action mask in Q. Only the second one is
told which acquisitions the training episode may still make, which is exactly
what training missingness removes. This figure puts them side by side.
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
    METHOD_COLORS,
    METHOD_LABELS,
    TEXT_WIDTH_IN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}
OL_STATES = ("ol_with_mask", "ol_full_state")
# The rungs that answer "was there damage, and did restoring the action space
# undo it", in the order a reader climbs them.
RUNGS = (
    ("restricted", "Direct"),
    ("pvae_label_conditioned", "Restored"),
    ("complete", "Complete"),
)
DATASET_LABELS = {
    "cube": "CUBE",
    "cube_nm": "CUBE-NM",
    "cube_nonuniform_costs": "CUBE non-uniform cost",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES mortality",
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in ACCURACY_DATASETS else "f_score"


def collect(
    metrics: Path, dataset: str, mechanism: str
) -> tuple[pd.DataFrame, str]:
    """Mean score per OL state, rung and rate, at the dataset's largest budget."""
    frame = pd.read_csv(metrics)
    frame = _rows(frame, _column(frame, "dataset") == dataset)
    if frame.empty:
        message = f"no rows for dataset {dataset} in {metrics}"
        raise SystemExit(message)
    budget = _column(frame, "eval_hard_budget").max()
    frame = _rows(frame, _column(frame, "eval_hard_budget") == budget)
    metric = primary_metric(dataset)

    rows = []
    for method in OL_STATES:
        per_method = _rows(frame, _column(frame, "method") == method)
        # The complete-data cell carries mechanism "none", so it is one value
        # per method rather than one per rate. It is drawn as the ceiling.
        ceiling = _rows(
            per_method, _column(per_method, "strategy") == "complete"
        )[metric].mean()
        for rate in sorted(
            {
                float(value)
                for value in _column(per_method, "p")
                if float(value) > 0.0
            }
        ):
            cell = _rows(
                per_method,
                (_column(per_method, "mechanism") == mechanism)
                & (_column(per_method, "p") == rate),
            )
            for strategy, label in RUNGS:
                score = (
                    ceiling
                    if strategy == "complete"
                    else _rows(cell, _column(cell, "strategy") == strategy)[
                        metric
                    ].mean()
                )
                rows.append(
                    {
                        "method": method,
                        "rate": rate,
                        "rung": label,
                        "score": float(score),
                    }
                )
    return pd.DataFrame(rows), metric


def plot(
    frame: pd.DataFrame,
    dataset: str,
    mechanism: str,
    metric: str,
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
    rates = sorted(set(frame["rate"]))
    figure, axes = plt.subplots(
        1, 2, figsize=(TEXT_WIDTH_IN, 2.8), sharey=True, sharex=True
    )
    for axis, method in zip(
        cast("list[Axes]", list(axes)), OL_STATES, strict=True
    ):
        per_method = _rows(frame, _column(frame, "method") == method)
        color = METHOD_COLORS[method]
        for index, (_, label) in enumerate(RUNGS):
            rung = _rows(per_method, _column(per_method, "rung") == label)
            rung = rung.sort_values("rate")
            axis.plot(
                rung["rate"],
                rung["score"],
                marker=("o", "s", None)[index],
                markersize=4.0,
                linewidth=1.6,
                linestyle=("solid", "solid", (0, (4, 3)))[index],
                color=color if index < 2 else INK_MUTED,
                markerfacecolor=SURFACE if index == 0 else color,
                markeredgecolor=color,
                markeredgewidth=1.2,
                zorder=3 - index,
            )
        axis.set_title(METHOD_LABELS[method], fontsize=8.5)
        axis.set_xticks(rates)
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.55)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel(
        "Macro-F1" if metric == "f_score" else "Accuracy",
    )
    figure.supxlabel("Training missingness rate", fontsize=8, y=0.15)
    figure.suptitle(
        f"{DATASET_LABELS.get(dataset, dataset)}, {mechanism.upper()}",
        fontsize=9,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=4.0,
            linewidth=1.6,
            color=INK_MUTED,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED,
            label="Direct (restricted view)",
        ),
        Line2D(
            [],
            [],
            marker="s",
            markersize=4.0,
            linewidth=1.6,
            color=INK_MUTED,
            label="Restored (label-conditioned PVAE)",
        ),
        Line2D(
            [],
            [],
            linestyle=(0, (4, 3)),
            linewidth=1.6,
            color=INK_MUTED,
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
        bbox_to_anchor=(0.5, -0.01),
    )
    figure.subplots_adjust(top=0.84, bottom=0.28, left=0.10, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instance-metrics",
        type=Path,
        required=True,
    )
    parser.add_argument("--dataset", default="actg")
    parser.add_argument("--mechanism", default="mcar")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/missing_data/analysis_figures/ol_state.pdf"
        ),
    )
    arguments = parser.parse_args()

    frame, metric = collect(
        arguments.instance_metrics, arguments.dataset, arguments.mechanism
    )
    plot(
        frame,
        arguments.dataset,
        arguments.mechanism,
        metric,
        arguments.output,
    )
    print(frame.round(4).to_string(index=False))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
