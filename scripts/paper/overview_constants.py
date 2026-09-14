"""
Generate the exact constants used by the paper's overview figure prototypes.

Kept separate from ``conceptual_constants.py`` so the constants already
consumed by ``figs/planning_effect.tex`` and ``figs/availability.tex`` are
untouched while the overview composition is still being chosen.

Two quantities drive every prototype and must not be conflated.

Availability is how much of the training data can supply an estimate: replaying
an acquisition out of a state with ``|S| = k`` acquired features needs all
``k + 1`` of them observed, so the lattice fades with depth at rate ``1 - p``.
Legality is whether an action exists at all under a sampled mask. The first is
Theorem 1, the second is Proposition 1, and the figures encode them with
opacity and with severed edges respectively.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import numpy.typing as npt

# The lattice is drawn at this rate because the exact availabilities are then
# 0.5, 0.25 and 0.125, which stay visible as literal opacities. No floor, no
# rescaling, so "edge opacity is availability" is true as drawn.
LATTICE_P = 0.5
LATTICE_DEPTH = 3
LATTICE_WIDTH = 4

# Example 1 as solved in the exact control study.
EXACT_D, EXACT_B = 10, 2
EXACT_N = 100_000

# UCI Cleveland, as loaded by HeartDiseaseDataset. Costs are uniform in this
# checkout: no extra/data/misc/feature_costs/heart_disease.csv is present, so
# load_feature_costs falls back to unit costs and the budget counts tests.
HEART_PATH = Path("extra/data/misc/heart_disease.csv")
HEART_D, HEART_B = 13, 7
HEART_LABELS = [
    "Age",
    "Sex",
    "Chest pain",
    "Resting BP",
    "Cholesterol",
    "Blood sugar",
    "Resting ECG",
    "Max heart rate",
    "Angina",
    "ST depression",
    "ST slope",
    "Vessels",
    "Thallium",
]
HEART_ROWS = 6
HEART_SHORT = [
    "Age",
    "Sex",
    "CP",
    "BP",
    "Chol",
    "FBS",
    "ECG",
    "HR",
    "Ang",
    "ST",
    "Slope",
    "Vess",
    "Thal",
]
HEART_SEED = 20260910


# One acquisition tree, drawn as the object the planner searches. With four
# features and depth three the tree of ordered acquisitions is complete at
# 4/12/24 nodes, so nothing is truncated away and the displayed legal depth
# is the true one.
FAN_PITCH = 1.30
FAN_SPAN = 2.55
# The displayed mask. Three of six features missing is the modal draw at
# p = 0.5; it is fixed here so the severed edges are the same in every panel.
FAN_MASK = (2, 3)
# Four further draws, for showing that each instance exposes a different
# subtree. Fixed rather than sampled so the panels are stable across rebuilds.
FAN_MASK_DRAWS = ((2, 3), (1,), (3, 4), (1, 2, 4), ())
# Four Cleveland tests, used where the concrete register needs a tree small
# enough to read. Indices are into HEART_LABELS.
HEART_PICK = (3, 4, 8, 13)
# The plan the evaluation problem supports, as a feature sequence.
FAN_EVAL_PATH = (1, 2, 3)


type State = tuple[int, ...]


class FanNode(TypedDict):
    level: int
    children: list[State]
    x: float
    y: float


type FanTree = dict[State, FanNode]


def _fan_tree() -> tuple[FanTree, list[State]]:
    """Build the displayed acquisition tree and lay it out bottom-up."""
    nodes: FanTree = {(): {"level": 0, "children": [], "x": 0.0, "y": 0.0}}
    frontier: list[State] = [()]
    for level in range(1, LATTICE_DEPTH + 1):
        nxt = []
        for state in frontier:
            free = [f for f in range(1, LATTICE_WIDTH + 1) if f not in state]
            for feature in free:
                child = (*state, feature)
                nodes[child] = {
                    "level": level,
                    "children": [],
                    "x": 0.0,
                    "y": 0.0,
                }
                nodes[state]["children"].append(child)
                nxt.append(child)
        frontier = nxt

    leaves = [s for s, n in nodes.items() if n["level"] == LATTICE_DEPTH]
    for index, leaf in enumerate(leaves):
        nodes[leaf]["y"] = FAN_SPAN / 2 - FAN_SPAN * index / (len(leaves) - 1)
    for level in range(LATTICE_DEPTH - 1, -1, -1):
        for node in nodes.values():
            if node["level"] == level:
                node["y"] = sum(nodes[c]["y"] for c in node["children"]) / len(
                    node["children"]
                )
    for node in nodes.values():
        node["x"] = node["level"] * FAN_PITCH
    return nodes, leaves


def _fan_edges(nodes: FanTree, mask: State | None = None) -> str:
    """
    Emit x1/y1/x2/y2/depth/feature/severed for every displayed edge.

    An edge is replayable only if every feature on the path leading to it, and
    the one it acquires, is present in the instance. Severing therefore
    cascades: nothing below an illegal acquisition is reachable either.
    """
    parts = []
    for _state, node in sorted(
        nodes.items(), key=lambda kv: (kv[1]["level"], kv[0])
    ):
        for child in node["children"]:
            feature = child[-1]
            active = FAN_MASK if mask is None else mask
            severed = 1 if any(f in active for f in child) else 0
            parts.append(
                f"{node['x']:.3f}/{node['y']:.3f}/{nodes[child]['x']:.3f}/"
                f"{nodes[child]['y']:.3f}/{nodes[child]['level']}/{feature}/{severed}"
            )
    return ",".join(parts)


def _fan_path(nodes: FanTree, features: State) -> str:
    """Emit the polyline for one acquisition plan."""
    points, state = [], ()
    for feature in features:
        points.append(f"({nodes[state]['x']:.3f},{nodes[state]['y']:.3f})")
        state = (*state, feature)
        if state not in nodes:
            return ""
    points.append(f"({nodes[state]['x']:.3f},{nodes[state]['y']:.3f})")
    return " ".join(points)


def _fan_dots(nodes: FanTree, features: State) -> str:
    """Emit a comma list so TikZ can place a marker per vertex."""
    dots, state = [], ()
    for feature in (None, *features):
        if feature is not None:
            state = (*state, feature)
        dots.append(f"{nodes[state]['x']:.3f}/{nodes[state]['y']:.3f}")
    return ",".join(dots)


def _fan_legal_path(nodes: FanTree) -> State:
    """Find the deepest legal plan, preferring low indices."""
    best: State = ()
    stack: list[State] = [()]
    while stack:
        state = stack.pop()
        if len(state) > len(best):
            best = state
        stack.extend(
            child
            for child in nodes[state]["children"]
            if child[-1] not in FAN_MASK
        )
    return best


def _fmt(value: float) -> str:
    """Render a probability without exponent notation LaTeX would have to fix."""
    if value >= 1e-3:
        return f"{value:.4g}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def _curve(exponent: int, x_span: float, y_span: float, decades: float) -> str:
    """Sample (1-p)^exponent onto a log axis, in figure centimetres."""
    points = []
    for step in range(96):
        p = step / 100.0
        decade = math.log10((1.0 - p) ** exponent)
        if decade < -decades:
            break
        points.append(
            f"({p / 0.95 * x_span:.3f},{decade / decades * y_span:.3f})"
        )
    return " ".join(points)


def _heart_sample() -> tuple[list[list[str]], npt.NDArray[np.bool_]]:
    """
    Read real patient records and draw one seeded MCAR mask over them.

    Masks that would remove every acquisition are resampled, matching the
    protocol described in the appendix.
    """
    with HEART_PATH.open(newline="") as handle:
        rows = list(csv.reader(handle))
    body = rows[1 : 1 + HEART_ROWS]
    values = [
        [
            f"{float(cell):g}" if cell not in {"", "?"} else "--"
            for cell in row[:HEART_D]
        ]
        for row in body
    ]
    rng = np.random.default_rng(HEART_SEED)
    observed = np.zeros((HEART_ROWS, HEART_D), dtype=bool)
    for index in range(HEART_ROWS):
        while True:
            draw = rng.random(HEART_D) >= LATTICE_P
            if draw.any():
                observed[index] = draw
                break
    return values, observed


def render() -> str:
    nodes, leaves = _fan_tree()
    legal = _fan_legal_path(nodes)
    values, observed = _heart_sample()
    cells = ",".join(
        f"{row + 1}/{column + 1}"
        for row, column in zip(*np.nonzero(observed), strict=True)
    )
    avail_budget = (1.0 - 0.7) ** EXACT_B
    avail_dim = (1.0 - 0.7) ** EXACT_D

    lines = [
        "% Generated by scripts/paper/overview_constants.py.",
        "% Regenerate rather than editing by hand.",
        rf"\newcommand{{\ovP}}{{{LATTICE_P:g}}}",
        rf"\newcommand{{\ovDepth}}{{{LATTICE_DEPTH}}}",
        rf"\newcommand{{\ovWidth}}{{{LATTICE_WIDTH}}}",
    ]
    # Availability at each lattice depth, used directly as an opacity.
    for depth, name in enumerate(["One", "Two", "Three"], start=1):
        lines.append(
            rf"\newcommand{{\ovAvail{name}}}{{{(1.0 - LATTICE_P) ** depth:g}}}"
        )
    lines += [
        rf"\newcommand{{\ovExactD}}{{{EXACT_D}}}",
        rf"\newcommand{{\ovExactB}}{{{EXACT_B}}}",
        r"\newcommand{\ovExactP}{0.7}",
        rf"\newcommand{{\ovExactN}}{{{EXACT_N:,}}}".replace(",", "{,}"),
        rf"\newcommand{{\ovAvailBudget}}{{{_fmt(avail_budget)}}}",
        rf"\newcommand{{\ovAvailDim}}{{{_fmt(avail_dim)}}}",
        rf"\newcommand{{\ovUsableBudget}}{{{avail_budget * EXACT_N:,.0f}}}".replace(
            ",", "{,}"
        ),
        rf"\newcommand{{\ovUsableDim}}{{{avail_dim * EXACT_N:.1f}}}",
        rf"\newcommand{{\ovAvailRatio}}{{{avail_budget / avail_dim:,.0f}}}".replace(
            ",", "{,}"
        ),
        # Headline numbers quoted in sections/experiments.tex.
        r"\newcommand{\ovRestoreN}{562}",
        r"\newcommand{\ovFilterRegret}{0.4408}",
        r"\newcommand{\ovAliasFloor}{0.25}",
        # Availability curves for the payoff panel, in figure centimetres.
        rf"\newcommand{{\ovCurveBudget}}{{{_curve(EXACT_B, 5.6, 2.5, 8.0)}}}",
        rf"\newcommand{{\ovCurveDim}}{{{_curve(EXACT_D, 5.6, 2.5, 8.0)}}}",
        rf"\newcommand{{\ovFanEdges}}{{{_fan_edges(nodes)}}}",
        *[
            rf"\newcommand{{\ovFanEdges{_word(i)}}}{{{_fan_edges(nodes, m)}}}"
            for i, m in enumerate(FAN_MASK_DRAWS, start=1)
        ],
        *[
            rf"\newcommand{{\ovFanMask{_word(i)}}}{{{','.join(str(f) for f in m)}}}"
            for i, m in enumerate(FAN_MASK_DRAWS, start=1)
        ],
        rf"\newcommand{{\ovFanDraws}}{{{len(FAN_MASK_DRAWS)}}}",
        rf"\newcommand{{\ovFanPitch}}{{{FAN_PITCH:g}}}",
        rf"\newcommand{{\ovFanSpan}}{{{FAN_SPAN:g}}}",
        rf"\newcommand{{\ovFanLeaves}}{{{len(leaves)}}}",
        rf"\newcommand{{\ovFanMask}}{{{','.join(str(f) for f in FAN_MASK)}}}",
        rf"\newcommand{{\ovFanPathEval}}{{{_fan_path(nodes, FAN_EVAL_PATH)}}}",
        rf"\newcommand{{\ovFanPathTrain}}{{{_fan_path(nodes, legal)}}}",
        rf"\newcommand{{\ovFanDotsEval}}{{{_fan_dots(nodes, FAN_EVAL_PATH)}}}",
        rf"\newcommand{{\ovFanDotsTrain}}{{{_fan_dots(nodes, legal)}}}",
        rf"\newcommand{{\ovFanEvalDepth}}{{{len(FAN_EVAL_PATH)}}}",
        rf"\newcommand{{\ovFanTrainDepth}}{{{len(legal)}}}",
        rf"\newcommand{{\ovFanEvalFirst}}{{{FAN_EVAL_PATH[0]}}}",
        rf"\newcommand{{\ovFanTrainFirst}}{{{legal[0] if legal else 0}}}",
        # UCI Cleveland heart disease, real names and real values.
        rf"\newcommand{{\ovHeartD}}{{{HEART_D}}}",
        rf"\newcommand{{\ovHeartB}}{{{HEART_B}}}",
        rf"\newcommand{{\ovHeartRows}}{{{HEART_ROWS}}}",
        rf"\newcommand{{\ovHeartObserved}}{{{cells}}}",
    ]
    # The ledger as drawable cells: only what was actually recorded.
    lines.append(
        r"\newcommand{\ovHeartCells}{"
        + ",".join(
            f"{row + 1}/{col + 1}/{values[row][col]}"
            for row, col in zip(*np.nonzero(observed), strict=True)
        )
        + "}"
    )
    lines.append(
        r"\newcommand{\ovHeartShort}{"
        + ",".join(f"{i}/{n}" for i, n in enumerate(HEART_SHORT, start=1))
        + "}"
    )
    for slot, index in enumerate(HEART_PICK, start=1):
        lines.append(
            rf"\newcommand{{\ovHeartPick{_word(slot)}}}{{{HEART_LABELS[index - 1]}}}"
        )
        lines.append(
            rf"\newcommand{{\ovHeartCode{_word(slot)}}}{{{HEART_SHORT[index - 1]}}}"
        )
    for index, label in enumerate(HEART_LABELS, start=1):
        lines.append(rf"\newcommand{{\ovHeartName{_word(index)}}}{{{label}}}")
    for row, record in enumerate(values, start=1):
        for column, value in enumerate(record, start=1):
            lines.append(
                rf"\newcommand{{\ovHeartVal{_word(row)}{_word(column)}}}{{{value}}}"
            )
    return "\n".join(lines) + "\n"


_WORDS = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
    "Twelve",
    "Thirteen",
]


def _word(index: int) -> str:
    """LaTeX control sequences cannot contain digits."""
    return _WORDS[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/overview_constants.tex"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
