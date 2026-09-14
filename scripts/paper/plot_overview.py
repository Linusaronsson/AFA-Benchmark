"""
Overview figure: what incomplete training data makes of an acquisition value.

Two candidate acquisitions at the same state span a plane. Because both axes
carry the same quantity on the same scale, the diagonal is the decision
boundary: an estimate on the far side of it produces a policy that acquires the
other feature. Each approach is drawn as the distribution of its estimate over
repeated training sets, so bias and support are visible as position and spread.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from scipy.stats import gaussian_kde

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    INK,
    INK_MUTED,
    TEXT_WIDTH_IN,
    apply_paper_style,
)
from scripts.paper.overview_backup import ARMS, draws

if TYPE_CHECKING:
    from matplotlib.axes import Axes

COLORS = {
    "mask_agnostic": "#B5479B",
    "mask_local": "#4C92D9",
    "generative": "#0E7A4A",
}
LABELS = {
    "mask_agnostic": "Aliasing",
    "mask_local": "Filtering",
    "generative": "Generative restoration",
}
D, P, SEED = 6, 0.7, 0
PANELS = (100, 10_000)
# Square limits, so the decision boundary is a true 45 degree line.
LO, HI = 0.10, 1.06
STAR = (1.0, 0.75)
BANDWIDTH = 2.0
CACHE = Path("extra/output/paper/experiments/results/overview_draws.npz")


def _load(reps: int) -> dict[tuple[str, int], npt.NDArray[np.float64]]:
    if CACHE.exists():
        stored = np.load(CACHE)
        wanted = [f"{a}_{n}" for a in ARMS for n in PANELS]
        if int(stored["reps"]) == reps and all(k in stored for k in wanted):
            return {(a, n): stored[f"{a}_{n}"] for a in ARMS for n in PANELS}
    out: dict[tuple[str, int], npt.NDArray[np.float64]] = {}
    for n in PANELS:
        for arm, pairs in draws(D, P, n, reps, SEED).items():
            out[(arm, n)] = pairs
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE,
        allow_pickle=False,
        reps=reps,
        **{f"{a}_{n}": out[(a, n)] for a, n in out},
    )
    return out


def _density(axis: Axes, pairs: npt.NDArray[np.float64], color: str) -> None:
    """
    Draw one approach's sampling distribution and its centre.

    The estimator has a degenerate branch: with no training instance to support
    an estimate it falls back to a fixed value, so a large share of the draws
    can sit on one point. The kernel renders that pile as a small lobe rather
    than a spike, which is where the mass is but is smoother than the truth;
    drawing it as a separate point mass instead splits one approach into two
    disconnected marks and reads worse than the smoothing costs.
    """
    if pairs.std(axis=0).min() > 1e-6:
        grid_x, grid_y = np.mgrid[LO:HI:260j, LO:HI:260j]
        kernel = gaussian_kde(pairs.T)
        # With very little support an estimate takes only a handful of discrete
        # values, and the default bandwidth resolves them as separate ridges.
        # Widening it shows the shape of the distribution rather than the
        # lattice the estimator happens to land on.
        kernel.set_bandwidth(cast("float", kernel.factor) * BANDWIDTH)
        density = kernel(np.vstack([grid_x.ravel(), grid_y.ravel()]))
        density = (density / density.max()).reshape(grid_x.shape)
        axis.contourf(
            grid_x,
            grid_y,
            density,
            levels=[0.12, 0.45, 1.0],
            colors=[color] * 2,
            alpha=0.24,
            zorder=3,
        )
        axis.contour(
            grid_x,
            grid_y,
            density,
            levels=[0.12],
            colors=[color],
            linewidths=0.9,
            zorder=4,
        )
        peak = np.unravel_index(density.argmax(), density.shape)
        centre = (grid_x[peak], grid_y[peak])
    else:
        centre = tuple(pairs.mean(axis=0))
    # Always inside its own contour, so a narrow distribution still reads.
    axis.plot(
        *centre,
        marker="o",
        markersize=4.2,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=8,
    )


def _panel(
    axis: Axes,
    data: dict[tuple[str, int], npt.NDArray[np.float64]],
    n: int,
    first: bool,
) -> None:
    axis.plot([LO, HI], [LO, HI], color=INK_MUTED, linewidth=0.7, zorder=2)
    for arm in ARMS:
        _density(axis, data[(arm, n)], COLORS[arm])
    axis.plot(
        *STAR,
        marker="*",
        markersize=15,
        markerfacecolor="none",
        markeredgecolor=INK,
        markeredgewidth=1.0,
        zorder=8,
    )

    axis.set_xlim(LO, HI)
    axis.set_ylim(LO, HI)
    axis.set_aspect("equal")
    axis.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_title(
        f"$n=10^{{{round(np.log10(n))}}}$", fontsize=8.5, color=INK, pad=3
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    if first:
        axis.set_ylabel(r"$\widehat{Q}(s,a_2)$", fontsize=8.5, labelpad=1)
        # The boundary, named on the boundary.
        axis.text(
            0.05,
            0.95,
            "Prefers $a_2$",
            transform=axis.transAxes,
            fontsize=8,
            color=INK_MUTED,
            ha="left",
            va="top",
        )
        axis.text(
            0.95,
            0.05,
            "Prefers $a_1$",
            transform=axis.transAxes,
            fontsize=8,
            color=INK_MUTED,
            ha="right",
            va="bottom",
        )
    else:
        axis.set_yticklabels([])


def build(reps: int, out: Path) -> None:
    apply_paper_style()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
        }
    )
    data = _load(reps)

    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 3.02))
    axes = [
        fig.add_axes((0.070, 0.235, 0.400, 0.715)),
        fig.add_axes((0.545, 0.235, 0.400, 0.715)),
    ]
    for axis, n in zip(axes, PANELS, strict=True):
        _panel(axis, data, n, first=n == PANELS[0])
    fig.text(
        0.51,
        0.108,
        r"$\widehat{Q}(s,a_1)$",
        ha="center",
        fontsize=8.5,
        color=INK_MUTED,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5,
            color=COLORS[arm],
            label=LABELS[arm],
        )
        for arm in ARMS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markersize=8,
            color=INK,
            label=r"Optimum $Q^{\star}$",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
        handletextpad=0.35,
        columnspacing=1.5,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400)
    fig.savefig(out.with_suffix(".png"), dpi=400)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=1200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/paper/experiments/results/overview.pdf"),
    )
    args = parser.parse_args()
    build(args.reps, args.output)


if __name__ == "__main__":
    main()
