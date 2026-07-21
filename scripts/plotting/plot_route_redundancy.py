"""
Plot route structure against what the trained methods achieve (section 9).

Two views from route_redundancy.csv:
  band     per dataset, achieved accuracy of each strategy against the
           achievable static band (a_rand, a_best) - does the direct/naive
           policy find the good routes, does the generative model restore them?
  scatter  route_corr (and planning_gain) vs restoration_gap across datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd
import plotnine as p9
from matplotlib import font_manager
from omegaconf import OmegaConf

from afabench.plotting.config import PlottingDisplayConfig
from scripts.plotting.plot_eval_perf import calculate_figure_dimensions

BAND_ORDER = [
    "a_rand",
    "a_best",
    "complete",
    "restricted",
    "mean_fill",
    "pvae_label_conditioned",
]
BAND_LABEL = {
    "a_rand": "random subset",
    "a_best": "best static",
    "complete": "complete",
    "restricted": "restricted (direct)",
    "mean_fill": "mean (naive)",
    "pvae_label_conditioned": "PVAE (generative)",
}


def load_display(path: Path) -> PlottingDisplayConfig:
    raw = OmegaConf.load(path)
    merged = OmegaConf.merge(OmegaConf.structured(PlottingDisplayConfig), raw)
    cfg = cast("PlottingDisplayConfig", OmegaConf.to_object(merged))
    try:
        font_manager.findfont(cfg.plot_font_family, fallback_to_default=False)
    except ValueError:
        cfg.plot_font_family = "DejaVu Serif"
    return cfg


def band_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for key in BAND_ORDER:
            col = key if key in ("a_rand", "a_best") else f"acc_{key}"
            if col in r and bool(pd.notna(r[col])):
                rows.append(
                    {
                        "dataset": r["dataset_display"],
                        "kind": "reference"
                        if key in ("a_rand", "a_best")
                        else "achieved",
                        "route": BAND_LABEL[key],
                        "accuracy": float(r[col]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame["route"] = pd.Categorical(
        frame["route"], [BAND_LABEL[k] for k in BAND_ORDER], ordered=True
    )
    return frame


def save(
    plot: p9.ggplot,
    folder: Path,
    name: str,
    w: float,
    h: float,
    fmts: list[str],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for fmt in fmts:
        plot.save(folder / f"{name}.{fmt}", width=w, height=h, verbose=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, nargs="+", required=True)
    ap.add_argument("--output-folder", type=Path, required=True)
    ap.add_argument(
        "--plotting-config",
        type=Path,
        default=Path("extra/conf/scripts/plotting/common/default.yaml"),
    )
    ap.add_argument("--formats", nargs="+", default=["pdf", "svg"])
    a = ap.parse_args()

    cfg = load_display(a.plotting_config)
    df = pd.concat([pd.read_csv(p) for p in a.input], ignore_index=True)
    df["dataset_display"] = df["dataset"].map(
        lambda d: cfg.dataset_name_mapping.get(d, d)
    )
    theme = p9.theme_bw() + p9.theme(
        text=p9.element_text(family=cfg.plot_font_family, size=12)
    )

    # band: achieved strategies vs achievable static band, per dataset
    bframe = band_frame(df)
    n_datasets = int(df["dataset"].nunique())
    ncol = min(4, n_datasets)
    w, h = calculate_figure_dimensions(
        n_datasets, cfg.plot_width, ncol=ncol, subplot_height=3.0
    )
    band = (
        p9.ggplot(bframe, p9.aes("route", "accuracy", color="kind"))
        + p9.geom_point(size=3)
        + p9.facet_wrap("dataset", ncol=ncol)
        + p9.scale_color_brewer(type="qual", palette=cfg.color_palette_name)
        + p9.labs(
            x="",
            y="Evaluation accuracy",
            color="",
            title="Routes the methods take vs the achievable band",
        )
        + theme
        + p9.theme(axis_text_x=p9.element_text(rotation=45, ha="right"))
    )
    save(band, a.output_folder, "route_band", w, max(h, 3.5), a.formats)

    # scatter: route structure vs restoration gain (PVAE - restricted), one
    # point per dataset. planning/"harmful" datasets sit upper-left.
    for xcol in ("route_corr", "planning_gain"):
        sub = df.dropna(subset=[xcol, "restoration_gain"])
        if sub.empty:
            continue
        scatter = (
            p9.ggplot(sub, p9.aes(xcol, "restoration_gain"))
            + p9.geom_hline(yintercept=0.0, color="#999999", linetype="dashed")
            + p9.geom_point(p9.aes(color="dataset_display"), size=3)
            + p9.geom_text(
                p9.aes(label="dataset_display"), nudge_y=0.004, size=9
            )
            + p9.scale_color_brewer(
                type="qual", palette=cfg.color_palette_name
            )
            + p9.labs(
                x=xcol,
                y="restoration gain (PVAE - restricted)",
                color="",
            )
            + theme
        )
        save(
            scatter,
            a.output_folder,
            f"scatter_{xcol}",
            cfg.plot_width,
            4.0,
            a.formats,
        )
    print(f"wrote figures to {a.output_folder}")


if __name__ == "__main__":
    main()
