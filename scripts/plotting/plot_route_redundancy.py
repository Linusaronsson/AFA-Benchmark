"""
Plot legal static references and paired route effects.

The figures deliberately avoid a cross-dataset regression: six heterogeneous
datasets cannot identify a relationship between route structure and restoration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib import font_manager
from omegaconf import OmegaConf

from afabench.plotting.config import PlottingDisplayConfig
from scripts.plotting.plot_eval_perf import calculate_figure_dimensions

_REFERENCE_LABELS = {
    "random_route_score_mean": "Random legal route",
    "static_reference_score": "Best sampled static route",
}
_EFFECT_LABELS = {
    "adaptive_gain": "Non-myopic - static",
    "nongreedy_gain": "Non-myopic - DIME",
}


def load_display(path: Path) -> PlottingDisplayConfig:
    raw = OmegaConf.load(path)
    merged = OmegaConf.merge(OmegaConf.structured(PlottingDisplayConfig), raw)
    config = cast("PlottingDisplayConfig", OmegaConf.to_object(merged))
    try:
        font_manager.findfont(
            config.plot_font_family, fallback_to_default=False
        )
    except ValueError:
        config.plot_font_family = "DejaVu Serif"
    return config


def _mean_se(
    frame: pd.DataFrame,
    groups: list[str],
    value: str,
) -> pd.DataFrame:
    output = (
        frame.groupby(groups, dropna=False)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output["se"] = output["std"].fillna(0.0) / np.sqrt(output["count"])
    return output


def reference_frame(routes: pd.DataFrame) -> pd.DataFrame:
    long = routes.melt(
        id_vars=["dataset", "instance", "budget"],
        value_vars=list(_REFERENCE_LABELS),
        var_name="reference",
        value_name="score",
    )
    long["reference"] = long["reference"].map(
        lambda value: _REFERENCE_LABELS[str(value)]
    )
    return _mean_se(long, ["dataset", "budget", "reference"], "score")


def planning_frame(planning: pd.DataFrame) -> pd.DataFrame:
    long = planning.melt(
        id_vars=["dataset", "method", "instance", "eval_hard_budget"],
        value_vars=list(_EFFECT_LABELS),
        var_name="effect",
        value_name="gain",
    )
    long["effect"] = long["effect"].map(
        lambda value: _EFFECT_LABELS[str(value)]
    )
    return _mean_se(
        long,
        ["dataset", "method", "eval_hard_budget", "effect"],
        "gain",
    )


def restoration_frame(missingness: pd.DataFrame) -> pd.DataFrame:
    return _mean_se(
        missingness,
        ["dataset", "method", "mechanism", "p", "eval_hard_budget"],
        "restoration_gain",
    )


def save(
    plot: p9.ggplot,
    folder: Path,
    name: str,
    width: float,
    height: float,
    formats: list[str],
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        plot.save(
            folder / f"{name}.{file_format}",
            width=width,
            height=height,
            verbose=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--planning", type=Path)
    parser.add_argument("--missingness", type=Path)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument(
        "--plotting-config",
        type=Path,
        default=Path("extra/conf/scripts/plotting/common/default.yaml"),
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "svg"])
    arguments = parser.parse_args()

    config = load_display(arguments.plotting_config)
    routes = pd.read_csv(arguments.routes)
    mapping = config.dataset_name_mapping
    theme = p9.theme_bw() + p9.theme(
        text=p9.element_text(family=config.plot_font_family, size=12)
    )
    reference = reference_frame(routes)
    reference["dataset"] = reference["dataset"].map(
        lambda value: mapping.get(value, value)
    )
    n_datasets = int(reference["dataset"].nunique())
    width, height = calculate_figure_dimensions(
        n_datasets,
        config.plot_width,
        ncol=min(4, n_datasets),
        subplot_height=2.8,
    )
    reference_plot = (
        p9.ggplot(
            reference,
            p9.aes("budget", "mean", color="reference", group="reference"),
        )
        + p9.geom_line()
        + p9.geom_point(size=2)
        + p9.geom_errorbar(p9.aes(ymin="mean-se", ymax="mean+se"), width=0.2)
        + p9.facet_wrap("dataset", ncol=min(4, n_datasets))
        + p9.scale_color_brewer(type="qual", palette=config.color_palette_name)
        + p9.labs(x="Acquisition budget", y="Primary metric", color="")
        + theme
    )
    save(
        reference_plot,
        arguments.output_folder,
        "route_static_reference",
        width,
        max(height, 3.2),
        arguments.formats,
    )

    if arguments.planning is not None:
        planning = planning_frame(pd.read_csv(arguments.planning))
        planning["dataset"] = planning["dataset"].map(
            lambda value: mapping.get(value, value)
        )
        planning_plot = (
            p9.ggplot(planning, p9.aes("method", "mean", color="effect"))
            + p9.geom_hline(yintercept=0.0, color="#888888", linetype="dashed")
            + p9.geom_point(position=p9.position_dodge(width=0.35), size=2.5)
            + p9.geom_errorbar(
                p9.aes(ymin="mean-se", ymax="mean+se"),
                position=p9.position_dodge(width=0.35),
                width=0.2,
            )
            + p9.facet_wrap("dataset", ncol=min(4, n_datasets))
            + p9.scale_color_brewer(
                type="qual", palette=config.color_palette_name
            )
            + p9.labs(x="", y="Paired primary-metric gain", color="")
            + theme
        )
        save(
            planning_plot,
            arguments.output_folder,
            "route_planning_effects",
            width,
            max(height, 3.2),
            arguments.formats,
        )

    if arguments.missingness is not None:
        restoration = restoration_frame(pd.read_csv(arguments.missingness))
        restoration["dataset"] = restoration["dataset"].map(
            lambda value: mapping.get(value, value)
        )
        restoration["method_mechanism"] = (
            restoration["method"].astype(str)
            + "/"
            + restoration["mechanism"].astype(str)
        )
        restoration_plot = (
            p9.ggplot(
                restoration,
                p9.aes(
                    "p",
                    "mean",
                    color="method",
                    linetype="mechanism",
                    group="method_mechanism",
                ),
            )
            + p9.geom_hline(yintercept=0.0, color="#888888", linetype="dashed")
            + p9.geom_line()
            + p9.geom_point(size=2)
            + p9.geom_errorbar(
                p9.aes(ymin="mean-se", ymax="mean+se"), width=0.015
            )
            + p9.facet_wrap("dataset", ncol=min(4, n_datasets))
            + p9.scale_color_brewer(
                type="qual", palette=config.color_palette_name
            )
            + p9.labs(
                x="Training missingness rate",
                y="Primary-metric gain from generative restoration",
                color="Method",
                linetype="Mechanism",
            )
            + theme
        )
        save(
            restoration_plot,
            arguments.output_folder,
            "route_restoration_effects",
            width,
            max(height, 3.2),
            arguments.formats,
        )
    print(f"wrote figures to {arguments.output_folder}")


if __name__ == "__main__":
    main()
