import hydra
import yaml
import torch
import wandb
import logging
import tempfile
import numpy as np
import matplotlib as mpl

from typing import Any
from pathlib import Path
from omegaconf import OmegaConf
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from common.config_classes import PlotConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

log = logging.getLogger(__name__)

# METHOD_STYLES = {
#     "aaco": {"color": "#0173B2", "linestyle": "-", "name": "AACO"},
#     "cae": {"color": "#DE8F05", "linestyle": "--", "name": "CAE-S"},
#     "permutation": {"color": "#029E73", "linestyle": "-.", "name": "PT-S"},
#     "covert2023": {"color": "#CC78BC", "linestyle": ":", "name": "GDFS-DG"},
#     "gadgil2023": {"color": "#CA3542", "linestyle": "-", "name": "DIME-DG"},
#     "ma2018": {"color": "#CA3542", "linestyle": "-", "name": "EDDI-GG"},
#     "shim2018": {"color": "#FB4F14", "linestyle": "--", "name": "JAFA-MFRL"},
#     "kachuee2019": {"color": "#56B4E9", "linestyle": "-.", "name": "OL-MFRL"},
#     "zannone2019": {"color": "#949494", "linestyle": ":", "name": "ODIN-MBRL"},
# }

METHOD_STYLES = {
    "aaco": {
        "color": "#56B4E9",
        "linestyle": "-",
        "marker": "o",
        "name": "AACO",
    },
    "cae": {
        "color": "#E69F00",
        "linestyle": "--",
        "marker": "s",
        "name": "CAE-S",
    },
    "permutation": {
        "color": "#009E73",
        "linestyle": "-.",
        "marker": "^",
        "name": "PT-S",
    },
    "covert2023": {
        "color": "#CC79A7",
        "linestyle": ":",
        "marker": "D",
        "name": "GDFS-DG",
    },
    "gadgil2023": {
        "color": "#D55E00",
        "linestyle": (0, (5, 10)),
        "marker": "v",
        "name": "DIME-DG",
    },
    "ma2018": {
        "color": "#0072B2",
        "linestyle": (0, (3, 5, 1, 5, 1, 5)),
        "marker": "P",
        "name": "EDDI-GG",
    },
    "shim2018": {
        "color": "#F0E442",
        "linestyle": (0, (1, 1)),
        "marker": "X",
        "name": "JAFA-MFRL",
    },
    "kachuee2019": {
        "color": "#000000",
        "linestyle": (0, (10, 3)),
        "marker": "x",
        "name": "OL-MFRL",
    },
    "zannone2019": {
        "color": "#999999",
        "linestyle": (0, (2, 2)),
        "marker": "d",
        "name": "ODIN-MBRL",
    },
    "zannone2019-model-free": {
        "color": "#A1D76A",
        "linestyle": (0, (2, 2)),
        "marker": "d",
        "name": "ODIN-MFRL",
    },
}

plt.style.use(["seaborn-v0_8-whitegrid"])
mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "lines.markersize": 4,
    }
)

mpl.use("pgf")
mpl.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)


def create_figure(x, grouped_metrics, metric_cfg):
    fig, ax = plt.subplots(figsize=(6, 4))

    for method_type, metrics_list in grouped_metrics.items():
        data = torch.stack([m[metric_cfg.key] for m in metrics_list])
        mean = data.mean(dim=0)
        std = data.std(dim=0)

        style = METHOD_STYLES.get(
            method_type,
            {
                "color": "#000000",
                "linestyle": "-",
                "marker": "o",
                "name": method_type.replace("_", " ").title(),
            },
        )

        ax.plot(
            x,
            mean,
            label=style["name"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            marker=style["marker"],
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.5,
        )

        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=style["color"])

    ax.set_xlabel("Number of Features Selected")
    ax.set_ylabel(metric_cfg.description)
    ax.legend(frameon=True, fancybox=False, edgecolor="black", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if metric_cfg.ylim is not None:
        ax.set_ylim(*metric_cfg.ylim)

    plt.tight_layout()
    return fig


def save_figures(fig, filename_base):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        svg_path = tmp_path / f"{filename_base}.svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")

        pdf_path = tmp_path / f"{filename_base}.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")

        png_path = tmp_path / f"{filename_base}.png"
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

        for format_ext, file_path in [
            ("svg", svg_path),
            ("pdf", pdf_path),
            ("png", png_path),
        ]:
            artifact = wandb.Artifact(
                name=f"figure-{filename_base}-{format_ext}", type="publication_figure"
            )
            artifact.add_file(str(file_path))
            wandb.log_artifact(artifact)

        wandb.log({f"{filename_base}_publication": wandb.Image(fig)})


# def load_eval_results(config_path: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
#     """Load eval results from wandb artifacts.
#
#     The results are returned as a list of tuples, where each tuple contains:
#     1. Info describing the evaluation (dataset type, method type, split, seed, classifier type)
#     2. The actual metrics dictionary.
#     Skips missing artifacts.
#     """
#     # Read artifact_names from the config file
#
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#     artifact_names = config.get("eval_artifact_names", [])
#
#     eval_results = []
#     wandb_api = wandb.Api()
#     for artifact_name in artifact_names:
#         # Check if the artifact exists before using it
#         try:
#             eval_artifact = wandb_api.artifact(artifact_name, type="eval_results")
#             # Optionally, check if artifact is actually logged
#             if eval_artifact is None:
#                 log.warning(f"Artifact {artifact_name} does not exist. Skipping.")
#                 continue
#         except Exception as e:
#             log.warning(f"Could not find artifact {artifact_name}: {e}. Skipping.")
#             continue
#         log.info(f"Downloading evaluation artifact {artifact_name}")
#         eval_artifact_dir = Path(eval_artifact.download())
#         metrics = torch.load(
#             eval_artifact_dir / "metrics.pt", map_location=torch.device("cpu")
#         )
#         info = {
#             "dataset_type": eval_artifact.metadata["dataset_type"],
#             "method_type": eval_artifact.metadata["method_type"],
#             "budget": eval_artifact.metadata["budget"],
#             "seed": eval_artifact.metadata["seed"],
#             "classifier_type": eval_artifact.metadata["classifier_type"],
#         }
#         eval_results.append((info, metrics))
#     return eval_results


def load_single_artifact(
    api: wandb.Api, artifact_name: str
) -> tuple[dict[str, Any], Any] | None:
    try:
        eval_artifact = api.artifact(artifact_name, type="eval_results")
        artifact_dir = Path(eval_artifact.download())
        metrics = torch.load(artifact_dir / "metrics.pt", map_location="cpu")
        # HACK: we have a model-based and model-free zannone2019 variant
        # we differentiate between them by looking at their aliases
        if any(alias.endswith("model-free") for alias in eval_artifact.aliases):
            method_type = "zannone2019-MFRL"
        else:
            method_type = eval_artifact.metadata.get("method_type", None)
        info = {
            "dataset_type": eval_artifact.metadata.get("dataset_type", None),
            "method_type": method_type,
            "budget": eval_artifact.metadata.get("budget", None),
            "seed": eval_artifact.metadata.get("seed", None),
            "classifier_type": eval_artifact.metadata.get("classifier_type", None),
        }
        return (info, metrics)
    except Exception as e:
        log.warning(f"Skipping artifact {artifact_name}: {e}")
        return None


def load_eval_results(
    config_path: Path, max_workers: int = 8
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    artifact_names = config.get("eval_artifact_names", [])

    api = wandb.Api()
    eval_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(load_single_artifact, api, name): name
            for name in artifact_names
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                eval_results.append(result)
    return eval_results


@hydra.main(version_base=None, config_path="../../conf/plot", config_name="config")
def main(cfg: PlotConfig):
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="plotting",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
    )

    eval_results = load_eval_results(
        Path(cfg.eval_artifact_config_path), max_workers=cfg.max_workers
    )
    log.info("All evaluation result artifacts loaded.")

    dataset_types = set(info["dataset_type"] for (info, _) in eval_results)

    for dataset_type in dataset_types:
        log.info(f"Plotting results for dataset type: {dataset_type}")
        classifier_types = set(
            info["classifier_type"]
            for (info, _) in eval_results
            if info["dataset_type"] == dataset_type
        )
        for classifier_type in classifier_types:
            log.info(f"  Plotting results for classifier type: {classifier_type}")
            budgets = set(
                info["budget"]
                for (info, _) in eval_results
                if info["dataset_type"] == dataset_type
                and info["classifier_type"] == classifier_type
            )
            for budget in budgets:
                log.info(f"    Plotting results for budget: {budget}")
                # x-axis will be [1, budget]
                x = np.arange(1, budget + 1)
                # Organize by method_type
                grouped_metrics: dict[str, list[dict[str, torch.Tensor]]] = defaultdict(
                    list
                )

                for info, metrics in eval_results:
                    if (
                        info["dataset_type"] == dataset_type
                        and info["classifier_type"] == classifier_type
                        and info["budget"] == budget
                    ):
                        method_type = info["method_type"]
                        grouped_metrics[method_type].append(metrics)

                if not grouped_metrics:
                    continue

                for metric_cfg in cfg.metric_keys_and_descriptions:
                    fig = create_figure(x, grouped_metrics, metric_cfg)
                    filename_base = f"{dataset_type}_{classifier_type}_budget{budget}_{
                        metric_cfg.key
                    }"
                    save_figures(fig, filename_base)

                    plt.close(fig)


if __name__ == "__main__":
    main()
