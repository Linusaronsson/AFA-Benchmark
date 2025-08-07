import re
import logging
from pathlib import Path
import shutil

import hydra
import torch
import wandb

from common.config_classes import PlotDownloadConfig
from omegaconf import OmegaConf


def process_figure_artifact(figure_artifact, files):
    artifact_dir = Path(figure_artifact.download())
    figure_file_path = [f for f in artifact_dir.iterdir()][0]
    files.append(figure_file_path)


def process_plot_artifact(cfg: PlotDownloadConfig, plot_run):
    files = []
    figure_artifacts = [
        artifact
        for artifact in plot_run.logged_artifacts()
        if artifact.type == "publication_figure"
    ]
    for figure_artifact in figure_artifacts:
        log.debug(f"Processing {figure_artifact.name}")
        for dataset_name, budgets, metric in zip(
            cfg.datasets, cfg.budgets, cfg.metrics, strict=False
        ):
            if budgets.strip() == ".":
                budget = None
                if is_match(
                    figure_artifact.name,
                    dataset_name,
                    budget,
                    metric,
                    cfg.file_type,
                ):
                    log.info(f"MATCH: {figure_artifact.name}")
                    process_figure_artifact(figure_artifact, files)
            else:
                for budget in map(int, budgets.split(" ")):
                    if is_match(
                        figure_artifact.name,
                        dataset_name,
                        budget,
                        metric,
                        cfg.file_type,
                    ):
                        log.info(f"MATCH: {figure_artifact.name}")
                        process_figure_artifact(figure_artifact, files)
    return files


def is_match(
    artifact_name: str, dataset: str, budget: int | None, metric: str, file_type: str
):
    if budget is None:
        budget_pattern = r"budget\d+"
    else:
        budget_pattern = f"budget{budget}"
    pattern = rf"^figure-{re.escape(dataset)}_.*_{budget_pattern}_{re.escape(metric)}-{file_type}:v\d+$"
    return re.match(pattern, artifact_name) is not None


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/misc",
    config_name="download_plot_results",
)
def main(cfg: PlotDownloadConfig) -> None:
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    # Only use the API, do not create a run
    plotting_run = wandb.Api().run(cfg.plotting_run_name)

    files = process_plot_artifact(cfg, plotting_run)

    output_path = Path(cfg.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        dest_path = output_path / Path(file_path).name
        shutil.move(str(file_path), str(dest_path))
        log.info(f"Moved {file_path} to {dest_path}")


if __name__ == "__main__":
    main()
