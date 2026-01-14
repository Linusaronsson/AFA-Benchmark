import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import NamedTemporaryFile

import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

from afabench.common.config_classes import TrainingTimeCalculationConfig


def process_eval_artifact_sync(eval_artifact, training_times):
    eval_run = eval_artifact.logged_by()
    method_artifacts = [
        artifact
        for artifact in eval_run.used_artifacts()
        if artifact.type == "trained_method"
    ]
    assert len(method_artifacts) == 1
    method_artifact = method_artifacts[0]
    method_run = method_artifact.logged_by()
    runtime = method_run.summary["_wandb"]["runtime"]
    method_type = method_artifact.metadata["method_type"]
    # Add pretraining runtime if needed
    pretraining_artifacts = [
        artifact
        for artifact in method_run.used_artifacts()
        if artifact.type == "pretrained_model"
    ]
    for pretraining_artifact in pretraining_artifacts:
        pretraining_run = pretraining_artifact.logged_by()
        runtime += pretraining_run.summary["_wandb"]["runtime"]
    training_times[method_type].append(runtime)


def process_all_eval_artifacts(plotting_runs, training_times, max_workers=8):
    """Accepts a list of plotting runs and processes all their eval artifacts concurrently."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for plotting_run in plotting_runs:
            futures.extend(
                executor.submit(
                    process_eval_artifact_sync, eval_artifact, training_times
                )
                for eval_artifact in plotting_run.used_artifacts()
            )
        for future in as_completed(futures):
            future.result()


log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/misc",
    config_name="calculate_training_time",
)
def main(cfg: TrainingTimeCalculationConfig) -> None:
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        job_type="time_calculation",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        dir="extra/logs/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # In seconds
    training_times: defaultdict[str, list[int]] = defaultdict(list)

    plotting_runs = [
        wandb.Api().run(run_name) for run_name in cfg.plotting_run_names
    ]

    process_all_eval_artifacts(
        plotting_runs, training_times, max_workers=cfg.max_workers
    )

    # We also want to store the mean and std
    processed_training_times = {}
    for method_name, values in training_times.items():
        processed_training_times[method_name] = {}
        processed_training_times[method_name]["values"] = values
        # Calculate mean and std using numpy
        processed_training_times[method_name]["mean"] = float(np.mean(values))
        processed_training_times[method_name]["std"] = float(np.std(values))

    # Save results as wandb artifact
    training_time_artifact = wandb.Artifact(
        name=f"{'_'.join(cfg.plotting_run_names)}-training_time",
        type="training_time",
        metadata={},
    )
    with NamedTemporaryFile("w", delete=False) as f:
        metrics_save_path = Path(f.name)
        torch.save(processed_training_times, metrics_save_path)
    training_time_artifact.add_file(str(metrics_save_path), name="metrics.pt")
    run.log_artifact(
        training_time_artifact, aliases=cfg.output_artifact_aliases
    )
    run.finish()


if __name__ == "__main__":
    main()
