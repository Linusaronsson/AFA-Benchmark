import gc
import torch
import wandb
import hydra
import logging

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory

from afabench import SAVE_PATH
from afabench.common.utils import load_dataset, set_seed, save_artifact
from afabench.common.afa_methods import RandomDummyAFAMethod

from afabench.common.config_classes import (
    RandomDummyTrainConfig,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/randomdummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
        job_type="training",
        tags=["randomdummy"],
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset artifact
    train_dataset, _, _, dataset_metadata = load_dataset(
        cfg.dataset_artifact_name
    )

    dataset_type = dataset_metadata["dataset_type"]
    split = dataset_metadata["split_idx"]
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    log.info(f"Dataset: {dataset_type}, Split: {split}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    # Get cost parameter
    cost = (
        cfg.cost_param
        if cfg.cost_param is not None
        else cfg.aco.acquisition_cost
    )

    # Get number of classes from the dataset
    n_classes = train_dataset.labels.shape[-1]

    afa_method = RandomDummyAFAMethod(
        device=torch.device("cpu"),
        n_classes=n_classes,
        prob_select_0=cfg.cost_param,
    )

    # Save the method to a temporary directory and load it again to ensure it is saved correctly
    with TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        afa_method.save(temp_path)

        # Build artifact identifier with optional experiment_id suffix
        artifact_identifier = f"{dataset_type.lower()}_split_{
            split
        }_costparam_{cost}_seed_{cfg.seed}"
        if hasattr(cfg, "experiment_id") and cfg.experiment_id:
            artifact_identifier = f"{artifact_identifier}_{cfg.experiment_id}"

        # Direct path, no get_artifact_path wrapper
        artifact_dir = SAVE_PATH / artifact_identifier

        # Prepare metadata
        metadata = {
            "method_type": "RandomDummy",
            "dataset_type": dataset_type,
            "dataset_artifact_name": cfg.dataset_artifact_name,
            "budget": None,
            "seed": cfg.seed,
            "cost_param": cost,
            "split_idx": split,
        }

        # Save artifact to filesystem
        save_artifact(
            artifact_dir=artifact_dir,
            files={f.name: f for f in temp_path.iterdir() if f.is_file()},
            metadata=metadata,
        )

        log.info(f"RandomDummy method saved to {artifact_dir}")

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
