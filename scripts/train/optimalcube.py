import gc
import torch
import wandb
import hydra
import logging

from pathlib import Path
from typing import Any, cast
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory

from afabench.common.afa_methods import OptimalCubeAFAMethod
from afabench.common.utils import load_dataset_artifact, set_seed

from afabench.common.config_classes import (
    OptimalCubeTrainConfig,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/optimalcube",
    config_name="config",
)
def main(cfg: OptimalCubeTrainConfig):
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
    train_dataset, val_dataset, _, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name
    )

    # Get number of classes from the dataset
    n_classes = train_dataset.labels.shape[-1]

    afa_method = OptimalCubeAFAMethod(
        device=torch.device("cpu"), n_classes=n_classes
    )
    # Save the method to a temporary directory and load it again to ensure it is saved correctly
    with TemporaryDirectory(delete=False) as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        afa_method.save(tmp_path)

        # Save the model as a WandB artifact
        # Save the name of the afa method class as metadata
        afa_method_artifact = wandb.Artifact(
            name=f"train_optimalcube-{
                cfg.dataset_artifact_name.split(':')[0]
            }-budget_{cfg.hard_budget}-seed_{cfg.seed}",
            type="trained_method",
            metadata={
                "method_type": "optimalcube",
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "dataset_type": dataset_metadata["dataset_type"],
                "budget": cfg.hard_budget,
                "seed": cfg.seed,
            },
        )

        afa_method_artifact.add_dir(str(tmp_path))
        run.log_artifact(
            afa_method_artifact, aliases=cfg.output_artifact_aliases
        )

    run.finish()

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
