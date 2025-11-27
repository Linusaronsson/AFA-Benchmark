import logging
from pathlib import Path

import hydra
import torch

from afabench.afa_oracle import create_aaco_method
from afabench.common.config_classes import AACOTrainConfig
from afabench.common.utils import (
    load_dataset_artifact,
    save_method_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/train/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    # run = wandb.init(
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     job_type="training",
    #     tags=["aaco"],
    #     dir="extra/wandb",
    # )

    # log.info(f"W&B run initialized: {run.name} ({run.id})")
    # log.info(f"W&B run URL: {run.url}")

    # Load dataset from filesystem
    train_dataset, dataset_metadata = load_dataset_artifact(
        cfg.dataset_artifact_name,
        split="train",
    )

    dataset_type = dataset_metadata["dataset_type"]
    split = dataset_metadata["split_idx"]

    log.info(f"Dataset: {dataset_type}, Split: {split}")
    log.info(f"Training samples: {len(train_dataset)}")

    # Get cost parameter
    cost = cfg.cost_param or cfg.aco.acquisition_cost

    # Create and train AACO method
    aaco_method = create_aaco_method(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cost,
        hide_val=cfg.aco.hide_val,
        dataset_name=dataset_type.lower(),
        split=split,
        device=device,
    )
    if cfg.hard_budget:
        aaco_method.set_hard_budget(cfg.hard_budget)

    log.info("Fitting AACO oracle on training data...")
    X_train = train_dataset.features.to(device)
    y_train = train_dataset.labels.to(device)
    aaco_method.aaco_oracle.fit(X_train, y_train)
    log.info("AACO method fitted and ready for evaluation")

    # One clean save call
    save_method_artifact(
        method=aaco_method,
        save_path=Path(cfg.save_path),
        metadata={
            "method_type": aaco_method.__class__.__name__,
            "dataset_type": dataset_type,
            "dataset_artifact_path": cfg.dataset_artifact_name,
            "split_idx": split,
            "seed": cfg.seed,
            "cost_param": cost,
            "hard_budget": cfg.hard_budget,
            "initializer_type": cfg.initializer_type,
            "unmasker_type": cfg.unmasker_type,
        },
    )
    log.info(f"Saved AACO method to: {cfg.save_path}")

    # if run:
    #     run.finish()


if __name__ == "__main__":
    main()
