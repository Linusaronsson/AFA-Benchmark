import torch
import wandb
import hydra
import logging

from pathlib import Path
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory

from afabench import SAVE_PATH
from afabench.afa_oracle import create_aaco_method
from afabench.common.config_classes import AACOTrainConfig

from afabench.common.utils import (
    load_dataset,
    set_seed,
    save_artifact,
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

    # Optional wandb logging
    # run = wandb.init(
    #     config=OmegaConf.to_container(cfg, resolve=True),
    #     job_type="training",
    #     tags=["aaco"],
    #     dir="extra/wandb",
    # )

    # log.info(f"W&B run initialized: {run.name} ({run.id})")
    # log.info(f"W&B run URL: {run.url}")

    # Load dataset from filesystem
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

    # Create and train AACO method
    aaco_method = create_aaco_method(
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cost,
        hide_val=cfg.aco.hide_val,
        dataset_name=dataset_type.lower(),
        split=split,
        device=device,
    )

    log.info("Fitting AACO oracle on training data...")
    X_train = train_dataset.features.to(device)
    y_train = train_dataset.labels.to(device)
    aaco_method.aaco_oracle.fit(X_train, y_train)
    log.info("AACO method fitted and ready for evaluation")

    # Save to temporary directory then move to artifacts
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        aaco_method.save(temp_path)

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
            "method_type": "aaco",
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

        log.info(f"AACO method saved to: {artifact_dir}")

    # if run:
    #     run.finish()


if __name__ == "__main__":
    main()
