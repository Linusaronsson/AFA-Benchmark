import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.utils import set_seed
from afabench.afa_oracle import create_aaco_method
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import AACOTrainConfig
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config

logger = logging.getLogger(__name__)


def run(cfg: AACOTrainConfig) -> None:
    logger.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    dataset_bundle_path = (
        cfg.train_dataset_bundle_path or cfg.dataset_artifact_name
    )
    assert dataset_bundle_path is not None, (
        "Expected train_dataset_bundle_path or dataset_artifact_name."
    )

    # Load dataset bundle
    dataset_obj, dataset_manifest = load_bundle(Path(dataset_bundle_path))
    dataset_name = (
        dataset_manifest["class_name"].replace("Dataset", "").lower()
    )
    split = dataset_manifest["metadata"].get("split_idx", None)

    logger.info(f"Dataset: {dataset_manifest['class_name']}, Split: {split}")
    logger.info(f"Training samples: {len(dataset_obj)}")

    # Get training data (flatten if needed - AACO works on flat features)
    X_train, y_train = dataset_obj.get_all_data()
    feature_shape = dataset_obj.feature_shape

    if len(feature_shape) > 1:
        X_train = X_train.view(X_train.shape[0], -1)
        logger.info(
            f"Flattened features from {feature_shape} to {X_train.shape[1]}"
        )

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    logger.debug(
        "X_train shape %s, y_train shape %s",
        X_train.shape,
        y_train.shape,
    )
    logger.debug(f"Feature shape: {feature_shape}")

    # Determine soft budget parameter
    soft_budget_param = (
        cfg.soft_budget_param
        if cfg.soft_budget_param is not None
        else cfg.aco.acquisition_cost
    )
    force_acquisition = cfg.hard_budget is not None

    assert cfg.classifier_bundle_path is not None, (
        "classifier_bundle_path must be provided. Train a classifier first."
    )
    classifier_bundle_path = Path(cfg.classifier_bundle_path)

    assert classifier_bundle_path.exists(), (
        f"Classifier bundle not found at: {classifier_bundle_path}"
    )

    # Determine selection space from unmasker
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
    selection_size = unmasker.get_n_selections(
        feature_shape=dataset_obj.feature_shape
    )
    selection_costs = unmasker.get_selection_costs(
        feature_costs=dataset_obj.get_feature_acquisition_costs()
    ).to(device)
    if cfg.unmasker.kwargs is None:
        unmasker_kwargs: dict[str, Any] = {}
    elif OmegaConf.is_config(cfg.unmasker.kwargs):
        unmasker_kwargs = cast(
            "dict[str, Any]",
            OmegaConf.to_container(cfg.unmasker.kwargs, resolve=True),
        )
    else:
        unmasker_kwargs = dict(cfg.unmasker.kwargs)

    # Create AACO method (classifier will be loaded in __post_init__)
    aaco_method = create_aaco_method(
        dataset_name=dataset_name,
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=soft_budget_param,
        hide_val=cfg.aco.hide_val,
        force_acquisition=force_acquisition,
        selection_size=selection_size,
        unmasker_class_name=cfg.unmasker.class_name,
        unmasker_kwargs=unmasker_kwargs,
        selection_costs=selection_costs,
        classifier_bundle_path=classifier_bundle_path,
        device=device,
    )

    # Fit oracle on training data
    logger.info("Fitting AACO oracle on training data...")
    aaco_method.aaco_oracle.fit(X_train, y_train)
    logger.info(
        "AACO oracle fitted with classifier from %s",
        classifier_bundle_path,
    )

    # Save
    save_bundle(
        obj=aaco_method,
        path=Path(cfg.save_path),
        metadata={
            "dataset_artifact": str(dataset_bundle_path),
            "dataset_name": dataset_name,
            "split_idx": split,
            "seed": cfg.seed,
            "soft_budget_param": soft_budget_param,
            "hard_budget": cfg.hard_budget,
            "force_acquisition": force_acquisition,
            "selection_size": selection_size,
            "k_neighbors": cfg.aco.k_neighbors,
            "hide_val": cfg.aco.hide_val,
            "classifier_bundle_path": str(classifier_bundle_path),
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
        },
    )
    logger.info(f"Saved AACO method to: {cfg.save_path}")


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
