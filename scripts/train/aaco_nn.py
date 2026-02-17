"""
AACO+NN training script.

Trains a neural network policy via behavioral cloning.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch

from afabench.afa_oracle import (
    AACOPolicyNetwork,
    create_aaco_nn_method,
    create_rollout_data_loaders,
    generate_aaco_rollouts,
    train_policy_network,
)
from afabench.afa_oracle.afa_methods import AACOAFAMethod
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import set_seed

if TYPE_CHECKING:
    from afabench.common.config_classes import AACONNTrainConfig

logger = logging.getLogger(__name__)


def _configure_smoke_test(cfg: AACONNTrainConfig) -> None:
    if cfg.smoke_test:
        logger.info("Smoke test mode: reducing training samples and epochs")
        cfg.max_epochs = 2
        cfg.batch_size = min(cfg.batch_size, 32)


def _resolve_aaco_bundle_path(cfg: AACONNTrainConfig) -> Path:
    bundle_path = cfg.pretrained_model_bundle_path or cfg.aaco_bundle_path
    assert (
        bundle_path is not None
    ), "Expected pretrained_model_bundle_path or aaco_bundle_path."
    return Path(bundle_path)


def _resolve_train_dataset_path(cfg: AACONNTrainConfig) -> Path:
    dataset_path = cfg.train_dataset_bundle_path or cfg.dataset_artifact_name
    assert (
        dataset_path is not None
    ), "Expected train_dataset_bundle_path or dataset_artifact_name."
    return Path(dataset_path)


def _load_aaco_method(
    cfg: AACONNTrainConfig, device: torch.device
) -> tuple[AACOAFAMethod, bool]:
    aaco_bundle_path = _resolve_aaco_bundle_path(cfg)
    logger.info(f"Loading AACO method from {aaco_bundle_path}...")
    aaco_method, _aaco_manifest = load_bundle(aaco_bundle_path, device=device)
    assert isinstance(aaco_method, AACOAFAMethod)
    logger.info("Loaded AACO method")
    force_acquisition = cfg.hard_budget is not None
    aaco_method.force_acquisition = force_acquisition
    return aaco_method, force_acquisition


def _load_rollout_dataset(
    cfg: AACONNTrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, str, int | None]:
    dataset_bundle_path = _resolve_train_dataset_path(cfg)
    logger.info(f"Loading dataset from {dataset_bundle_path}...")
    dataset_obj, dataset_manifest = load_bundle(dataset_bundle_path)
    dataset_name = (
        dataset_manifest["class_name"].replace("Dataset", "").lower()
    )
    split = dataset_manifest["metadata"].get("split_idx", None)
    dataset: Any = dataset_obj
    x_train, y_train = dataset.get_all_data()
    return x_train, y_train, dataset.feature_shape, dataset_name, split


def _prepare_rollout_data(
    cfg: AACONNTrainConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    feature_shape: torch.Size,
    _selection_size: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if len(feature_shape) > 1:
        x_train = x_train.view(x_train.shape[0], -1)
        logger.info(
            f"Flattened features from {feature_shape} to {x_train.shape[1]}"
        )

    if cfg.smoke_test:
        max_samples = min(100, len(x_train))
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]
        logger.info(
            f"Smoke test: using only {max_samples} samples for rollouts"
        )

    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    logger.info(
        "Dataset: %s samples, %s features, %s classes",
        len(x_train),
        n_features,
        n_classes,
    )
    return x_train, y_train, n_features, n_classes


def _resolve_rollout_max(cfg: AACONNTrainConfig) -> int | None:
    if cfg.max_acquisitions is not None:
        return cfg.max_acquisitions
    if cfg.hard_budget is not None:
        return int(cfg.hard_budget)
    return None


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco_nn",
    config_name="config",
)
def main(cfg: AACONNTrainConfig) -> None:
    logger.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    _configure_smoke_test(cfg)
    aaco_method, force_acquisition = _load_aaco_method(cfg, device)
    if cfg.soft_budget_param is not None:
        aaco_method.set_cost_param(cfg.soft_budget_param)
    x_train, y_train, feature_shape, dataset_name, split = (
        _load_rollout_dataset(cfg)
    )
    initializer = get_afa_initializer_from_config(cfg.initializer)
    initializer.set_seed(cfg.seed)
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
    selection_size = unmasker.get_n_selections(feature_shape=feature_shape)
    x_train, y_train, n_features, n_classes = _prepare_rollout_data(
        cfg, x_train, y_train, feature_shape, selection_size
    )
    rollout_max_acquisitions = _resolve_rollout_max(cfg)

    # Generate rollouts from AACO oracle
    logger.info("Generating AACO rollouts...")
    aaco_method.set_exclude_instance(True)
    masked_features, feature_masks, actions = generate_aaco_rollouts(
        aaco_method=aaco_method,
        features=x_train,
        labels=y_train,
        feature_shape=feature_shape,
        unmasker=unmasker,
        initializer=initializer,
        max_acquisitions=rollout_max_acquisitions,
        device=device,
    )
    logger.info(f"Generated {len(actions)} state-action pairs")

    # Create data loaders
    train_loader, val_loader, n_train, n_val = create_rollout_data_loaders(
        masked_features,
        feature_masks,
        actions,
        cfg.batch_size,
        cfg.val_split,
        cfg.seed,
    )
    logger.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # Create policy network
    n_actions = selection_size + 1  # selection_size + stop action
    policy_network = AACOPolicyNetwork(
        n_features=n_features,
        n_actions=n_actions,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    logger.info(f"Created policy network with {n_actions} actions")

    # Train policy network
    logger.info("Training policy network...")
    policy_network = train_policy_network(
        policy_network=policy_network,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        patience=cfg.early_stopping_patience,
        device=device,
    )
    logger.info("Training complete")

    # Load classifier for the final method
    logger.info(f"Loading classifier from {cfg.classifier_bundle_path}...")
    classifier, _ = load_bundle(
        Path(cfg.classifier_bundle_path), device=device
    )
    classifier = cast("Any", classifier)
    logger.info("Loaded classifier")

    # Create AACO+NN method
    aaco_nn_method = create_aaco_nn_method(
        policy_network=policy_network,
        classifier=classifier,
        dataset_name=dataset_name,
        classifier_bundle_path=Path(cfg.classifier_bundle_path),
        force_acquisition=force_acquisition,
        device=device,
    )

    # Save
    save_bundle(
        obj=aaco_nn_method,
        path=Path(cfg.save_path),
        metadata={
            "aaco_bundle_path": str(_resolve_aaco_bundle_path(cfg)),
            "dataset_artifact": str(_resolve_train_dataset_path(cfg)),
            "dataset_name": dataset_name,
            "classifier_bundle_path": str(cfg.classifier_bundle_path),
            "split_idx": split,
            "seed": cfg.seed,
            "hard_budget": cfg.hard_budget,
            "soft_budget_param": cfg.soft_budget_param,
            "force_acquisition": force_acquisition,
            "max_acquisitions": rollout_max_acquisitions,
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "n_features": n_features,
            "n_classes": n_classes,
            "selection_size": selection_size,
            "n_rollout_samples": len(actions),
        },
    )
    logger.info(f"Saved AACO+NN method to: {cfg.save_path}")


if __name__ == "__main__":
    main()
