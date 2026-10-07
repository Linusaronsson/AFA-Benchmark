"""
AACO+NN training script.

Trains a neural network policy via behavioral cloning.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from omegaconf import OmegaConf

import afabench.components.methods.oracle.aaco.config  # noqa: F401  # pyright: ignore[reportUnusedImport]
from afabench.components.methods.oracle import (
    AACOPolicyNetwork,
    create_aaco_nn_method,
    create_rollout_data_loaders,
    generate_aaco_rollouts,
    train_policy_network,
)
from afabench.components.methods.oracle.aaco.afa_methods import AACOAFAMethod
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result

if TYPE_CHECKING:
    from afabench.components.methods.oracle.aaco.config import (
        AACONNTrainConfig,
    )
    from afabench.core.types import AFAClassifier

logger = logging.getLogger(__name__)


def _configure_smoke_test(cfg: AACONNTrainConfig) -> AACONNTrainConfig:
    if not cfg.smoke_test:
        return cfg
    logger.info("Smoke test mode: reducing training samples and epochs")
    return replace(cfg, max_epochs=2, batch_size=min(cfg.batch_size, 32))


def _prepare_rollout_data(
    cfg: AACONNTrainConfig,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    feature_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor, int]:
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
    return x_train, y_train, n_features


def _resolve_rollout_max(cfg: AACONNTrainConfig) -> int | None:
    if cfg.max_acquisitions is not None:
        return cfg.max_acquisitions
    if cfg.hard_budget is not None:
        return int(cfg.hard_budget)
    return None


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/aaco_nn",
    config_name="config",
)
def main(cfg: AACONNTrainConfig) -> None:
    cfg = cast("AACONNTrainConfig", OmegaConf.to_object(cfg))
    logger.debug(cfg)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    with fit_run(cfg, tags=["aaco_nn"], config=cfg) as metrics:
        cfg = _configure_smoke_test(cfg)
        inputs = load_inputs(cfg)
        aaco_method = inputs.pretrained_model(AACOAFAMethod)
        force_acquisition = cfg.hard_budget is not None
        aaco_method.force_acquisition = force_acquisition
        if cfg.soft_budget_param is not None:
            aaco_method.set_cost_param(cfg.soft_budget_param)
        dataset = inputs.train_dataset()
        x_train, y_train = dataset.get_all_data()
        feature_shape = dataset.feature_shape
        initializer = inputs.initializer()
        initializer.set_seed(cfg.seed)
        unmasker = inputs.unmasker()
        selection_size = unmasker.get_n_selections(feature_shape=feature_shape)
        x_train, y_train, n_features = _prepare_rollout_data(
            cfg, x_train, y_train, feature_shape
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
            metric_logger=metrics.log,
        )
        logger.info("Training complete")

        classifier = cast("AFAClassifier", inputs.classifier(object))

        # Create AACO+NN method
        aaco_nn_method = create_aaco_nn_method(
            policy_network=policy_network,
            classifier=classifier,
            dataset_name=cfg.dataset_key,
            classifier_bundle_path=Path(cfg.classifier_bundle_path),
            force_acquisition=force_acquisition,
            device=device,
        )

        save_result(aaco_nn_method, cfg, cfg)


if __name__ == "__main__":
    main()
