import logging
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from afabench.components.methods.oracle import create_aaco_method
from afabench.components.methods.oracle.aaco.afa_methods import AACOAFAMethod
from afabench.components.methods.oracle.aaco.config import AACOTrainConfig
from afabench.fit.inputs import load_inputs
from afabench.fit.smoke_test import training_subset

logger = logging.getLogger(__name__)


def run(cfg: AACOTrainConfig) -> AACOAFAMethod:
    logger.debug(cfg)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    inputs = load_inputs(cfg)
    dataset = inputs.train_dataset()
    logger.info(f"Training instances: {len(dataset)}")

    X_train, y_train = dataset.get_all_data()
    feature_shape = dataset.feature_shape

    if len(feature_shape) > 1:
        X_train = X_train.view(X_train.shape[0], -1)
        logger.info(
            f"Flattened features from {feature_shape} to {X_train.shape[1]}"
        )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_train, y_train = training_subset(
        X_train,
        y_train,
        smoke_test=cfg.smoke_test,
    )

    logger.debug(
        "X_train shape %s, y_train shape %s",
        X_train.shape,
        y_train.shape,
    )
    logger.debug(f"Feature shape: {feature_shape}")

    soft_budget_param = (
        cfg.soft_budget_param
        if cfg.soft_budget_param is not None
        else cfg.aco.acquisition_cost
    )
    force_acquisition = cfg.hard_budget is not None

    classifier_bundle_path = Path(cfg.classifier_bundle_path)

    assert classifier_bundle_path.exists(), (
        f"Classifier bundle not found at: {classifier_bundle_path}"
    )

    unmasker = inputs.unmasker()
    selection_size = unmasker.get_n_selections(
        feature_shape=dataset.feature_shape
    )
    selection_costs = unmasker.get_selection_costs(
        feature_costs=dataset.get_feature_acquisition_costs()
    ).to(device)
    if OmegaConf.is_config(cfg.unmasker.kwargs):
        unmasker_kwargs = cast(
            "dict[str, Any]",
            OmegaConf.to_container(cfg.unmasker.kwargs, resolve=True),
        )
    else:
        unmasker_kwargs = dict(cfg.unmasker.kwargs)

    aaco_method = create_aaco_method(
        dataset_name=cfg.dataset_key,
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=soft_budget_param,
        hide_val=cfg.aco.hide_val,
        mask_seed=cfg.aco.mask_seed,
        force_acquisition=force_acquisition,
        selection_size=selection_size,
        unmasker_class_name=cfg.unmasker.class_name,
        unmasker_kwargs=unmasker_kwargs,
        selection_costs=selection_costs,
        classifier_bundle_path=classifier_bundle_path,
        device=device,
    )

    logger.info("Fitting AACO oracle on training data...")
    aaco_method.aaco_oracle.fit(X_train, y_train)
    logger.info(
        "AACO oracle fitted with classifier from %s",
        classifier_bundle_path,
    )

    return aaco_method
