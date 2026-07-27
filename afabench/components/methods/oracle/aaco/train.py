import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from omegaconf import OmegaConf

from afabench.components.methods.oracle import create_aaco_method
from afabench.components.methods.oracle.aaco.config import AACOTrainConfig
from afabench.components.unmaskers.utils import get_afa_unmasker_from_config
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.naming import infer_dataset_key_from_class_name
from afabench.core.utils import set_seed
from afabench.datasets.training_views import TrainingDatasetView
from afabench.training.smoke_test import training_subset

if TYPE_CHECKING:
    from afabench.core.types import AFADataset

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

    dataset_obj, dataset_manifest = load_bundle(Path(dataset_bundle_path))
    source_class_name = getattr(
        dataset_obj,
        "source_dataset_class_name",
        dataset_manifest["class_name"],
    )
    dataset_name = infer_dataset_key_from_class_name(source_class_name)
    split = dataset_manifest["metadata"].get("split_idx", None)
    dataset = cast("AFADataset", cast("object", dataset_obj))

    logger.info(f"Dataset: {dataset_manifest['class_name']}, Split: {split}")
    logger.info(f"Training samples: {len(dataset)}")

    X_train, y_train = dataset.get_all_data()
    feature_shape = dataset.feature_shape

    if len(feature_shape) > 1:
        X_train = X_train.view(X_train.shape[0], -1)
        logger.info(
            f"Flattened features from {feature_shape} to {X_train.shape[1]}"
        )

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    train_observed_mask = None
    if (
        isinstance(dataset_obj, TrainingDatasetView)
        and dataset_obj.strategy == "restricted"
    ):
        train_observed_mask = dataset_obj.source_availability.reshape(
            len(dataset),
            -1,
        ).to(device)
    X_train, y_train = training_subset(
        X_train,
        y_train,
        smoke_test=cfg.smoke_test,
    )
    if train_observed_mask is not None:
        train_observed_mask = train_observed_mask[: len(X_train)]

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

    assert cfg.classifier_bundle_path is not None, (
        "classifier_bundle_path must be provided. Train a classifier first."
    )
    classifier_bundle_path = Path(cfg.classifier_bundle_path)

    assert classifier_bundle_path.exists(), (
        f"Classifier bundle not found at: {classifier_bundle_path}"
    )

    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
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
        dataset_name=dataset_name,
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=soft_budget_param,
        hide_val=cfg.aco.hide_val,
        missingness_objective=cfg.aco.missingness_objective,
        dr_min_propensity=cfg.aco.dr_min_propensity,
        dr_max_weight=cfg.aco.dr_max_weight,
        mask_seed=cfg.aco.mask_seed,
        force_acquisition=force_acquisition,
        selection_size=selection_size,
        unmasker_class_name=cfg.unmasker.class_name,
        unmasker_kwargs=unmasker_kwargs,
        selection_costs=selection_costs,
        classifier_bundle_path=classifier_bundle_path,
        stepwise_pvae_bundle_path=cfg.stepwise_pvae_bundle_path,
        stepwise_seed=cfg.seed,
        stepwise_n_classes=y_train.shape[-1],
        device=device,
    )

    logger.info("Fitting AACO oracle on training data...")
    aaco_method.aaco_oracle.fit(
        X_train,
        y_train,
        observed_mask=train_observed_mask,
        observation_group_ids=dataset.get_missingness_group_ids(),
    )
    logger.info(
        "AACO oracle fitted with classifier from %s",
        classifier_bundle_path,
    )

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
            "missingness_objective": cfg.aco.missingness_objective,
            "dr_min_propensity": cfg.aco.dr_min_propensity,
            "dr_max_weight": cfg.aco.dr_max_weight,
            "mask_seed": cfg.aco.mask_seed,
            "training_view_strategy": getattr(dataset, "strategy", None),
            "classifier_bundle_path": str(classifier_bundle_path),
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "stepwise_pvae_bundle_path": (
                str(cfg.stepwise_pvae_bundle_path)
                if cfg.stepwise_pvae_bundle_path is not None
                else None
            ),
        },
    )
    logger.info(f"Saved AACO method to: {cfg.save_path}")
