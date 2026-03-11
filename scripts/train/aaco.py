import logging
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.afa_oracle import create_aaco_method
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import AACOTrainConfig, InitializerConfig
from afabench.common.custom_types import AFAInitializer
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import set_seed

logger = logging.getLogger(__name__)


_TRAIN_MISSING_MODES = {"full", "zero_fill", "impute_mean", "mask_aware"}


def _validate_train_missing_mode(mode: str) -> None:
    if mode in _TRAIN_MISSING_MODES:
        return
    msg = (
        "train_missing_mode must be one of "
        f"{sorted(_TRAIN_MISSING_MODES)}, got {mode!r}."
    )
    raise ValueError(msg)


def _derive_train_support_masks(
    initializer: AFAInitializer,
    *,
    seed: int,
    features: torch.Tensor,
    feature_shape: torch.Size,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    initializer.set_seed(seed)
    observed_mask = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    ).bool()
    forbidden_mask = initializer.get_training_forbidden_mask(
        observed_mask
    ).bool()
    if bool((observed_mask & forbidden_mask).any().item()):
        msg = (
            "Initializer returned overlapping observed and forbidden "
            "training masks."
        )
        raise ValueError(msg)
    train_support_mask = ~forbidden_mask
    return observed_mask, train_support_mask


def _apply_train_missingness(
    x_train: torch.Tensor,
    train_support_mask: torch.BoolTensor,
    mode: str,
    hide_val: float,
) -> tuple[torch.Tensor, torch.BoolTensor | None]:
    masked_x = x_train.clone()
    missing = ~train_support_mask

    if mode in {"zero_fill", "mask_aware"}:
        masked_x[missing] = hide_val
        returned_mask = train_support_mask if mode == "mask_aware" else None
        return masked_x, returned_mask

    if mode == "impute_mean":
        observed_float = train_support_mask.float()
        counts = observed_float.sum(dim=0).clamp_min(1.0)
        means = (x_train * observed_float).sum(dim=0) / counts
        masked_x[missing] = means.unsqueeze(0).expand_as(masked_x)[missing]
        return masked_x, None

    msg = f"Unsupported train missing mode: {mode}"
    raise ValueError(msg)


def _prepare_train_matrix(
    *,
    x_train: torch.Tensor,
    feature_shape: torch.Size,
    initializer_cfg: InitializerConfig,
    seed: int,
    mode: str,
    hide_val: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.BoolTensor | None, float, float]:
    initial_observed_mask_flat: torch.BoolTensor | None = None
    train_support_mask_flat: torch.BoolTensor | None = None

    if mode != "full":
        initializer = get_afa_initializer_from_config(initializer_cfg)
        initial_observed_mask, train_support_mask = (
            _derive_train_support_masks(
                initializer,
                seed=seed,
                features=x_train,
                feature_shape=feature_shape,
            )
        )
        initial_observed_mask_flat = initial_observed_mask.reshape(
            x_train.shape[0], -1
        )
        train_support_mask_flat = train_support_mask.reshape(
            x_train.shape[0], -1
        )

    if len(feature_shape) > 1:
        x_train = x_train.view(x_train.shape[0], -1)
        logger.info(
            f"Flattened features from {feature_shape} to {x_train.shape[1]}"
        )
        if initial_observed_mask_flat is not None:
            initial_observed_mask_flat = initial_observed_mask_flat.to(
                x_train.device
            )
        if train_support_mask_flat is not None:
            train_support_mask_flat = train_support_mask_flat.to(
                x_train.device
            )

    x_train = x_train.to(device)
    if initial_observed_mask_flat is not None:
        initial_observed_mask_flat = initial_observed_mask_flat.to(device)
    if train_support_mask_flat is not None:
        train_support_mask_flat = train_support_mask_flat.to(device)

    initial_fraction = (
        initial_observed_mask_flat.float().mean().item()
        if initial_observed_mask_flat is not None
        else 1.0
    )
    support_fraction = (
        train_support_mask_flat.float().mean().item()
        if train_support_mask_flat is not None
        else 1.0
    )

    if train_support_mask_flat is None:
        logger.info("Train missingness mode: full (no masking)")
        return x_train, None, initial_fraction, support_fraction

    x_train, train_observed_mask = _apply_train_missingness(
        x_train=x_train,
        train_support_mask=train_support_mask_flat,
        mode=mode,
        hide_val=hide_val,
    )
    observed_fraction = train_observed_mask.float().mean().item() if (
        train_observed_mask is not None
    ) else support_fraction
    logger.info(
        "Train missingness mode: %s (initial observed %.4f, "
        "training support %.4f, oracle-visible %.4f)",
        mode,
        initial_fraction,
        support_fraction,
        observed_fraction,
    )
    return x_train, train_observed_mask, initial_fraction, support_fraction


def run(cfg: AACOTrainConfig) -> None:
    logger.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    _validate_train_missing_mode(cfg.train_missing_mode)

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

    X_train, train_observed_mask, initial_fraction, support_fraction = (
        _prepare_train_matrix(
            x_train=X_train,
            feature_shape=feature_shape,
            initializer_cfg=cfg.initializer,
            seed=cfg.seed,
            mode=cfg.train_missing_mode,
            hide_val=cfg.aco.hide_val,
            device=device,
        )
    )
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
    aaco_method.aaco_oracle.fit(
        X_train,
        y_train,
        observed_mask=train_observed_mask,
    )
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
            "train_missing_mode": cfg.train_missing_mode,
            "force_acquisition": force_acquisition,
            "selection_size": selection_size,
            "k_neighbors": cfg.aco.k_neighbors,
            "hide_val": cfg.aco.hide_val,
            "classifier_bundle_path": str(classifier_bundle_path),
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
            "initial_observed_fraction": initial_fraction,
            "train_support_fraction": support_fraction,
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
