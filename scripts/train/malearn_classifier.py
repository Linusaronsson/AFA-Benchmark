import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.classifiers import WrappedMALearnClassifier
from afabench.common.config_classes import TrainMALearnClassifierConfig
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.naming import infer_dataset_key_from_class_name
from afabench.common.utils import set_seed
from afabench.missing_values.malearn import (
    MADTClassifier,
    MAGBTClassifier,
    MALassoClassifier,
    MARFClassifier,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset


def _to_numpy_data(
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    n_feature_dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert feature/label tensors to flat numpy arrays for sklearn."""
    X = X_t.detach().flatten(start_dim=-n_feature_dims).cpu().numpy()
    y = y_t.detach().argmax(dim=-1).cpu().numpy().astype(int)
    return X, y


def _sample_missingness_mask(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    min_probability: float,
    max_probability: float,
) -> np.ndarray:
    per_sample_p = rng.uniform(
        low=min_probability,
        high=max_probability,
        size=(n_samples, 1),
    )
    return (rng.uniform(size=(n_samples, n_features)) < per_sample_p).astype(
        np.int8
    )


def _sample_missingness_mask_from_initializer(
    cfg: TrainMALearnClassifierConfig,
    x: np.ndarray,
) -> np.ndarray | None:
    if cfg.initializer.class_name != "MissingnessInitializer":
        return None

    initializer = get_afa_initializer_from_config(cfg.initializer)
    initializer.set_seed(cfg.seed)

    x_t = torch.tensor(x, dtype=torch.float32)
    observed_mask = initializer.initialize(
        features=x_t,
        feature_shape=torch.Size([x.shape[1]]),
    )
    # MA-learn convention: M=1 means missing.
    return (~observed_mask.bool()).cpu().numpy().astype(np.int8)


def _build_model(
    cfg: TrainMALearnClassifierConfig,
) -> object:
    model_name = cfg.model_name.lower()
    n_estimators = cfg.n_estimators
    max_depth = cfg.max_depth
    if cfg.smoke_test:
        n_estimators = min(n_estimators, 32)
        max_depth = min(max_depth, 6)

    if model_name == "malasso":
        return MALassoClassifier(
            alpha=cfg.alpha,
            beta=cfg.beta,
            random_state=cfg.seed,
        )

    if model_name == "madt":
        return MADTClassifier(
            max_depth=max_depth,
            alpha=cfg.alpha,
            random_state=cfg.seed,
        )

    if model_name == "marf":
        return MARFClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            alpha=cfg.alpha,
            random_state=cfg.seed,
            n_jobs=cfg.n_jobs,
        )

    if model_name == "magbt":
        return MAGBTClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            alpha=cfg.alpha,
            learning_rate=cfg.learning_rate,
            subsample=cfg.subsample,
            random_state=cfg.seed,
        )

    msg = "Unknown model_name. Expected one of: malasso, madt, marf, magbt."
    raise ValueError(msg)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/classifiers/malearn_classifier",
    config_name="config",
)
def main(cfg: TrainMALearnClassifierConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)

    train_dataset, train_manifest = load_bundle(Path(cfg.train_dataset_path))
    train_dataset = cast("AFADataset", cast("object", train_dataset))

    val_dataset, _ = load_bundle(Path(cfg.val_dataset_path))
    val_dataset = cast("AFADataset", cast("object", val_dataset))

    feature_shape = train_dataset.feature_shape
    n_feature_dims = len(feature_shape)

    X_train, y_train = _to_numpy_data(*train_dataset.get_all_data(), n_feature_dims)
    X_val, y_val = _to_numpy_data(*val_dataset.get_all_data(), n_feature_dims)

    rng = np.random.default_rng(cfg.seed)

    if cfg.smoke_test:
        log.info("Smoke test mode detected. Reducing MA model complexity.")
        n_smoke_samples = min(500, X_train.shape[0])
        smoke_idx = rng.choice(
            X_train.shape[0], n_smoke_samples, replace=False
        )
        X_train = X_train[smoke_idx]
        y_train = y_train[smoke_idx]

    M_train = _sample_missingness_mask_from_initializer(cfg, X_train)
    M_val = _sample_missingness_mask_from_initializer(cfg, X_val)
    if M_train is not None and M_val is not None:
        log.info(
            "Using MissingnessInitializer masks for MA training/evaluation."
        )
    else:
        M_train = _sample_missingness_mask(
            rng=rng,
            n_samples=X_train.shape[0],
            n_features=X_train.shape[1],
            min_probability=cfg.min_masking_probability,
            max_probability=cfg.max_masking_probability,
        )

        M_val = _sample_missingness_mask(
            rng=rng,
            n_samples=X_val.shape[0],
            n_features=X_val.shape[1],
            min_probability=cfg.min_masking_probability,
            max_probability=cfg.max_masking_probability,
        )

    log.info(
        "Mask prevalence: train=%.4f, val=%.4f",
        float(M_train.mean()),
        float(M_val.mean()),
    )

    model = _build_model(cfg)
    model.fit(X_train, y_train, M=M_train)

    wrapped_classifier = WrappedMALearnClassifier(
        model=model,
        model_name=cfg.model_name.lower(),
        n_classes=train_dataset.label_shape[-1],
        device=torch.device(cfg.device),
    )

    # Quick sanity metric on randomly masked validation features.
    X_val_masked_t = torch.tensor(X_val, dtype=torch.float32)
    feature_mask_t = torch.tensor(~M_val.astype(bool), dtype=torch.bool)
    X_val_masked_t[~feature_mask_t] = 0.0
    val_probs = wrapped_classifier(
        masked_features=X_val_masked_t,
        feature_mask=feature_mask_t,
        feature_shape=feature_shape,
    )
    val_pred = val_probs.argmax(dim=-1).cpu().numpy()
    val_acc = float(np.mean(val_pred == y_val))
    log.info("Validation accuracy (masked): %.4f", val_acc)

    dataset_name = infer_dataset_key_from_class_name(
        train_manifest["class_name"]
    )

    save_bundle(
        obj=wrapped_classifier,
        path=Path(cfg.save_path),
        metadata={
            "config": OmegaConf.to_container(cfg, resolve=True),
            "dataset_name": dataset_name,
            "validation_accuracy_masked": val_acc,
        },
    )
    log.info("Saved MA classifier to: %s", cfg.save_path)


if __name__ == "__main__":
    main()
