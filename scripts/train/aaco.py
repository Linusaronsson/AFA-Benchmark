import hydra
import torch
import logging

from pathlib import Path

from afabench.common.utils import set_seed
from afabench.afa_oracle import create_aaco_method
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import AACOTrainConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig):
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    # Load dataset bundle
    dataset_obj, dataset_manifest = load_bundle(
        Path(cfg.dataset_artifact_name))
    dataset_name = dataset_manifest["class_name"].replace(
        "Dataset", "").lower()
    split = dataset_manifest["metadata"].get("split_idx", None)

    log.info(f"Dataset: {dataset_manifest['class_name']}, Split: {split}")
    log.info(f"Training samples: {len(dataset_obj)}")

    # Get training data (flatten if needed - AACO works on flat features)
    X_train, y_train = dataset_obj.get_all_data()
    feature_shape = dataset_obj.feature_shape
    if len(feature_shape) > 1:
        X_train = X_train.view(X_train.shape[0], -1)
        log.info(f"Flattened features from {
                 feature_shape} to {X_train.shape[1]}")

    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # Determine cost parameter
    cost = cfg.cost_param if cfg.cost_param is not None else cfg.aco.acquisition_cost

    # Validate classifier path is provided
    if cfg.classifier_bundle_path is None:
        msg = "classifier_bundle_path must be provided. Train an MLP classifier first."
        raise ValueError(msg)

    classifier_bundle_path = Path(cfg.classifier_bundle_path)
    if not classifier_bundle_path.exists():
        msg = f"Classifier bundle not found at: {classifier_bundle_path}"
        raise FileNotFoundError(msg)

    # Create AACO method (classifier will be loaded in __post_init__)
    aaco_method = create_aaco_method(
        dataset_name=dataset_name,
        k_neighbors=cfg.aco.k_neighbors,
        acquisition_cost=cost,
        hide_val=cfg.aco.hide_val,
        hard_budget=cfg.hard_budget,
        classifier_bundle_path=classifier_bundle_path,
        device=device,
    )

    # Fit oracle on training data
    log.info("Fitting AACO oracle on training data...")
    aaco_method.aaco_oracle.fit(X_train, y_train)
    log.info(f"AACO oracle fitted with classifier from {classifier_bundle_path}")

    # Save
    save_bundle(
        obj=aaco_method,
        path=Path(cfg.save_path),
        metadata={
            "dataset_artifact": cfg.dataset_artifact_name,
            "dataset_name": dataset_name,
            "split_idx": split,
            "seed": cfg.seed,
            "cost_param": cost,
            "hard_budget": cfg.hard_budget,
            "k_neighbors": cfg.aco.k_neighbors,
            "hide_val": cfg.aco.hide_val,
            "classifier_bundle_path": str(classifier_bundle_path),
            "n_features": X_train.shape[1],
            "n_train_samples": len(X_train),
        },
    )
    log.info(f"Saved AACO method to: {cfg.save_path}")


if __name__ == "__main__":
    main()
