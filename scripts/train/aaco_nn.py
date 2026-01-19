from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, cast

import hydra
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from afabench.afa_oracle import (
    AACOPolicyNetwork,
    create_aaco_nn_method,
    generate_aaco_rollouts,
)
from afabench.afa_oracle.afa_methods import AACOAFAMethod
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import AACONNTrainConfig
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


class _Unmasker(Protocol):
    def get_n_selections(self, feature_shape: torch.Size) -> int: ...


class _Classifier(Protocol):
    def __call__(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> _Classifier: ...

    def to(self, device: torch.device) -> _Classifier: ...

    @property
    def device(self) -> torch.device: ...


def _run_train_epoch(
    policy_network: AACOPolicyNetwork,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single training epoch and return (loss, accuracy)."""
    policy_network.train()
    total_loss, correct, total = 0.0, 0, 0
    for features_cpu, masks_cpu, actions_cpu in train_loader:
        features = features_cpu.to(device)
        masks = masks_cpu.to(device)
        actions = actions_cpu.to(device)
        optimizer.zero_grad()
        logits = policy_network(features, masks)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(actions)
        correct += (logits.argmax(dim=-1) == actions).sum().item()
        total += len(actions)
    return total_loss / total, correct / total


def _run_val_epoch(
    policy_network: AACOPolicyNetwork,
    val_loader: DataLoader[Any],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single validation epoch and return (loss, accuracy)."""
    policy_network.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for features_cpu, masks_cpu, actions_cpu in val_loader:
            features = features_cpu.to(device)
            masks = masks_cpu.to(device)
            actions = actions_cpu.to(device)
            logits = policy_network(features, masks)
            loss = criterion(logits, actions)
            total_loss += loss.item() * len(actions)
            correct += (logits.argmax(dim=-1) == actions).sum().item()
            total += len(actions)
    return total_loss / total, correct / total


def train_policy_network(
    policy_network: AACOPolicyNetwork,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    max_epochs: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
) -> AACOPolicyNetwork:
    """Train the policy network via behavioral cloning."""
    policy_network = policy_network.to(device)
    optimizer = torch.optim.Adam(policy_network.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None

    for epoch in range(max_epochs):
        train_loss, train_acc = _run_train_epoch(
            policy_network, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = _run_val_epoch(
            policy_network, val_loader, criterion, device
        )

        log.info(
            f"Epoch {epoch + 1}/{max_epochs}: "
            f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state_dict = policy_network.state_dict().copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state_dict is not None:
        policy_network.load_state_dict(best_state_dict)
        log.info(f"Loaded best model with validation loss {best_val_loss:.4f}")

    return policy_network


def _create_data_loaders(
    masked_features: torch.Tensor,
    feature_masks: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    val_split: float,
    seed: int,
) -> tuple[DataLoader[Any], DataLoader[Any], int, int]:
    """Create train and validation data loaders from rollout data."""
    tensor_dataset = TensorDataset(
        masked_features.cpu(), feature_masks.cpu().float(), actions.cpu()
    )
    val_size = int(len(tensor_dataset) * val_split)
    train_size = len(tensor_dataset) - val_size

    train_dataset, val_dataset = random_split(
        tensor_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def _configure_smoke_test(cfg: AACONNTrainConfig) -> None:
    if cfg.smoke_test:
        log.info("Smoke test mode: reducing training samples and epochs")
        cfg.max_epochs = 2
        cfg.batch_size = min(cfg.batch_size, 32)


def _load_aaco_method(
    cfg: AACONNTrainConfig, device: torch.device
) -> tuple[AACOAFAMethod, bool]:
    log.info(f"Loading AACO method from {cfg.aaco_bundle_path}...")
    aaco_method, _aaco_manifest = load_bundle(
        Path(cfg.aaco_bundle_path), device=device
    )
    assert isinstance(aaco_method, AACOAFAMethod)
    log.info("Loaded AACO method")
    force_acquisition = cfg.hard_budget is not None
    aaco_method.force_acquisition = force_acquisition
    return aaco_method, force_acquisition


def _load_rollout_dataset(
    cfg: AACONNTrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size, str, int | None]:
    log.info(f"Loading dataset from {cfg.dataset_artifact_name}...")
    dataset_obj, dataset_manifest = load_bundle(
        Path(cfg.dataset_artifact_name)
    )
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
    unmasker: _Unmasker,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    selection_size = unmasker.get_n_selections(feature_shape=feature_shape)
    if len(feature_shape) > 1:
        x_train = x_train.view(x_train.shape[0], -1)
        log.info(
            f"Flattened features from {feature_shape} to {x_train.shape[1]}"
        )

    if cfg.smoke_test:
        max_samples = min(100, len(x_train))
        x_train = x_train[:max_samples]
        y_train = y_train[:max_samples]
        log.info(f"Smoke test: using only {max_samples} samples for rollouts")

    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]
    log.info(
        "Dataset: %s samples, %s features, %s classes",
        len(x_train),
        n_features,
        n_classes,
    )
    return x_train, y_train, selection_size, n_features, n_classes


def _resolve_rollout_max(cfg: AACONNTrainConfig) -> int | None:
    if cfg.max_acquisitions is not None:
        return cfg.max_acquisitions
    if cfg.hard_budget is not None:
        return int(cfg.hard_budget)
    return None


def _load_classifier(
    cfg: AACONNTrainConfig, device: torch.device
) -> _Classifier:
    log.info(f"Loading classifier from {cfg.classifier_bundle_path}...")
    classifier, _ = load_bundle(
        Path(cfg.classifier_bundle_path), device=device
    )
    classifier = cast("_Classifier", cast("object", classifier))
    log.info("Loaded classifier")
    return classifier


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco_nn",
    config_name="config",
)
def main(cfg: AACONNTrainConfig) -> None:
    assert isinstance(cfg, AACONNTrainConfig)
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    _configure_smoke_test(cfg)
    aaco_method, force_acquisition = _load_aaco_method(cfg, device)
    x_train, y_train, feature_shape, dataset_name, split = (
        _load_rollout_dataset(cfg)
    )
    unmasker = get_afa_unmasker_from_config(cfg.unmasker)
    x_train, y_train, selection_size, n_features, n_classes = (
        _prepare_rollout_data(cfg, x_train, y_train, feature_shape, unmasker)
    )
    rollout_max_acquisitions = _resolve_rollout_max(cfg)

    # Generate rollouts from AACO oracle
    log.info("Generating AACO rollouts...")
    aaco_method.set_exclude_instance(True)
    masked_features, feature_masks, actions = generate_aaco_rollouts(
        aaco_method=aaco_method,
        features=x_train,
        labels=y_train,
        feature_shape=feature_shape,
        unmasker=unmasker,
        max_acquisitions=rollout_max_acquisitions,
        device=device,
    )
    log.info(f"Generated {len(actions)} state-action pairs")

    # Create data loaders
    train_loader, val_loader, n_train, n_val = _create_data_loaders(
        masked_features,
        feature_masks,
        actions,
        cfg.batch_size,
        cfg.val_split,
        cfg.seed,
    )
    log.info(f"Train: {n_train} samples, Val: {n_val} samples")

    # Create policy network
    n_actions = selection_size + 1  # selection_size + stop action
    policy_network = AACOPolicyNetwork(
        n_features=n_features,
        n_actions=n_actions,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    log.info(f"Created policy network with {n_actions} actions")

    # Train policy network
    log.info("Training policy network...")
    policy_network = train_policy_network(
        policy_network=policy_network,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg.max_epochs,
        learning_rate=cfg.learning_rate,
        patience=cfg.early_stopping_patience,
        device=device,
    )
    log.info("Training complete")

    classifier = _load_classifier(cfg, device)

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
            "aaco_bundle_path": str(cfg.aaco_bundle_path),
            "dataset_artifact": cfg.dataset_artifact_name,
            "dataset_name": dataset_name,
            "classifier_bundle_path": str(cfg.classifier_bundle_path),
            "split_idx": split,
            "seed": cfg.seed,
            "hard_budget": cfg.hard_budget,
            "force_acquisition": force_acquisition,
            "max_acquisitions": rollout_max_acquisitions,
            "hidden_dims": list(
                cfg.hidden_dims
            ),  # Convert OmegaConf ListConfig to list
            "dropout": cfg.dropout,
            "n_features": n_features,
            "n_classes": n_classes,
            "selection_size": selection_size,
            "n_rollout_samples": len(actions),
        },
    )
    log.info(f"Saved AACO+NN method to: {cfg.save_path}")


if __name__ == "__main__":
    main()
