from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from afabench.afa_oracle import (
    AACOAFAMethod,
    AACOPolicyNetwork,
    create_aaco_nn_method,
    generate_aaco_rollouts,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.utils import set_seed

if TYPE_CHECKING:
    from afabench.common.config_classes import AACONNTrainConfig
    from afabench.common.custom_types import AFAClassifier

log = logging.getLogger(__name__)


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


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco_nn",
    config_name="config",
)
def main(cfg: AACONNTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    # Handle smoke test
    if cfg.smoke_test:
        log.info("Smoke test mode: reducing training samples and epochs")
        cfg.max_epochs = 2
        cfg.batch_size = min(cfg.batch_size, 32)

    # Load AACO method
    log.info(f"Loading AACO method from {cfg.aaco_bundle_path}...")
    aaco_method, _aaco_manifest = load_bundle(
        Path(cfg.aaco_bundle_path), device=device
    )
    aaco_method = cast("AACOAFAMethod", cast("object", aaco_method))
    log.info("Loaded AACO method")

    # Load dataset for rollout generation
    log.info(f"Loading dataset from {cfg.dataset_artifact_name}...")
    dataset_obj, dataset_manifest = load_bundle(
        Path(cfg.dataset_artifact_name)
    )
    dataset_name = (
        dataset_manifest["class_name"].replace("Dataset", "").lower()
    )
    split = dataset_manifest["metadata"].get("split_idx", None)

    dataset: Any = dataset_obj
    X_train, y_train = dataset.get_all_data()
    feature_shape = dataset.feature_shape
    if len(feature_shape) > 1:
        X_train = X_train.view(X_train.shape[0], -1)
        log.info(
            f"Flattened features from {feature_shape} to {X_train.shape[1]}"
        )

    # Limit samples in smoke test mode
    if cfg.smoke_test:
        max_samples = min(100, len(X_train))
        X_train = X_train[:max_samples]
        y_train = y_train[:max_samples]
        log.info(f"Smoke test: using only {max_samples} samples for rollouts")

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    log.info(
        f"Dataset: {len(X_train)} samples, {n_features} features, {
            n_classes
        } classes"
    )

    # Generate rollouts from AACO oracle
    log.info("Generating AACO rollouts...")
    masked_features, feature_masks, actions = generate_aaco_rollouts(
        aaco_method=aaco_method,
        features=X_train,
        labels=y_train,
        max_acquisitions=cfg.max_acquisitions,
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
    n_actions = n_features + 1  # n_features + stop action
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

    # Load classifier for the AACO+NN method
    log.info(f"Loading classifier from {cfg.classifier_bundle_path}...")
    classifier, _ = load_bundle(
        Path(cfg.classifier_bundle_path), device=device
    )
    classifier = cast("AFAClassifier", cast("object", classifier))
    log.info("Loaded classifier")

    # Create AACO+NN method
    aaco_nn_method = create_aaco_nn_method(
        policy_network=policy_network,
        classifier=classifier,
        dataset_name=dataset_name,
        hard_budget=cfg.hard_budget,
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
            "max_acquisitions": cfg.max_acquisitions,
            "hidden_dims": cfg.hidden_dims,
            "dropout": cfg.dropout,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_rollout_samples": len(actions),
        },
    )
    log.info(f"Saved AACO+NN method to: {cfg.save_path}")


if __name__ == "__main__":
    main()
