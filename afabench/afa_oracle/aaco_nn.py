import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self, cast, final, override

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from afabench.afa_oracle.afa_methods import AACOAFAMethod
from afabench.afa_oracle.utils import (
    compute_patch_selection_mask,
    flatten_for_aaco,
    uses_patch_selection,
)
from afabench.common.bundle import load_bundle
from afabench.common.custom_types import (
    AFAAction,
    AFAClassifier,
    AFAMethod,
    AFAUnmasker,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.initializers.random_initializer import RandomInitializer

logger = logging.getLogger(__name__)


def _build_selection_mask(
    feature_mask_structured: torch.Tensor,
    selection_size: int,
    feature_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    feature_mask_flat = feature_mask_structured.reshape(-1)
    if selection_size == feature_mask_flat.numel():
        return feature_mask_flat.bool().clone()
    if uses_patch_selection(selection_size, feature_shape):
        return compute_patch_selection_mask(
            feature_mask_structured, selection_size, feature_shape
        ).squeeze(0)
    return torch.zeros(selection_size, dtype=torch.bool, device=device)


def _init_rollout_state(
    x_flat: torch.Tensor,
    feature_shape: torch.Size,
    selection_size: int,
    initializer: RandomInitializer,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    x_view = x_flat.view(feature_shape) if len(feature_shape) > 1 else x_flat
    feature_mask_structured = initializer.initialize(
        features=x_view.unsqueeze(0),
        label=None,
        feature_shape=feature_shape,
    ).squeeze(0)
    feature_mask_structured = feature_mask_structured.to(device)
    masked_features_structured = x_view * feature_mask_structured.float()
    feature_mask_flat = feature_mask_structured.reshape(-1)
    masked_features_flat = masked_features_structured.reshape(-1)
    selection_mask = _build_selection_mask(
        feature_mask_structured, selection_size, feature_shape, device
    )
    return (
        x_view,
        feature_mask_structured,
        masked_features_structured,
        feature_mask_flat,
        masked_features_flat,
        selection_mask,
    )


def _apply_selection(
    selection_idx: int,
    *,
    x_view: torch.Tensor,
    selection_mask: torch.Tensor,
    feature_mask_structured: torch.Tensor,
    masked_features_structured: torch.Tensor,
    unmasker: AFAUnmasker,
    feature_shape: torch.Size,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selection_mask[selection_idx] = True
    new_feature_mask_structured = unmasker.unmask(
        masked_features=masked_features_structured.unsqueeze(0),
        feature_mask=feature_mask_structured.unsqueeze(0),
        features=x_view.unsqueeze(0),
        afa_selection=torch.tensor([[selection_idx]], device=device),
        selection_mask=selection_mask.unsqueeze(0),
        label=None,
        feature_shape=feature_shape,
    ).squeeze(0)
    feature_mask_structured = new_feature_mask_structured
    masked_features_structured = x_view * feature_mask_structured.float()
    feature_mask_flat = feature_mask_structured.reshape(-1)
    masked_features_flat = masked_features_structured.reshape(-1)
    return (
        feature_mask_structured,
        masked_features_structured,
        feature_mask_flat,
        masked_features_flat,
    )


def generate_aaco_rollouts(
    aaco_method: AACOAFAMethod,
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_shape: torch.Size | None = None,
    unmasker: AFAUnmasker | None = None,
    max_acquisitions: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate expert rollouts from an AACO oracle.

    For each sample, we simulate the AACO acquisition process and collect
    (state, action) pairs at each step.

    Args:
        aaco_method: Trained AACOAFAMethod instance
        features: Training features (N x d)
        labels: Training labels (N x n_classes), one-hot encoded
        feature_shape: Shape of features excluding batch dim
        unmasker: Unmasker to apply selections
        max_acquisitions: Maximum acquisitions per sample (None = until oracle stops)
        device: Device to use

    Returns:
        Tuple of (masked_features, feature_masks, actions):
        - masked_features: (total_steps, d) - observed features at each step
        - feature_masks: (total_steps, d) - boolean mask of observed features
        - actions: (total_steps,) - action taken (0=stop, 1-d=feature to acquire)
    """
    del labels  # Unused, kept for API compatibility
    if device is None:
        device = features.device
    assert isinstance(aaco_method, AACOAFAMethod)

    n_samples, n_features = features.shape
    if feature_shape is None:
        feature_shape = torch.Size([n_features])

    assert unmasker is not None, (
        "AACO+NN rollouts require an unmasker to apply selections."
    )

    selection_size = unmasker.get_n_selections(feature_shape=feature_shape)

    # Lists to collect rollout data
    all_masked_features = []
    all_feature_masks = []
    all_actions = []

    logger.info(f"Generating rollouts for {n_samples} samples...")

    initializer = RandomInitializer(num_initial_features=1)

    for i in range(n_samples):
        x_flat = features[i].to(device)
        (
            x_view,
            feature_mask_structured,
            masked_features_structured,
            feature_mask_flat,
            masked_features_flat,
            selection_mask,
        ) = _init_rollout_state(
            x_flat,
            feature_shape,
            selection_size,
            initializer,
            device,
        )
        n_acquired = int(selection_mask.sum().item())
        while True:
            # Check budget limit
            if max_acquisitions is not None and n_acquired >= max_acquisitions:
                # Forced stop at budget
                all_masked_features.append(masked_features_flat.clone())
                all_feature_masks.append(feature_mask_flat.clone())
                all_actions.append(
                    torch.tensor(0, device=device)
                )  # Stop action
                break

            # Record current state
            all_masked_features.append(masked_features_flat.clone())
            all_feature_masks.append(feature_mask_flat.clone())

            # Get AACO's action
            action = aaco_method.act(
                masked_features=masked_features_flat.unsqueeze(0),
                feature_mask=feature_mask_flat.float().unsqueeze(0),
                selection_mask=selection_mask.unsqueeze(0),
                feature_shape=feature_shape,
            )
            action_val = action.item()

            # Record action
            all_actions.append(torch.tensor(action_val, device=device))

            if action_val == 0:
                # Oracle decided to stop
                break

            # Apply action (acquire the feature)
            selection_idx = int(action_val) - 1
            (
                feature_mask_structured,
                masked_features_structured,
                feature_mask_flat,
                masked_features_flat,
            ) = _apply_selection(
                selection_idx,
                x_view=x_view,
                selection_mask=selection_mask,
                feature_mask_structured=feature_mask_structured,
                masked_features_structured=masked_features_structured,
                unmasker=unmasker,
                feature_shape=feature_shape,
                device=device,
            )
            n_acquired = int(selection_mask.sum().item())

        if (i + 1) % 1000 == 0:
            logger.info(
                f"  Generated rollouts for {i + 1}/{n_samples} samples"
            )

    # Stack into tensors
    all_masked_features = torch.stack(all_masked_features)
    all_feature_masks = torch.stack(all_feature_masks)
    all_actions = torch.stack(all_actions)

    logger.info(
        "Generated %s state-action pairs from %s samples",
        len(all_actions),
        n_samples,
    )
    logger.info(
        f"  Average steps per sample: {len(all_actions) / n_samples:.2f}"
    )

    return all_masked_features, all_feature_masks, all_actions


class AACOPolicyNetwork(nn.Module):
    """
    Neural network policy for AACO+NN.

    Takes (masked_features, feature_mask) as input and outputs action logits.
    Architecture follows the masked MLP classifier pattern.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,  # n_features + 1 (including stop action)
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Input: concatenated (masked_features, feature_mask)
        input_dim = n_features * 2

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network: nn.Sequential = nn.Sequential(*layers)
        self.n_features: int = n_features
        self.n_actions: int = n_actions

    @override
    def forward(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            masked_features: (batch, d) observed features (unobserved = 0)
            feature_mask: (batch, d) boolean mask of observed features

        Returns:
            action_logits: (batch, n_actions) logits for each action
        """
        # Concatenate masked features and mask
        x = torch.cat([masked_features, feature_mask.float()], dim=-1)
        return self.network(x)


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
    optimizer = torch.optim.AdamW(
        policy_network.parameters(), lr=learning_rate
    )
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

        logger.info(
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
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state_dict is not None:
        policy_network.load_state_dict(best_state_dict)
        logger.info(
            f"Loaded best model with validation loss {best_val_loss:.4f}"
        )

    return policy_network


def create_rollout_data_loaders(
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


@dataclass
@final
class AACONNAFAMethod(AFAMethod):
    """
    AACO+NN: Neural network approximation of AACO.

    This method uses a trained neural network policy to select features,
    providing much faster inference than the KNN-based AACO oracle while
    maintaining similar acquisition quality.
    """

    policy_network: AACOPolicyNetwork
    classifier: AFAClassifier
    dataset_name: str
    classifier_bundle_path: Path | None = None
    _device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    force_acquisition: bool = False

    def __post_init__(self):
        """Move to device after initialization."""
        self.policy_network = self.policy_network.to(self._device)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """
        Select next feature to acquire using the neural network policy.

        Args:
            masked_features: Currently observed features (unobserved = 0)
            feature_mask: Boolean mask of observed features
            selection_mask: Which selections have been made (unused)
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            AFAAction tensor with shape (*batch, 1):
            - 0 = stop acquiring
            - 1 to N = 1-indexed feature to acquire
        """
        del label  # Unused
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        masked_features, feature_mask, batch_shape = flatten_for_aaco(
            masked_features, feature_mask, feature_shape
        )
        n_features = masked_features.shape[-1]

        # Get policy network predictions
        self.policy_network.eval()
        with torch.no_grad():
            action_logits = self.policy_network(masked_features, feature_mask)
            # Shape: (batch, n_actions) where n_actions = n_features + 1

        # Apply mask to already-acquired features (can't re-acquire)
        # Action 0 = stop, actions 1-n = acquire feature 0 to n-1
        # Set logits to -inf for already acquired features
        selection_size = self.policy_network.n_actions - 1
        selection_mask_flat = (
            selection_mask.view(-1, selection_size).bool()
            if selection_mask is not None
            else None
        )

        if selection_size == n_features:
            acquired_mask = feature_mask.bool()
        elif uses_patch_selection(selection_size, feature_shape):
            assert feature_shape is not None
            fm = feature_mask.view(-1, *feature_shape)
            acquired_mask = compute_patch_selection_mask(
                fm, selection_size, feature_shape
            )
        else:
            acquired_mask = torch.zeros(
                (masked_features.shape[0], selection_size),
                dtype=torch.bool,
                device=self._device,
            )

        if selection_mask_flat is not None:
            acquired_mask = acquired_mask | selection_mask_flat

        action_logits[:, 1:][acquired_mask] = float("-inf")

        if self.force_acquisition:
            has_available = ~acquired_mask.all(dim=-1)
            action_logits[has_available, 0] = float("-inf")

        # Select best action
        actions = action_logits.argmax(dim=-1)

        # Reshape to match batch shape
        return actions.view(*batch_shape, 1).to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Make prediction using the classifier.

        Args:
            masked_features: Currently observed features
            feature_mask: Boolean mask of observed features
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            Class probabilities with shape (*batch, n_classes)
        """
        del label  # Unused
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        masked_features_flat, feature_mask_flat, batch_shape = (
            flatten_for_aaco(masked_features, feature_mask, feature_shape)
        )

        # Get classifier predictions
        classifier_shape = torch.Size([masked_features_flat.shape[1]])
        with torch.no_grad():
            logits = self.classifier(
                masked_features_flat,
                feature_mask_flat,
                feature_shape=classifier_shape,
            )
            probs = F.softmax(logits, dim=-1)

        # Reshape and return
        n_classes = probs.shape[-1]
        output_shape = batch_shape + (n_classes,)
        return probs.view(*output_shape).to(original_device)

    @property
    @override
    def cost_param(self) -> float:
        """Return a dummy cost param (not used by AACO+NN)."""
        return 0.0

    @override
    def set_cost_param(self, cost_param: float) -> None:
        """Set cost param (no-op for AACO+NN, budget is implicit in training)."""

    @override
    def save(self, path: Path) -> None:
        """Save method state to disk."""
        state = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "n_features": self.policy_network.n_features,
            "n_actions": self.policy_network.n_actions,
            "dataset_name": self.dataset_name,
            "force_acquisition": self.force_acquisition,
            "classifier_bundle_path": str(self.classifier_bundle_path)
            if self.classifier_bundle_path is not None
            else None,
        }
        torch.save(state, path / f"aaco_nn_{self.dataset_name}.pt")
        logger.info(f"Saved AACO+NN method to {path}")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device | None = None) -> Self:
        """Load method from disk."""
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        nn_files = list(path.glob("aaco_nn_*.pt"))
        if not nn_files:
            msg = f"No AACO+NN files found in {path}"
            raise FileNotFoundError(msg)

        state = torch.load(
            nn_files[0], map_location=device, weights_only=False
        )

        # Reconstruct policy network
        policy_network = AACOPolicyNetwork(
            n_features=state["n_features"],
            n_actions=state["n_actions"],
        )
        policy_network.load_state_dict(state["policy_network_state_dict"])

        # Load classifier from saved bundle path
        classifier_bundle_path = state.get("classifier_bundle_path")
        assert classifier_bundle_path is not None, (
            "AACO+NN loading requires classifier_bundle_path in saved state."
        )

        classifier_bundle_path = Path(classifier_bundle_path)
        classifier, _ = load_bundle(classifier_bundle_path, device=device)
        classifier = cast("AFAClassifier", cast("object", classifier))

        force_acquisition = state.get("force_acquisition")
        if force_acquisition is None:
            force_acquisition = state.get("hard_budget") is not None

        method = cls(
            policy_network=policy_network,
            classifier=classifier,
            dataset_name=state["dataset_name"],
            classifier_bundle_path=classifier_bundle_path,
            force_acquisition=force_acquisition,
            _device=device,
        )

        return method

    @override
    def to(self, device: torch.device) -> Self:
        """Move method to device."""
        self._device = device
        self.policy_network = self.policy_network.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """Get current device."""
        return self._device


def create_aaco_nn_method(
    policy_network: AACOPolicyNetwork,
    classifier: AFAClassifier,
    dataset_name: str,
    classifier_bundle_path: Path | None = None,
    *,
    force_acquisition: bool = False,
    device: torch.device | None = None,
) -> AACONNAFAMethod:
    """
    Create an AACO+NN method instance.

    Args:
        policy_network: Trained policy network
        classifier: Classifier for predictions
        dataset_name: Name of dataset
        classifier_bundle_path: Path to classifier bundle (for save/load)
        force_acquisition: If True, never stop early (hard budget mode)
        device: Device to use

    Returns:
        Configured AACONNAFAMethod instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return AACONNAFAMethod(
        policy_network=policy_network,
        classifier=classifier,
        dataset_name=dataset_name,
        classifier_bundle_path=classifier_bundle_path,
        force_acquisition=force_acquisition,
        _device=device,
    )
