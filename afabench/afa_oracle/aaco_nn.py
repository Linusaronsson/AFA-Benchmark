"""
AACO+NN: Neural network approximation of AACO via behavioral cloning.

This module implements the AACO+NN approach from Valancius et al. 2024 (Section 3.5),
which trains a neural network to imitate the AACO oracle for faster inference.

The approach:
1. Generate expert rollouts from a trained AACO oracle
2. Collect (state, action) pairs from these rollouts
3. Train a policy network via behavioral cloning (supervised learning)
4. At inference time, use the fast NN policy instead of the expensive KNN oracle
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self, final, override

import torch
import torch.nn.functional as F
from torch import nn

from afabench.common.custom_types import (
    AFAAction,
    AFAClassifier,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)

logger = logging.getLogger(__name__)


def generate_aaco_rollouts(
    aaco_method: "AFAMethod",
    features: torch.Tensor,
    labels: torch.Tensor,
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

    n_samples, n_features = features.shape

    # Lists to collect rollout data
    all_masked_features = []
    all_feature_masks = []
    all_actions = []

    logger.info(f"Generating rollouts for {n_samples} samples...")

    for i in range(n_samples):
        x = features[i].to(device)

        # Initialize with first feature (following AACODefaultInitializer)
        # Start with feature 0 observed
        feature_mask = torch.zeros(n_features, dtype=torch.bool, device=device)
        feature_mask[0] = True
        masked_features = x * feature_mask.float()

        n_acquired = 1
        while True:
            # Check budget limit
            if max_acquisitions is not None and n_acquired >= max_acquisitions:
                # Forced stop at budget
                all_masked_features.append(masked_features.clone())
                all_feature_masks.append(feature_mask.clone())
                all_actions.append(
                    torch.tensor(0, device=device)
                )  # Stop action
                break

            # Record current state
            all_masked_features.append(masked_features.clone())
            all_feature_masks.append(feature_mask.clone())

            # Get AACO's action
            action = aaco_method.act(
                masked_features=masked_features.unsqueeze(0),
                feature_mask=feature_mask.float().unsqueeze(0),
                feature_shape=torch.Size([n_features]),
            )
            action_val = action.item()

            # Record action
            all_actions.append(torch.tensor(action_val, device=device))

            if action_val == 0:
                # Oracle decided to stop
                break

            # Apply action (acquire the feature)
            feature_idx = int(action_val) - 1  # Convert 1-indexed to 0-indexed
            feature_mask[feature_idx] = True
            masked_features = x * feature_mask.float()
            n_acquired += 1

        if (i + 1) % 1000 == 0:
            logger.info(
                f"  Generated rollouts for {i + 1}/{n_samples} samples"
            )

    # Stack into tensors
    all_masked_features = torch.stack(all_masked_features)
    all_feature_masks = torch.stack(all_feature_masks)
    all_actions = torch.stack(all_actions)

    logger.info(
        f"Generated {len(all_actions)} state-action pairs from {n_samples} samples"
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
    _device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    _hard_budget: int | None = None

    def __post_init__(self):
        """Move to device after initialization."""
        self.policy_network = self.policy_network.to(self._device)

    def set_hard_budget(self, budget: int | None) -> None:
        """Set hard budget. None = soft budget mode."""
        self._hard_budget = budget

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
        del selection_mask, label  # Unused
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Flatten features if needed
        batch_shape = (
            feature_mask.shape[:-1]
            if feature_shape is None
            else feature_mask.shape[: -len(feature_shape)]
        )
        if feature_shape is not None:
            n_features = feature_shape.numel()
            masked_features = masked_features.view(-1, n_features)
            feature_mask = feature_mask.view(-1, n_features)
        else:
            n_features = masked_features.shape[-1]
            masked_features = masked_features.view(-1, n_features)
            feature_mask = feature_mask.view(-1, n_features)

        # Get policy network predictions
        self.policy_network.eval()
        with torch.no_grad():
            action_logits = self.policy_network(masked_features, feature_mask)
            # Shape: (batch, n_actions) where n_actions = n_features + 1

        # Apply mask to already-acquired features (can't re-acquire)
        # Action 0 = stop, actions 1-n = acquire feature 0 to n-1
        # Set logits to -inf for already acquired features
        acquired_mask = feature_mask.bool()
        # Shift by 1 because action 0 is stop
        action_logits[:, 1:][acquired_mask] = float("-inf")

        # Select best action
        actions = action_logits.argmax(dim=-1)

        # Handle hard budget
        if self._hard_budget is not None:
            n_acquired = feature_mask.sum(dim=-1)
            at_budget = n_acquired >= self._hard_budget
            # Force stop if at budget
            actions = torch.where(
                at_budget, torch.zeros_like(actions), actions
            )

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

        # Flatten if needed
        if feature_shape is not None:
            batch_shape = feature_mask.shape[: -len(feature_shape)]
            n_features = feature_shape.numel()
            masked_features_flat = masked_features.view(-1, n_features)
            feature_mask_flat = feature_mask.view(-1, n_features)
        else:
            batch_shape = feature_mask.shape[:-1]
            masked_features_flat = masked_features.view(
                -1, masked_features.shape[-1]
            )
            feature_mask_flat = feature_mask.view(-1, feature_mask.shape[-1])

        # Get classifier predictions
        with torch.no_grad():
            logits = self.classifier(masked_features_flat, feature_mask_flat)
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

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        """AACO+NN has a builtin classifier for predictions."""
        return True

    @override
    def save(self, path: Path) -> None:
        """Save method state to disk."""
        state = {
            "policy_network_state_dict": self.policy_network.state_dict(),
            "n_features": self.policy_network.n_features,
            "n_actions": self.policy_network.n_actions,
            "dataset_name": self.dataset_name,
            "hard_budget": self._hard_budget,
        }
        torch.save(state, path / f"aaco_nn_{self.dataset_name}.pt")

        # Save classifier reference (assumed to be saved separately)
        # The classifier bundle path should be provided during training
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

        state = torch.load(nn_files[0], map_location=device)

        # Reconstruct policy network
        policy_network = AACOPolicyNetwork(
            n_features=state["n_features"],
            n_actions=state["n_actions"],
        )
        policy_network.load_state_dict(state["policy_network_state_dict"])

        # Load classifier (needs classifier_bundle_path in manifest)
        # For now, we'll require it to be passed during method creation
        # TODO: Save and load classifier path like in AACOAFAMethod
        msg = "AACO+NN loading requires classifier. Use create_aaco_nn_method() instead."
        raise NotImplementedError(msg)

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
    hard_budget: int | None = None,
    device: torch.device | None = None,
) -> AACONNAFAMethod:
    """
    Create an AACO+NN method instance.

    Args:
        policy_network: Trained policy network
        classifier: Classifier for predictions
        dataset_name: Name of dataset
        hard_budget: Max features to acquire (None = soft budget)
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
        _device=device,
        _hard_budget=hard_budget,
    )
