import torch
import logging

from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Self, cast, final, override

from afabench.afa_oracle.aaco_core import AACOOracle, load_mask_generator
from afabench.common.bundle import load_bundle
from afabench.common.custom_types import (
    AFAClassifier,
    AFAMethod,
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)

logger = logging.getLogger(__name__)


@dataclass
@final
class AACOAFAMethod(AFAMethod):
    """
    AACO-based Active Feature Acquisition method.

    This method uses the Acquisition Conditioned Oracle (AACO) from
    Valancius et al. 2024 to select features for acquisition.

    The method is designed to work with:
    - AFAInitializer (e.g., AACODefaultInitializer) for initial feature selection
    - AFAUnmasker (e.g., DirectUnmasker) for action-to-mask mapping

    Supports both soft budget (cost-based stopping) and hard budget
    (fixed number of acquisitions) modes.
    """
    aaco_oracle: AACOOracle
    dataset_name: str
    classifier_bundle_path: Path | None = None  # Path to trained classifier bundle
    _device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    _hard_budget: int | None = None  # None = soft budget

    def __post_init__(self):
        """Move oracle to device after initialization and load classifier if path provided."""
        self.aaco_oracle = self.aaco_oracle.to(self._device)

        # Load classifier from bundle if path provided
        if self.classifier_bundle_path is not None:
            classifier, _ = load_bundle(self.classifier_bundle_path, device=self._device)
            classifier = cast(AFAClassifier, cast(object, classifier))
            self.aaco_oracle.set_classifier(classifier)
            logger.info(f"Loaded classifier from {self.classifier_bundle_path}")

    def set_hard_budget(self, budget: int | None) -> None:
        """Set hard budget. None = soft budget mode."""
        self._hard_budget = budget

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None = None,
        label: Label | None = None,
        selection_mask: SelectionMask | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """
        Select next feature to acquire using AACO.

        Args:
            masked_features: Currently observed features (unobserved = 0)
            feature_mask: Boolean mask of observed features
            features: Original features (unused, for protocol compatibility)
            label: True label (unused, AACO doesn't cheat)
            selection_mask: Which selections have been made (unused for DirectUnmasker)
            feature_shape: Shape of features excluding batch dim

        Returns:
            AFAAction tensor with shape (*batch, 1):
            - 0 = stop acquiring
            - 1 to N = 1-indexed feature to acquire (for DirectUnmasker)
        """
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Flatten features if needed (AACO works on flat features)
        batch_shape = feature_mask.shape[:-
                                         1] if feature_shape is None else feature_mask.shape[: -len(feature_shape)]
        if feature_shape is not None:
            n_features = feature_shape.numel()
            masked_features = masked_features.view(-1, n_features)
            feature_mask = feature_mask.view(-1, n_features)
        else:
            n_features = masked_features.shape[-1]
            masked_features = masked_features.view(-1, n_features)
            feature_mask = feature_mask.view(-1, n_features)

        batch_size = masked_features.shape[0]
        selections = []

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i].bool()
            n_acquired = obs_mask.sum().item()

            # Check if no features observed (should use initializer first)
            if n_acquired == 0:
                # This shouldn't happen if using an AFAInitializer
                # But handle gracefully by stopping
                logger.warning(
                    "AACO select() called with no features observed. "
                    "Use an AFAInitializer for initial feature selection."
                )
                selections.append(0)
                continue

            # Hard budget mode: check if at limit
            if self._hard_budget is not None:
                if n_acquired >= self._hard_budget:
                    selections.append(0)  # Stop - reached budget
                    continue
                force_acquisition = True
            else:
                force_acquisition = False

            # Get next feature from AACO oracle
            next_feature = self.aaco_oracle.select_next_feature(
                x_obs,
                obs_mask,
                instance_idx=i,
                force_acquisition=force_acquisition,
            )

            if next_feature is None:
                # Soft budget: oracle decided to stop
                selections.append(0)
            else:
                # Convert 0-indexed to 1-indexed for DirectUnmasker
                selections.append(next_feature + 1)

        selection_tensor = torch.tensor(
            selections, dtype=torch.long, device=original_device
        )

        # Reshape to match batch shape
        return selection_tensor.view(*batch_shape, 1)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Make prediction using the oracle's classifier.

        Args:
            masked_features: Currently observed features
            feature_mask: Boolean mask of observed features
            features: Original features (unused)
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            Class probabilities with shape (*batch, n_classes)
        """
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
                -1, masked_features.shape[-1])
            feature_mask_flat = feature_mask.view(-1, feature_mask.shape[-1])

        batch_size = masked_features_flat.shape[0]

        # Get n_classes from oracle's training data
        if self.aaco_oracle.y_train is not None:
            n_classes = self.aaco_oracle.y_train.shape[-1]
        else:
            n_classes = 10  # fallback

        predictions = torch.zeros(batch_size, n_classes, device=self._device)

        for i in range(batch_size):
            pred = self.aaco_oracle.predict_with_mask(
                masked_features_flat[i],
                feature_mask_flat[i].bool(),
            )
            predictions[i] = pred[:n_classes]

        # Reshape and return
        output_shape = batch_shape + (n_classes,)
        return predictions.view(*output_shape).to(original_device)

    @property
    @override
    def cost_param(self) -> float:
        """Return the current acquisition cost."""
        return self.aaco_oracle.acquisition_cost

    @override
    def set_cost_param(self, cost_param: float) -> None:
        """Set the acquisition cost (for soft budget mode)."""
        self.aaco_oracle.acquisition_cost = cost_param

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        """AACO has a builtin classifier for predictions."""
        return True

    @override
    def save(self, path: Path) -> None:
        """Save method state to disk."""
        oracle_state = {
            "k_neighbors": self.aaco_oracle.k_neighbors,
            "acquisition_cost": self.aaco_oracle.acquisition_cost,
            "hide_val": self.aaco_oracle.hide_val,
            "dataset_name": self.dataset_name,
            "hard_budget": self._hard_budget,
            "classifier_bundle_path": str(self.classifier_bundle_path)
            if self.classifier_bundle_path is not None
            else None,
            "X_train": self.aaco_oracle.X_train.cpu()
            if self.aaco_oracle.X_train is not None
            else None,
            "y_train": self.aaco_oracle.y_train.cpu()
            if self.aaco_oracle.y_train is not None
            else None,
        }
        torch.save(oracle_state, path / f"aaco_oracle_{self.dataset_name}.pt")
        logger.info(f"Saved AACO method to {path}")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device | None = None) -> Self:
        """Load method from disk."""
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        oracle_files = list(path.glob("aaco_oracle_*.pt"))
        if not oracle_files:
            msg = f"No AACO oracle files found in {path}"
            raise FileNotFoundError(msg)

        oracle_state = torch.load(oracle_files[0], map_location=device)

        aaco_oracle = AACOOracle(
            k_neighbors=oracle_state["k_neighbors"],
            acquisition_cost=oracle_state["acquisition_cost"],
            hide_val=oracle_state["hide_val"],
            device=device,
        )

        # Restore fitted state
        if oracle_state["X_train"] is not None and oracle_state["y_train"] is not None:
            aaco_oracle.fit(
                oracle_state["X_train"].to(device),
                oracle_state["y_train"].to(device),
            )

        # Get classifier path from saved state
        classifier_bundle_path = oracle_state.get("classifier_bundle_path")
        if classifier_bundle_path is not None:
            classifier_bundle_path = Path(classifier_bundle_path)

        method = cls(
            aaco_oracle=aaco_oracle,
            dataset_name=oracle_state["dataset_name"],
            classifier_bundle_path=classifier_bundle_path,
            _device=device,
            _hard_budget=oracle_state.get("hard_budget"),
        )

        logger.info(f"Loaded AACO method from {path}")
        return method

    @override
    def to(self, device: torch.device) -> Self:
        """Move method to device."""
        self._device = device
        self.aaco_oracle = self.aaco_oracle.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """Get current device."""
        return self._device


def create_aaco_method(
    dataset_name: str,
    k_neighbors: int = 5,
    acquisition_cost: float = 0.05,
    hide_val: float = 0.0,  # Use 0 for consistency with MLP training
    hard_budget: int | None = None,
    classifier_bundle_path: Path | None = None,
    device: torch.device | None = None,
) -> AACOAFAMethod:
    """
    Factory function to create AACO method.

    Args:
        dataset_name: Name of dataset (for mask generator config)
        k_neighbors: Number of neighbors for KNN
        acquisition_cost: Cost per feature acquisition (soft budget)
        hide_val: Value to use for unobserved features
        hard_budget: Max features to acquire (None = soft budget)
        classifier_bundle_path: Path to pre-trained classifier bundle
        device: Device to use

    Returns:
        Configured AACOAFAMethod instance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aaco_oracle = AACOOracle(
        k_neighbors=k_neighbors,
        acquisition_cost=acquisition_cost,
        hide_val=hide_val,
        device=device,
    )

    return AACOAFAMethod(
        aaco_oracle=aaco_oracle,
        dataset_name=dataset_name,
        classifier_bundle_path=classifier_bundle_path,
        _device=device,
        _hard_budget=hard_budget,
    )
