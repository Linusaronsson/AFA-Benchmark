import torch
import logging

from pathlib import Path
from dataclasses import dataclass
from typing import Self, final, override

from afabench.afa_oracle.aaco_core import AACOOracle
from afabench.common.custom_types import (
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
)

logger = logging.getLogger(__name__)


@dataclass
@final
class AACOAFAMethod(AFAMethod):
    aaco_oracle: AACOOracle
    dataset_name: str
    _device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    def __post_init__(self):
        # Move oracle to device
        self.aaco_oracle = self.aaco_oracle.to(self._device)

    @override
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features,
        labels,
    ) -> AFASelection:
        """Select next feature using AACO implementation with logging"""
        # Store original device for return tensor
        original_device = masked_features.device

        # Move inputs to working device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # AACO works on single instances, process batch one by one
        batch_size = masked_features.shape[0]
        selections = []

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i]

            # Get next feature from AACO oracle
            # next_feature = self.aaco_oracle.select_next_feature(
            #     x_obs, obs_mask, instance_idx=i
            # )
            # assert next_feature is not None, (
            #     "AACO oracle must return a valid feature index"
            # )
            # selections.append(next_feature)

            # Get next feature from AACO oracle
            next_feature = self.aaco_oracle.select_next_feature(
                x_obs, obs_mask, instance_idx=i
            )
            # Handle stop action when ACO returns None (u(x_o, o) = ∅)
            if next_feature is None:
                # Return stop action
                selections.append(0)
            else:
                selections.append(next_feature + 1)

        # Return selection tensor on original device
        selection_tensor = torch.tensor(
            selections, dtype=torch.long, device=original_device
        )
        return selection_tensor

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features,
        labels,
    ) -> Label:
        """Make prediction using classifier approach with proper device handling"""
        original_device = masked_features.device

        # Move inputs to working device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]
        n_classes = (
            self.aaco_oracle.y_train.shape[-1]
            if len(self.aaco_oracle.y_train.shape) > 1
            else self.aaco_oracle.y_train.max().item() + 1
        )

        predictions = torch.zeros(batch_size, n_classes, device=self._device)

        for i in range(batch_size):
            x_obs = masked_features[i]
            obs_mask = feature_mask[i]

            x_masked = (
                x_obs * obs_mask.float()
                - (1 - obs_mask.float()) * self.aaco_oracle.hide_val
            )
            x_with_mask = torch.cat([x_masked, obs_mask.float()])

            # Split concatenated features and mask
            n_features = x_with_mask.shape[0] // 2
            x_masked_split = x_with_mask[:n_features]
            mask_split = x_with_mask[n_features:]

            pred = self.aaco_oracle.classifier(
                masked_features=x_masked_split.unsqueeze(0),
                feature_mask=mask_split.unsqueeze(0),
                features=None,
                label=None,
            )
            pred = pred.squeeze()

            # Handle different types of classifier outputs
            if isinstance(pred, torch.Tensor):
                pred = pred.to(self._device)
                if pred.dim() == 0:  # scalar output
                    pred_one_hot = torch.zeros(n_classes, device=self._device)
                    pred_one_hot[pred.long()] = 1.0
                    predictions[i] = pred_one_hot
                else:
                    predictions[i] = pred[:n_classes]  # Ensure correct size
            else:
                # Convert numpy/other to tensor
                pred_tensor = torch.tensor(pred, device=self._device)
                if pred_tensor.dim() == 0:
                    pred_one_hot = torch.zeros(n_classes, device=self._device)
                    pred_one_hot[pred_tensor.long()] = 1.0
                    predictions[i] = pred_one_hot
                else:
                    predictions[i] = pred_tensor[:n_classes]

        # Return predictions on original device
        return predictions.to(original_device)

    @property
    @override
    def cost_param(self) -> float:
        """Return the current acquisition cost"""
        return self.aaco_oracle.acquisition_cost

    @override
    def set_cost_param(self, cost_param: float) -> None:
        """Set the acquisition cost at eval time"""
        self.aaco_oracle.acquisition_cost = cost_param

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @override
    def save(self, path: Path) -> None:
        """Save method state"""
        oracle_state = {
            "k_neighbors": self.aaco_oracle.k_neighbors,
            "acquisition_cost": self.aaco_oracle.acquisition_cost,
            "hide_val": self.aaco_oracle.hide_val,
            "dataset_name": self.dataset_name,
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
    def load(cls, path: Path, device: torch.device = None) -> Self:
        """Load method from saved state"""
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        # Find saved oracle file
        oracle_files = list(path.glob("aaco_oracle_*.pt"))
        if not oracle_files:
            raise FileNotFoundError(f"No AACO oracle files found in {path}")

        oracle_state = torch.load(oracle_files[0], map_location=device)

        # Recreate oracle and method
        aaco_oracle = AACOOracle(
            k_neighbors=oracle_state["k_neighbors"],
            acquisition_cost=oracle_state["acquisition_cost"],
            hide_val=oracle_state["hide_val"],
            dataset_name=oracle_state["dataset_name"],
            device=device,
        )

        # Restore fitted state
        if (
            oracle_state["X_train"] is not None
            and oracle_state["y_train"] is not None
        ):
            aaco_oracle.fit(
                oracle_state["X_train"].to(device),
                oracle_state["y_train"].to(device),
            )

        method = cls(
            aaco_oracle=aaco_oracle,
            dataset_name=oracle_state["dataset_name"],
            _device=device,
        )

        logger.info(f"Loaded AACO method from {path}")
        return method

    @override
    def to(self, device: torch.device) -> Self:
        """Move method to device"""
        self._device = device
        self.aaco_oracle = self.aaco_oracle.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        """Get device"""
        return self._device


def create_aaco_method(
    k_neighbors: int = 5,
    acquisition_cost: float = 0.05,
    hide_val: float = 10.0,
    dataset_name: str = "cube",
    split: str = "1",
    device: torch.device = None,
) -> AACOAFAMethod:
    """Factory function to create AACO method with proper device handling"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    aaco_oracle = AACOOracle(
        k_neighbors=k_neighbors,
        acquisition_cost=acquisition_cost,
        hide_val=hide_val,
        dataset_name=dataset_name,
        split=split,
        device=device,  # Pass device parameter
    )

    return AACOAFAMethod(
        aaco_oracle=aaco_oracle, dataset_name=dataset_name, _device=device
    )
