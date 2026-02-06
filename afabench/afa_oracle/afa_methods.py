from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Self, cast, final, override

import torch

from afabench.afa_oracle.aaco_core import AACOOracle
from afabench.afa_oracle.utils import (
    flatten_for_aaco,
    uses_patch_selection,
)
from afabench.common.bundle import load_bundle
from afabench.common.custom_types import (
    AFAClassifier,
    AFAMethod,
    AFAAction,
    AFAUnmasker,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.registry import get_class

logger = logging.getLogger(__name__)


@dataclass
@final
class AACOAFAMethod(AFAMethod):
    """
    AACO-based Active Feature Acquisition method.

    This method uses the Acquisition Conditioned Oracle (AACO) from
    Valancius et al. 2024 to select features for acquisition.

    The method is designed to work with:
    - AFAInitializer (e.g., RandomInitializer) for initial feature selection
    - AFAUnmasker (e.g., DirectUnmasker) for action-to-mask mapping

    Supports both soft budget (cost-based stopping) and hard budget
    (forced acquisition, stopping handled by the evaluator) modes.
    """

    aaco_oracle: AACOOracle
    dataset_name: str
    classifier_bundle_path: Path | None = (
        None  # Path to trained classifier bundle
    )
    force_acquisition: bool = False  # Hard budget mode uses forced acquisition
    unmasker_class_name: str | None = None
    unmasker_kwargs: dict[str, Any] | None = None
    _selection_costs: torch.Tensor | None = None
    _device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    _selection_size: int | None = None  # None = feature-level selections
    _exclude_instance: bool = False
    _selection_to_feature_mask_cache: dict[tuple[int, ...], torch.Tensor] = (
        field(default_factory=dict, init=False, repr=False)
    )
    _unmasker: AFAUnmasker | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Move oracle to device after initialization and load classifier if path provided."""
        self.aaco_oracle = self.aaco_oracle.to(self._device)
        if self._selection_costs is not None:
            self._selection_costs = torch.as_tensor(
                self._selection_costs, dtype=torch.float32, device=self._device
            )

        # Load classifier from bundle if path provided
        if self.classifier_bundle_path is not None:
            classifier, _ = load_bundle(
                self.classifier_bundle_path, device=self._device
            )
            classifier = cast(AFAClassifier, cast(object, classifier))
            self.aaco_oracle.set_classifier(classifier)
            logger.info(
                f"Loaded classifier from {self.classifier_bundle_path}"
            )

    def _get_unmasker(self) -> AFAUnmasker | None:
        if self._unmasker is not None:
            return self._unmasker
        if self.unmasker_class_name is None:
            return None
        kwargs = (
            self.unmasker_kwargs if self.unmasker_kwargs is not None else {}
        )
        cls = get_class(self.unmasker_class_name)
        self._unmasker = cast("AFAUnmasker", cls(**kwargs))
        return self._unmasker

    def _get_selection_to_feature_mask(
        self,
        feature_shape: torch.Size,
        selection_size: int,
    ) -> torch.Tensor:
        cache_key = (selection_size, *feature_shape)
        if cache_key in self._selection_to_feature_mask_cache:
            return self._selection_to_feature_mask_cache[cache_key]

        unmasker = self._get_unmasker()
        assert unmasker is not None, (
            "Selection-space AACO requires unmasker metadata in the bundle."
        )

        batch_size = selection_size
        zero_features = torch.zeros(
            (batch_size, *feature_shape),
            dtype=torch.float32,
            device=self._device,
        )
        zero_feature_mask = torch.zeros(
            (batch_size, *feature_shape),
            dtype=torch.bool,
            device=self._device,
        )
        zero_selection_mask = torch.zeros(
            (batch_size, selection_size),
            dtype=torch.bool,
            device=self._device,
        )
        selection_indices = torch.arange(
            selection_size, device=self._device, dtype=torch.long
        ).unsqueeze(-1)

        selection_feature_mask = unmasker.unmask(
            masked_features=zero_features,
            feature_mask=zero_feature_mask,
            features=zero_features,
            afa_selection=selection_indices,
            selection_mask=zero_selection_mask,
            label=None,
            feature_shape=feature_shape,
        ).view(selection_size, -1)
        selection_feature_mask = selection_feature_mask.bool()
        self._selection_to_feature_mask_cache[cache_key] = (
            selection_feature_mask
        )
        return selection_feature_mask

    def set_exclude_instance(self, exclude_instance: bool) -> None:
        """Set whether to exclude the query instance from KNN results."""
        self._exclude_instance = exclude_instance

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
        Select next feature to acquire using AACO.

        Args:
            masked_features: Currently observed features (unobserved = 0)
            feature_mask: Boolean mask of observed features
            selection_mask: Which selections have been made (unused for DirectUnmasker)
            label: True label (unused, AACO doesn't cheat)
            feature_shape: Shape of features excluding batch dim

        Returns:
            AFAAction tensor with shape (*batch, 1):
            - 0 = stop acquiring
            - 1 to N = 1-indexed feature to acquire (for DirectUnmasker)
        """
        with torch.no_grad():
            original_device = masked_features.device
            masked_features = masked_features.to(self._device)
            feature_mask = feature_mask.to(self._device)

            # Flatten features if needed (AACO works on flat features)
            masked_features, feature_mask, batch_shape = flatten_for_aaco(
                masked_features, feature_mask, feature_shape
            )
            n_features = masked_features.shape[-1]

            batch_size = masked_features.shape[0]
            selections = []
            selection_size = (
                selection_mask.shape[-1]
                if selection_mask is not None
                else self._selection_size
            )
            selection_mask_flat = (
                selection_mask.view(-1, selection_size).bool()
                if selection_mask is not None and selection_size is not None
                else None
            )
            use_selection_space = (
                selection_mask_flat is not None
                and selection_size is not None
                and selection_size != n_features
                and (
                    feature_shape is None
                    or not uses_patch_selection(selection_size, feature_shape)
                )
                and self.unmasker_class_name is not None
            )

            selection_to_feature_mask = None
            if use_selection_space and feature_shape is not None:
                selection_to_feature_mask = (
                    self._get_selection_to_feature_mask(
                        feature_shape=feature_shape,
                        selection_size=selection_size,
                    )
                )

            oracle_selection_costs = None
            if (
                self._selection_costs is not None
                and selection_size is not None
                and len(self._selection_costs) == selection_size
            ):
                oracle_selection_costs = self._selection_costs

            for i in range(batch_size):
                x_obs = masked_features[i]
                obs_mask = feature_mask[i].bool()

                if (
                    use_selection_space
                    and selection_to_feature_mask is not None
                    and selection_mask_flat is not None
                ):
                    next_selection = self.aaco_oracle.select_next_selection(
                        x_observed=x_obs,
                        observed_mask=obs_mask,
                        selection_mask=selection_mask_flat[i],
                        selection_to_feature_mask=selection_to_feature_mask,
                        selection_costs=oracle_selection_costs,
                        instance_idx=i,
                        force_acquisition=self.force_acquisition,
                        exclude_instance=self._exclude_instance,
                    )
                    if next_selection is None:
                        selections.append(0)
                    else:
                        selections.append(next_selection + 1)
                    continue

                # Default path: feature-level (or patch-level) oracle.
                next_feature = self.aaco_oracle.select_next_feature(
                    x_obs,
                    obs_mask,
                    instance_idx=i,
                    force_acquisition=self.force_acquisition,
                    exclude_instance=self._exclude_instance,
                    feature_shape=feature_shape,
                    selection_size=selection_size,
                    selection_costs=oracle_selection_costs,
                )
                if next_feature is None:
                    selections.append(0)
                else:
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
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Make prediction using the oracle's classifier.

        Args:
            masked_features: Currently observed features
            feature_mask: Boolean mask of observed features
            label: True label (unused)
            feature_shape: Shape of features excluding batch dim

        Returns:
            Class probabilities with shape (*batch, n_classes)
        """
        with torch.no_grad():
            original_device = masked_features.device
            masked_features = masked_features.to(self._device)
            feature_mask = feature_mask.to(self._device)

            masked_features_flat, feature_mask_flat, batch_shape = (
                flatten_for_aaco(masked_features, feature_mask, feature_shape)
            )

            batch_size = masked_features_flat.shape[0]

            # Get n_classes from oracle's training data
            if self.aaco_oracle.y_train is not None:
                n_classes = self.aaco_oracle.y_train.shape[-1]
            else:
                n_classes = 10  # fallback

            predictions = torch.zeros(
                batch_size, n_classes, device=self._device
            )

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

    @override
    def save(self, path: Path) -> None:
        """Save method state to disk."""
        unmasker_kwargs = (
            dict(self.unmasker_kwargs)
            if self.unmasker_kwargs is not None
            else None
        )
        oracle_state = {
            "k_neighbors": self.aaco_oracle.k_neighbors,
            "acquisition_cost": self.aaco_oracle.acquisition_cost,
            "hide_val": self.aaco_oracle.hide_val,
            "dataset_name": self.dataset_name,
            "force_acquisition": self.force_acquisition,
            "selection_size": self._selection_size,
            "unmasker_class_name": self.unmasker_class_name,
            "unmasker_kwargs": unmasker_kwargs,
            "selection_costs": (
                self._selection_costs.detach().cpu()
                if self._selection_costs is not None
                else None
            ),
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
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        oracle_files = list(path.glob("aaco_oracle_*.pt"))
        if not oracle_files:
            msg = f"No AACO oracle files found in {path}"
            raise FileNotFoundError(msg)

        oracle_state = torch.load(
            oracle_files[0], map_location=device, weights_only=False
        )

        aaco_oracle = AACOOracle(
            k_neighbors=oracle_state["k_neighbors"],
            acquisition_cost=oracle_state["acquisition_cost"],
            hide_val=oracle_state["hide_val"],
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

        # Get classifier path from saved state
        classifier_bundle_path = oracle_state.get("classifier_bundle_path")
        if classifier_bundle_path is not None:
            classifier_bundle_path = Path(classifier_bundle_path)

        force_acquisition = oracle_state.get("force_acquisition")
        if force_acquisition is None:
            force_acquisition = oracle_state.get("hard_budget") is not None

        method = cls(
            aaco_oracle=aaco_oracle,
            dataset_name=oracle_state["dataset_name"],
            classifier_bundle_path=classifier_bundle_path,
            force_acquisition=force_acquisition,
            unmasker_class_name=oracle_state.get("unmasker_class_name"),
            unmasker_kwargs=oracle_state.get("unmasker_kwargs"),
            _selection_costs=oracle_state.get("selection_costs"),
            _device=device,
            _selection_size=oracle_state.get("selection_size"),
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
    *,
    force_acquisition: bool = False,
    selection_size: int | None = None,
    unmasker_class_name: str | None = None,
    unmasker_kwargs: dict[str, Any] | None = None,
    selection_costs: torch.Tensor | None = None,
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
        force_acquisition: If True, never stop early (hard budget mode)
        selection_size: Optional selection space size for patch-based unmaskers
        unmasker_class_name: Unmasker class name for grouped selection spaces
        unmasker_kwargs: Unmasker constructor kwargs
        selection_costs: Optional per-selection costs
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
        force_acquisition=force_acquisition,
        unmasker_class_name=unmasker_class_name,
        unmasker_kwargs=unmasker_kwargs,
        _selection_costs=selection_costs,
        _device=device,
        _selection_size=selection_size,
    )
