import logging

import torch
import torch.nn.functional as F

from afabench.components.methods.oracle.aaco.mask_generator import (
    RandomMaskGenerator,
    random_mask_generator,
)
from afabench.components.methods.oracle.aaco.utils import (
    ensure_probabilities,
    get_patch_dimensions,
    uses_patch_selection,
)
from afabench.core.types import AFAClassifier
from afabench.core.utils import get_class_frequencies

logger = logging.getLogger(__name__)

MISSINGNESS_OBJECTIVES = {"support_aware", "doubly_robust"}


def get_knn(
    X_train: torch.Tensor,  # noqa: N803
    X_query: torch.Tensor,  # noqa: N803
    masks: torch.Tensor,
    num_neighbors: int,
    instance_idx: int = 0,
    exclude_instance: bool = True,  # noqa: FBT002
    batch_size: int = 1000,
    train_observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    K-NN implementation from the AACO paper.

    (https://github.com/lupalab/aaco/blob/3b2316661651699d11e904e9c5911c175e8b2fdc/src/aaco_rollout.py#L103C1-L103C4).

    Args:
        X_train: N x d Train Instances
        X_query: 1 x d Query Instances
        masks: d x R binary masks to try
        num_neighbors: Number of neighbors (k)
        instance_idx: Index of current instance (for exclusion)
        exclude_instance: Whether to exclude the query instance from results
        batch_size: Number of training samples to process at once (for memory)
    """
    N = X_train.shape[0]
    masks = masks.to(X_train.device)
    if train_observed_mask is None:
        X_query_squared = X_query**2
        query_term = torch.matmul(X_query_squared, masks)
        dist_squared_chunks = []
        for i in range(0, N, batch_size):
            X_batch = X_train[i : i + batch_size]
            X_batch_squared = X_batch**2
            X_batch_X_query = X_batch * X_query
            dist_batch = (
                torch.matmul(X_batch_squared, masks)
                - 2.0 * torch.matmul(X_batch_X_query, masks)
                + query_term
            )
            dist_squared_chunks.append(dist_batch)
        dist_squared = torch.cat(dist_squared_chunks, dim=0)
    else:
        train_observed_mask = train_observed_mask.to(X_train.device).bool()
        query_masks = masks.bool()
        dist_squared_chunks = []
        for i in range(0, N, batch_size):
            X_batch = X_train[i : i + batch_size]
            observed_batch = train_observed_mask[i : i + batch_size]
            squared_diff = (X_batch - X_query).pow(2)
            shared = observed_batch.unsqueeze(-1) & query_masks.unsqueeze(0)
            shared_counts = shared.sum(dim=1)
            dist_batch = (squared_diff.unsqueeze(-1) * shared.float()).sum(
                dim=1
            ) / shared_counts.clamp_min(1)
            dist_squared_chunks.append(
                dist_batch.masked_fill(shared_counts == 0, torch.inf)
            )
        dist_squared = torch.cat(dist_squared_chunks, dim=0)

    k = num_neighbors + int(exclude_instance)
    idx_topk = torch.topk(dist_squared, k, dim=0, largest=False)[1]
    if not exclude_instance:
        return idx_topk
    return idx_topk[idx_topk != instance_idx][:num_neighbors]


def load_mask_generator(input_dim: int) -> RandomMaskGenerator:
    """Their exact mask generator loading logic."""
    # Paper shows this works nearly as well as 10,000 (for MNIST)
    return random_mask_generator(100, input_dim, 100)


class AACOOracle:
    """
    Acquisition Conditioned Oracle for non-greedy active feature acquisition.

    This oracle implements the AACO algorithm from Valancius et al. 2024.
    (https://proceedings.mlr.press/v235/valancius24a.html)

    It selects features by optimizing a non-greedy objective that considers
    future acquisition costs.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        acquisition_cost: float = 0.05,
        hide_val: float = 0.0,  # Use 0 for consistency with MLP training
        missingness_objective: str = "support_aware",
        dr_min_propensity: float = 1e-3,
        dr_max_weight: float | None = 20.0,
        device: torch.device | None = None,
    ):
        if missingness_objective not in MISSINGNESS_OBJECTIVES:
            msg = (
                "missingness_objective must be one of "
                f"{sorted(MISSINGNESS_OBJECTIVES)}."
            )
            raise ValueError(msg)
        if not 0.0 < dr_min_propensity <= 1.0:
            msg = "dr_min_propensity must be in (0, 1]."
            raise ValueError(msg)
        if dr_max_weight is not None and dr_max_weight <= 0.0:
            msg = "dr_max_weight must be positive when provided."
            raise ValueError(msg)
        self.k_neighbors: int = k_neighbors
        self.acquisition_cost: float = acquisition_cost
        self.hide_val: float = hide_val
        self.missingness_objective: str = missingness_objective
        self.dr_min_propensity: float = dr_min_propensity
        self.dr_max_weight: float | None = dr_max_weight
        self.classifier: AFAClassifier | None = None
        self.mask_generator: RandomMaskGenerator | None = None
        self._patch_mask_generators: dict[int, RandomMaskGenerator] = {}
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.train_observed_mask: torch.Tensor | None = None
        self.marginal_observation_probabilities: torch.Tensor | None = None
        self.device: torch.device = device or torch.device("cpu")
        self.class_weights: torch.Tensor | None = None

    def fit(
        self,
        X_train: torch.Tensor,  # noqa: N803
        y_train: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Fit the oracle on training data.

        Args:
            X_train: Training features (N x d)
            y_train: Training labels (N x n_classes), one-hot encoded
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        if observed_mask is None:
            self.train_observed_mask = None
            self.marginal_observation_probabilities = None
        else:
            observed_mask = observed_mask.to(self.device).bool()
            if observed_mask.shape != self.X_train.shape:
                msg = "observed_mask must have the same shape as X_train."
                raise ValueError(msg)
            self.train_observed_mask = (
                None if observed_mask.all() else observed_mask
            )
            self.marginal_observation_probabilities = (
                None
                if self.train_observed_mask is None
                else self.train_observed_mask.float().mean(dim=0)
            )

        train_class_probabilities = get_class_frequencies(self.y_train)
        self.class_weights = len(train_class_probabilities) / (
            len(train_class_probabilities) * train_class_probabilities
        )

        input_dim = X_train.shape[1]
        self.mask_generator = load_mask_generator(input_dim)

        logger.info(f"Training data: {X_train.shape}")

    def set_classifier(self, classifier: AFAClassifier) -> None:
        """Set the classifier model used by the oracle."""
        self.classifier = classifier

    def to(self, device: torch.device) -> "AACOOracle":
        """Move oracle to device."""
        self.device = device
        if self.X_train is not None:
            self.X_train = self.X_train.to(device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        if self.train_observed_mask is not None:
            self.train_observed_mask = self.train_observed_mask.to(device)
        if self.marginal_observation_probabilities is not None:
            self.marginal_observation_probabilities = (
                self.marginal_observation_probabilities.to(device)
            )
        return self

    def _neighbor_observed_mask(
        self,
        neighbor_indices: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        neighbor_indices = neighbor_indices.reshape(-1)
        if self.train_observed_mask is None:
            return torch.ones(
                (len(neighbor_indices), feature_count),
                dtype=torch.bool,
                device=self.device,
            )
        return self.train_observed_mask[neighbor_indices]

    def _neighbor_losses(
        self,
        neighbor_features: torch.Tensor,
        neighbor_labels: torch.Tensor,
        feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        assert self.classifier is not None
        n_masks, n_neighbors, feature_count = feature_masks.shape
        mask_float = feature_masks.float()
        masked = neighbor_features.unsqueeze(0).expand(n_masks, -1, -1)
        masked = masked * mask_float + self.hide_val * (1 - mask_float)
        logits = self.classifier(
            masked.reshape(-1, feature_count),
            mask_float.reshape(-1, feature_count),
            feature_shape=torch.Size([feature_count]),
        )
        probabilities = ensure_probabilities(logits).view(
            n_masks,
            n_neighbors,
            -1,
        )
        expanded_labels = neighbor_labels.unsqueeze(0).expand(
            n_masks,
            -1,
            -1,
        )
        losses = -torch.sum(
            expanded_labels * torch.log(probabilities + 1e-10),
            dim=-1,
        )
        if self.class_weights is not None:
            class_indices = neighbor_labels.argmax(dim=-1)
            losses *= self.class_weights[class_indices].unsqueeze(0)
        return losses

    def _candidate_support_propensities(
        self,
        candidate_feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.marginal_observation_probabilities is None:
            return torch.ones(
                candidate_feature_masks.shape[0],
                device=self.device,
            )
        marginal = self.marginal_observation_probabilities.clamp_min(1e-12)
        log_propensity = (
            candidate_feature_masks.float() * marginal.log().unsqueeze(0)
        ).sum(dim=1)
        return log_propensity.exp()

    def _expected_candidate_losses(
        self,
        candidate_feature_masks: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.X_train is not None
        assert self.y_train is not None
        candidate_feature_masks = candidate_feature_masks.bool()
        neighbor_indices = neighbor_indices.reshape(-1)
        neighbor_features = self.X_train[neighbor_indices]
        neighbor_labels = self.y_train[neighbor_indices]
        observed = self._neighbor_observed_mask(
            neighbor_indices,
            candidate_feature_masks.shape[1],
        )
        overlap_masks = candidate_feature_masks.unsqueeze(1) & observed
        overlap_losses = self._neighbor_losses(
            neighbor_features,
            neighbor_labels,
            overlap_masks,
        )
        if (
            self.missingness_objective != "doubly_robust"
            or self.train_observed_mask is None
        ):
            return overlap_losses.mean(dim=1)

        full_masks = candidate_feature_masks.unsqueeze(1).expand(
            -1,
            len(neighbor_indices),
            -1,
        )
        full_losses = self._neighbor_losses(
            neighbor_features,
            neighbor_labels,
            full_masks,
        )
        fully_supported = (
            (~candidate_feature_masks.unsqueeze(1)) | observed
        ).all(dim=-1)
        inverse_weights = (
            self._candidate_support_propensities(candidate_feature_masks)
            .clamp_min(self.dr_min_propensity)
            .reciprocal()
        )
        if self.dr_max_weight is not None:
            inverse_weights = inverse_weights.clamp_max(self.dr_max_weight)
        baseline = overlap_losses.mean(dim=1, keepdim=True)
        corrected = baseline + fully_supported.float() * (
            inverse_weights.unsqueeze(1) * (full_losses - baseline)
        )
        return corrected.mean(dim=1)

    def select_next_feature(  # noqa: C901, PLR0912, PLR0915
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        instance_idx: int = 0,
        force_acquisition: bool = False,  # noqa: FBT002
        exclude_instance: bool = True,  # noqa: FBT002
        feature_shape: torch.Size | None = None,
        selection_size: int | None = None,
        selection_costs: torch.Tensor | None = None,
        selection_mask: torch.Tensor | None = None,
    ) -> int | None:
        """
        Select the next feature to acquire.

        Args:
            x_observed: 1D tensor of observed features (with unobserved = 0 or hide_val)
            observed_mask: 1D boolean tensor indicating which features are observed
            instance_idx: Index of current instance (for KNN exclusion)

            force_acquisition: If True, must return a feature (hard budget mode).
                               If False, can return None to indicate stopping (soft budget).
            exclude_instance: Whether to exclude instance_idx from KNN results.
            selection_costs: Optional per-selection costs. If provided, this
                             overrides the default unit-cost penalty.
            selection_mask: Optional mask of previously performed selections.
                            When provided, this determines which selections
                            are considered already acquired.

        Returns:
            Index of next feature to acquire (0-indexed), or None if should stop.

        Note:
            Supports o = ∅ (no observed features), consistent with AACO.
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )

        assert self.X_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )

        feature_count = len(x_observed)
        use_patch_selection = uses_patch_selection(
            selection_size, feature_shape
        )
        device = self.device
        observed_feature_mask = observed_mask.to(device).bool()
        observed_feature_mask_row = observed_feature_mask.float().unsqueeze(0)
        mask_width = 0
        n_channels = 0
        height = 0
        width = 0
        patch_h = 0
        patch_w = 0
        patch_mask_generator: RandomMaskGenerator | None = None
        mask_generator: RandomMaskGenerator | None = None

        # Get nearest neighbors based on currently observed features
        x_query = x_observed.unsqueeze(0).to(device)
        idx_nn = get_knn(
            self.X_train,
            x_query,
            observed_feature_mask_row.T.to(device),
            self.k_neighbors,
            instance_idx,
            exclude_instance=exclude_instance,
            train_observed_mask=self.train_observed_mask,
        ).squeeze()

        if use_patch_selection:
            assert feature_shape
            assert selection_size is not None
            n_channels, height, width, patch_h, patch_w = get_patch_dimensions(
                selection_size, feature_shape
            )
            mask_width = int(selection_size**0.5)
            selection_dim = selection_size

            patch_mask_generator = self._patch_mask_generators.get(
                selection_size
            )
            if patch_mask_generator is None:
                patch_mask_generator = random_mask_generator(
                    100, selection_size, 100
                )
                self._patch_mask_generators[selection_size] = (
                    patch_mask_generator
                )
        else:
            selection_dim = feature_count
            mask_generator = self.mask_generator
            assert mask_generator is not None

        if selection_mask is not None:
            current_selection_mask = selection_mask.to(device).bool().view(-1)
            assert len(current_selection_mask) == selection_dim, (
                "selection_mask has incompatible selection dimension."
            )
        elif use_patch_selection:
            observed_mask_2d = observed_feature_mask.view(
                n_channels, height, width
            )
            fm = observed_mask_2d.view(
                n_channels,
                mask_width,
                patch_h,
                mask_width,
                patch_w,
            )
            current_selection_mask = fm.any(dim=(0, 2, 4)).view(-1).bool()
        else:
            current_selection_mask = observed_feature_mask.clone()
        current_selection_mask_row = current_selection_mask.float().unsqueeze(
            0
        )

        if use_patch_selection:
            assert patch_mask_generator is not None
            new_masks = patch_mask_generator(current_selection_mask_row).to(
                device
            )
            candidate_selection_masks = torch.maximum(
                new_masks,
                current_selection_mask_row.repeat(new_masks.shape[0], 1),
            )
        else:
            # Generate candidate masks for Monte Carlo approximation.
            assert mask_generator is not None
            new_masks = mask_generator(current_selection_mask_row).to(device)
            candidate_selection_masks = torch.maximum(
                new_masks,
                current_selection_mask_row.repeat(new_masks.shape[0], 1).to(
                    device
                ),
            )

        if not force_acquisition:
            # Include current selection mask as option (allows stopping).
            candidate_selection_masks[0] = current_selection_mask_row
        # else: don't include current mask - must acquire something
        candidate_selection_masks = candidate_selection_masks.unique(dim=0)

        def _selection_to_feature_mask(
            selection_masks: torch.Tensor,
        ) -> torch.Tensor:
            if not use_patch_selection:
                return selection_masks.bool()
            mask_4d = selection_masks.view(-1, 1, mask_width, mask_width)
            mask_4d = F.interpolate(
                mask_4d.float(),
                scale_factor=(patch_h, patch_w),
                mode="nearest-exact",
            )
            if n_channels > 1:
                mask_4d = mask_4d.expand(-1, n_channels, height, width)
            return mask_4d.reshape(-1, feature_count).bool()

        # Candidate feature masks always include currently observed features.
        candidate_feature_masks = _selection_to_feature_mask(
            candidate_selection_masks
        ) | observed_feature_mask.unsqueeze(0)
        expected_losses = self._expected_candidate_losses(
            candidate_feature_masks,
            idx_nn,
        )

        # Add acquisition cost penalty.
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
            new_selection_mask = (
                candidate_selection_masks.bool()
                & ~current_selection_mask.unsqueeze(0)
            )
            acquisition_penalty = (
                new_selection_mask.float() * selection_costs.unsqueeze(0)
            ).sum(dim=1)
        else:
            acquisition_penalty = (
                candidate_selection_masks.sum(dim=1)
                - current_selection_mask_row.sum()
            )
        costs = expected_losses + self.acquisition_cost * acquisition_penalty

        # Select best mask
        best_idx = costs.argmin().item()
        best_selection_mask = candidate_selection_masks[best_idx].bool()

        # Find the new selection(s) to acquire.
        new_features = (best_selection_mask & ~current_selection_mask).nonzero(
            as_tuple=True
        )[0]

        if len(new_features) == 0:
            if force_acquisition:
                unobserved = (~current_selection_mask).nonzero(as_tuple=True)[
                    0
                ]
                if len(unobserved) > 0:
                    return int(unobserved[0].item())
            # Best action is to stop (only possible if force_acquisition=False)
            return None

        if len(new_features) == 1:
            return int(new_features[0].item())

        # Tie-break: select the single feature in the chosen subset
        # that most reduces expected loss when added alone.
        ordering_selection_masks = current_selection_mask_row.repeat(
            len(new_features), 1
        )
        ordering_selection_masks[
            torch.arange(
                len(new_features), device=ordering_selection_masks.device
            ),
            new_features,
        ] = 1

        ordering_feature_masks = _selection_to_feature_mask(
            ordering_selection_masks
        ) | observed_feature_mask.unsqueeze(0)

        avg_loss = self._expected_candidate_losses(
            ordering_feature_masks,
            idx_nn,
        )
        best_feature_idx = avg_loss.argmin().item()
        return int(new_features[best_feature_idx].item())

    def select_next_selection(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        selection_mask: torch.Tensor,
        selection_to_feature_mask: torch.Tensor,
        selection_costs: torch.Tensor | None = None,
        instance_idx: int = 0,
        force_acquisition: bool = False,  # noqa: FBT002
        exclude_instance: bool = True,  # noqa: FBT002
    ) -> int | None:
        """
        Select the next **selection** (not feature) to acquire.

        This is used for unmaskers where selections are not equivalent to
        individual features (e.g. grouped context selections).
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )
        assert self.X_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert selection_to_feature_mask.ndim == 2, (
            "selection_to_feature_mask must be 2D: (n_selections, n_features)"
        )

        device = self.device
        observed_mask = observed_mask.to(device).bool()
        selection_mask = selection_mask.to(device).bool()
        selection_to_feature_mask = selection_to_feature_mask.to(device).bool()

        assert selection_to_feature_mask.shape[1] == observed_mask.numel(), (
            "selection_to_feature_mask has incompatible feature dimension."
        )

        available_selection_indices = (~selection_mask).nonzero(as_tuple=True)[
            0
        ]
        if len(available_selection_indices) == 0:
            return None

        candidate_selection_masks = selection_to_feature_mask[
            available_selection_indices
        ]
        base_feature_mask = observed_mask.unsqueeze(0)
        candidate_feature_masks = (
            base_feature_mask | candidate_selection_masks
        ).float()

        candidate_indices = available_selection_indices.clone()
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
            candidate_costs = selection_costs[candidate_indices]
        else:
            newly_observed = (
                candidate_feature_masks.bool() & ~base_feature_mask
            )
            candidate_costs = newly_observed.float().sum(dim=1)

        if not force_acquisition:
            candidate_feature_masks = torch.cat(
                [base_feature_mask.float(), candidate_feature_masks], dim=0
            )
            candidate_costs = torch.cat(
                [torch.zeros(1, device=device), candidate_costs], dim=0
            )
            candidate_indices = torch.cat(
                [
                    torch.tensor([-1], device=device, dtype=torch.long),
                    candidate_indices,
                ],
                dim=0,
            )

        # Get nearest neighbors based on currently observed features.
        x_query = x_observed.unsqueeze(0).to(device)
        idx_nn = get_knn(
            self.X_train,
            x_query,
            observed_mask.float().unsqueeze(-1),
            self.k_neighbors,
            instance_idx,
            exclude_instance=exclude_instance,
            train_observed_mask=self.train_observed_mask,
        ).squeeze()
        if idx_nn.ndim == 0:
            idx_nn = idx_nn.unsqueeze(0)

        expected_losses = self._expected_candidate_losses(
            candidate_feature_masks,
            idx_nn,
        )

        costs = expected_losses + self.acquisition_cost * candidate_costs
        best_candidate_idx = int(costs.argmin().item())
        selected = int(candidate_indices[best_candidate_idx].item())
        if selected < 0:
            return None
        return selected

    def predict_with_mask(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make prediction given observed features.

        Args:
            x_observed: 1D tensor of observed features
            observed_mask: 1D boolean tensor indicating which features are observed

        Returns:
            Class probabilities (n_classes,)
        """
        if self.classifier is None:
            msg = "Oracle must have a classifier set."
            raise ValueError(msg)

        x_masked = x_observed.unsqueeze(0).to(self.device)
        mask = observed_mask.float().unsqueeze(0).to(self.device)

        # Apply masking
        x_input = x_masked * mask + self.hide_val * (1 - mask)

        with torch.no_grad():
            feature_shape = torch.Size([x_input.shape[1]])
            logits = self.classifier(
                x_input, mask, feature_shape=feature_shape
            )
            probs = ensure_probabilities(logits)

        return probs.squeeze(0)
