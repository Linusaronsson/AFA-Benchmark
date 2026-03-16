from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from afabench.afa_oracle.mask_generator import random_mask_generator
from afabench.afa_oracle.utils import (
    ensure_probabilities,
    get_patch_dimensions,
    uses_patch_selection,
)
from afabench.common.utils import get_class_frequencies

logger = logging.getLogger(__name__)

MISSINGNESS_OBJECTIVES = {"mask_aware", "doubly_robust"}


def get_knn(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    masks: torch.Tensor,
    num_neighbors: int,
    instance_idx: int = 0,
    exclude_instance: bool = True,
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
        train_observed_mask: Optional training observed mask (N x d). If
            provided, distances are computed over shared observed dimensions.
    """
    N = X_train.shape[0]
    masks = masks.to(X_train.device)

    if train_observed_mask is None:
        X_query_squared = X_query**2
        query_term = torch.matmul(X_query_squared, masks)  # 1 x R

        # Process in batches to avoid OOM on large datasets
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
        shared_mask = masks.bool()
        train_observed_mask = train_observed_mask.to(X_train.device).bool()

        dist_squared_chunks = []
        for i in range(0, N, batch_size):
            X_batch = X_train[i : i + batch_size]
            observed_batch = train_observed_mask[i : i + batch_size]

            squared_diff = (X_batch - X_query).pow(2)
            shared = observed_batch.unsqueeze(-1) & shared_mask.unsqueeze(0)
            shared_counts = shared.sum(dim=1)

            weighted_dist = (squared_diff.unsqueeze(-1) * shared.float()).sum(
                dim=1
            )
            dist_batch = weighted_dist / shared_counts.clamp_min(1).float()
            dist_batch = dist_batch.masked_fill(
                shared_counts == 0, float("inf")
            )
            dist_squared_chunks.append(dist_batch)
        dist_squared = torch.cat(dist_squared_chunks, dim=0)

    k = num_neighbors + int(exclude_instance)
    idx_topk = torch.topk(dist_squared, k, dim=0, largest=False)[1]
    if not exclude_instance:
        return idx_topk
    return idx_topk[idx_topk != instance_idx][:num_neighbors]


def load_mask_generator(input_dim: int):
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
        missingness_objective: str = "mask_aware",
        dr_min_propensity: float = 1e-3,
        dr_max_weight: float | None = 20.0,
        device: torch.device | None = None,
    ):
        if missingness_objective not in MISSINGNESS_OBJECTIVES:
            msg = (
                "missingness_objective must be one of "
                f"{sorted(MISSINGNESS_OBJECTIVES)}, got "
                f"{missingness_objective!r}."
            )
            raise ValueError(msg)
        if dr_min_propensity <= 0.0 or dr_min_propensity > 1.0:
            msg = (
                "dr_min_propensity must lie in (0, 1], got "
                f"{dr_min_propensity}."
            )
            raise ValueError(msg)
        if dr_max_weight is not None and dr_max_weight <= 0.0:
            msg = (
                "dr_max_weight must be positive when provided, got "
                f"{dr_max_weight}."
            )
            raise ValueError(msg)
        self.k_neighbors = k_neighbors
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.missingness_objective = missingness_objective
        self.dr_min_propensity = dr_min_propensity
        self.dr_max_weight = dr_max_weight
        self.classifier = None
        self.mask_generator = None
        self._patch_mask_generators: dict[int, random_mask_generator] = {}
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.train_observed_mask: torch.Tensor | None = None
        self.marginal_observation_probabilities: torch.Tensor | None = None
        self.device = device or torch.device("cpu")
        self.class_weights: torch.Tensor | None = None

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Fit the oracle on training data.

        Args:
            X_train: Training features (N x d)
            y_train: Training labels (N x n_classes), one-hot encoded
            observed_mask: Optional mask (N x d) where True means observed in
                retrospective training data. If None, assume fully observed.
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        if observed_mask is None:
            self.train_observed_mask = None
            self.marginal_observation_probabilities = None
        else:
            observed_mask = observed_mask.to(self.device).bool()
            if observed_mask.shape != self.X_train.shape:
                msg = (
                    "observed_mask must have the same shape as X_train. "
                    f"Got {observed_mask.shape} and {self.X_train.shape}."
                )
                raise ValueError(msg)
            self.train_observed_mask = (
                None if bool(observed_mask.all().item()) else observed_mask
            )
            if self.train_observed_mask is None:
                self.marginal_observation_probabilities = None
            else:
                self.marginal_observation_probabilities = (
                    self.train_observed_mask.float().mean(dim=0)
                )

        train_class_probabilities = get_class_frequencies(self.y_train)
        self.class_weights = len(train_class_probabilities) / (
            len(train_class_probabilities) * train_class_probabilities
        )

        input_dim = X_train.shape[1]
        self.mask_generator = load_mask_generator(input_dim)

        logger.info("Training data: %s", X_train.shape)

    def set_classifier(self, classifier: object) -> None:
        """Set the classifier model used by the oracle."""
        self.classifier = classifier

    def to(self, device: torch.device) -> AACOOracle:
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

    def _get_neighbor_observed_mask(
        self,
        idx_nn: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        if idx_nn.ndim == 0:
            idx_nn = idx_nn.unsqueeze(0)
        if self.train_observed_mask is None:
            return torch.ones(
                (idx_nn.numel(), feature_count),
                dtype=torch.bool,
                device=self.device,
            )
        return self.train_observed_mask[idx_nn].bool()

    def _compute_neighbor_losses(
        self,
        X_nn: torch.Tensor,
        y_nn: torch.Tensor,
        feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate classifier loss for each candidate mask-neighbor pair."""
        n_masks, n_neighbors, feature_count = feature_masks.shape
        mask_float = feature_masks.float()
        X_masked = X_nn.unsqueeze(0).expand(n_masks, -1, -1)
        X_masked = X_masked * mask_float + self.hide_val * (1 - mask_float)

        X_flat = X_masked.view(-1, feature_count)
        mask_flat = mask_float.view(-1, feature_count)
        with torch.no_grad():
            flat_shape = torch.Size([feature_count])
            logits = self.classifier(
                X_flat, mask_flat, feature_shape=flat_shape
            )
            probs = ensure_probabilities(logits)

        probs = probs.view(n_masks, n_neighbors, -1)
        y_nn_expanded = y_nn.unsqueeze(0).expand(n_masks, -1, -1)
        losses = -torch.sum(y_nn_expanded * torch.log(probs + 1e-10), dim=-1)

        if self.class_weights is not None:
            class_indices = y_nn.argmax(dim=-1)
            weights = self.class_weights[class_indices]
            losses = losses * weights.unsqueeze(0)
        return losses

    def _estimate_candidate_support_propensities(
        self,
        candidate_feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate feature-support propensities via marginal observation rates."""
        if self.marginal_observation_probabilities is None:
            return torch.ones(
                candidate_feature_masks.shape[0],
                device=self.device,
                dtype=torch.float32,
            )

        candidate_feature_masks = candidate_feature_masks.bool()
        if not candidate_feature_masks.any():
            return torch.ones(
                candidate_feature_masks.shape[0],
                device=self.device,
                dtype=torch.float32,
            )

        marginal_probs = self.marginal_observation_probabilities.clamp_min(
            1e-12
        )
        log_propensity = (
            candidate_feature_masks.float() * marginal_probs.log().unsqueeze(0)
        ).sum(dim=1)
        return torch.exp(log_propensity)

    def _compute_expected_losses(
        self,
        candidate_feature_masks: torch.Tensor,
        idx_nn: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        """
        Score each candidate subset using overlap-only or doubly robust loss.

        `mask_aware` uses the loss on the overlap between a candidate subset and
        each neighbor's observed support. `doubly_robust` keeps that overlap
        loss as an augmentation baseline and adds an inverse-propensity
        correction when the full candidate subset is supported.
        """
        candidate_feature_masks = candidate_feature_masks.to(
            self.device
        ).bool()
        X_nn = self.X_train[idx_nn]
        y_nn = self.y_train[idx_nn]
        nn_observed_mask = self._get_neighbor_observed_mask(
            idx_nn, feature_count
        )

        augmentation_masks = candidate_feature_masks.unsqueeze(
            1
        ) & nn_observed_mask.unsqueeze(0)
        overlap_losses = self._compute_neighbor_losses(
            X_nn,
            y_nn,
            augmentation_masks,
        )
        if (
            self.missingness_objective != "doubly_robust"
            or self.train_observed_mask is None
        ):
            return overlap_losses.mean(dim=1)

        full_masks = candidate_feature_masks.unsqueeze(1).expand(
            -1,
            nn_observed_mask.shape[0],
            -1,
        )
        full_losses = self._compute_neighbor_losses(
            X_nn,
            y_nn,
            full_masks,
        )
        support_available = (
            (~candidate_feature_masks.unsqueeze(1))
            | nn_observed_mask.unsqueeze(0)
        ).all(dim=-1)
        propensities = self._estimate_candidate_support_propensities(
            candidate_feature_masks
        ).clamp_min(self.dr_min_propensity)
        inverse_weights = propensities.reciprocal()
        if self.dr_max_weight is not None:
            inverse_weights = inverse_weights.clamp_max(self.dr_max_weight)

        augmentation_baseline = overlap_losses.mean(dim=1, keepdim=True)
        dr_losses = augmentation_baseline + support_available.float() * (
            inverse_weights.unsqueeze(1)
            * (full_losses - augmentation_baseline)
        )
        return dr_losses.mean(dim=1)

    def select_next_feature(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        instance_idx: int = 0,
        force_acquisition: bool = False,
        exclude_instance: bool = True,
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

        assert self.X_train is not None and self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )

        feature_count = len(x_observed)
        use_patch_selection = uses_patch_selection(
            selection_size, feature_shape
        )
        device = self.device
        observed_feature_mask = observed_mask.to(device).bool()
        observed_feature_mask_row = observed_feature_mask.float().unsqueeze(0)

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
        if idx_nn.ndim == 0:
            idx_nn = idx_nn.unsqueeze(0)

        if use_patch_selection:
            assert feature_shape and selection_size is not None
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
            assert self.mask_generator is not None

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
            new_masks = patch_mask_generator(current_selection_mask_row).to(
                device
            )
            candidate_selection_masks = torch.maximum(
                new_masks,
                current_selection_mask_row.repeat(new_masks.shape[0], 1),
            )
        else:
            # Generate candidate masks for Monte Carlo approximation.
            new_masks = self.mask_generator(current_selection_mask_row).to(
                device
            )
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
        expected_losses = self._compute_expected_losses(
            candidate_feature_masks=candidate_feature_masks,
            idx_nn=idx_nn,
            feature_count=feature_count,
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

        avg_loss = self._compute_expected_losses(
            candidate_feature_masks=ordering_feature_masks,
            idx_nn=idx_nn,
            feature_count=feature_count,
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
        force_acquisition: bool = False,
        exclude_instance: bool = True,
    ) -> int | None:
        """
        Select the next **selection** (not feature) to acquire.

        This is used for unmaskers where selections are not equivalent to
        individual features (e.g. grouped context selections).
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )
        assert self.X_train is not None and self.y_train is not None, (
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

        candidate_selection_feature_masks = selection_to_feature_mask[
            available_selection_indices
        ]
        base_feature_mask = observed_mask.unsqueeze(0)
        candidate_feature_masks = (
            base_feature_mask | candidate_selection_feature_masks
        )

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
                [base_feature_mask, candidate_feature_masks], dim=0
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

        feature_count = observed_mask.numel()
        expected_losses = self._compute_expected_losses(
            candidate_feature_masks=candidate_feature_masks,
            idx_nn=idx_nn,
            feature_count=feature_count,
        )

        costs = expected_losses + self.acquisition_cost * candidate_costs
        best_candidate_idx = costs.argmin().item()
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
