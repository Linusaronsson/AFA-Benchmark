import torch
import logging
import torch.nn.functional as F

from afabench.common.utils import get_class_frequencies
from afabench.afa_oracle.mask_generator import random_mask_generator

log = logging.getLogger(__name__)


def get_knn(
    X_train: torch.Tensor,
    X_query: torch.Tensor,
    masks: torch.Tensor,
    num_neighbors: int,
    instance_idx: int = 0,
    exclude_instance: bool = True,
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
    """
    X_train_squared = X_train**2
    X_query_squared = X_query**2
    X_train_X_query = X_train * X_query
    dist_squared = (
        torch.matmul(X_train_squared, masks)
        - 2.0 * torch.matmul(X_train_X_query, masks)
        + torch.matmul(X_query_squared, masks)
    )

    if exclude_instance:
        idx_topk = torch.topk(
            dist_squared, num_neighbors + 1, dim=0, largest=False
        )[1]
        return idx_topk[idx_topk != instance_idx][:num_neighbors]
    return torch.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]


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
        device: torch.device | None = None,
    ):
        self.k_neighbors = k_neighbors
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.classifier = None
        self.mask_generator = None
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.device = device or torch.device("cpu")
        self.class_weights: torch.Tensor | None = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor):
        """
        Fit the oracle on training data.

        Args:
            X_train: Training features (N x d)
            y_train: Training labels (N x n_classes), one-hot encoded
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

        train_class_probabilities = get_class_frequencies(self.y_train)
        self.class_weights = len(train_class_probabilities) / (
            len(train_class_probabilities) * train_class_probabilities
        )

        input_dim = X_train.shape[1]
        self.mask_generator = load_mask_generator(input_dim)

        log.info(f"Training data: {X_train.shape}")

    def set_classifier(self, classifier):
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
        return self

    def select_next_feature(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        instance_idx: int = 0,
        force_acquisition: bool = False,
    ) -> int | None:
        """
        Select the next feature to acquire.

        Args:
            x_observed: 1D tensor of observed features (with unobserved = 0 or hide_val)
            observed_mask: 1D boolean tensor indicating which features are observed
            instance_idx: Index of current instance (for KNN exclusion)

            force_acquisition: If True, must return a feature (hard budget mode).
                               If False, can return None to indicate stopping (soft budget).

        Returns:
            Index of next feature to acquire (0-indexed), or None if should stop.

        Note:
            This method assumes at least one feature is already observed.
            Initial feature selection should be handled by an AFAInitializer.
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )

        assert self.X_train is not None and self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )

        feature_count = len(x_observed)
        mask_curr = observed_mask.float().unsqueeze(0)

        assert observed_mask.any(), (
            "No features observed. Use an AFAInitializer (e.g., "
            "AACODefaultInitializer) to select the initial feature."
        )

        # Get nearest neighbors based on currently observed features
        x_query = x_observed.unsqueeze(0).to(self.device)
        idx_nn = get_knn(
            self.X_train,
            x_query,
            mask_curr.T.to(self.device),
            self.k_neighbors,
            instance_idx,
            exclude_instance=True,
        ).squeeze()

        #  Generate candidate masks for Monte Carlo approximation
        new_masks = self.mask_generator(mask_curr).to(self.device)
        mask = torch.maximum(
            new_masks, mask_curr.repeat(new_masks.shape[0], 1).to(self.device)
        )

        if not force_acquisition:
            # Include current mask as option (allows stopping)
            mask[0] = mask_curr
        # else: don't include current mask - must acquire something

        mask = mask.unique(dim=0)

        # Compute expected loss for each candidate mask
        n_masks = mask.shape[0]
        X_nn = self.X_train[idx_nn]  # k x d
        y_nn = self.y_train[idx_nn]  # k x n_classes

        # Prepare masked inputs for classifier
        X_masked = X_nn.unsqueeze(0).repeat(n_masks, 1, 1)  # n_masks x k x d
        mask_expanded = mask.unsqueeze(1).repeat(1, self.k_neighbors, 1)

        # Apply masking
        X_masked = X_masked * mask_expanded + \
            self.hide_val * (1 - mask_expanded)

        # Get predictions for all masks and neighbors
        X_flat = X_masked.view(-1, feature_count)
        mask_flat = mask_expanded.view(-1, feature_count)

        with torch.no_grad():
            logits = self.classifier(X_flat, mask_flat)
            probs = F.softmax(logits, dim=-1)

        probs = probs.view(n_masks, self.k_neighbors, -1)

        # Compute weighted cross-entropy loss
        y_nn_expanded = y_nn.unsqueeze(0).repeat(n_masks, 1, 1)
        losses = -torch.sum(y_nn_expanded * torch.log(probs + 1e-10), dim=-1)

        # Weight by class weights if available
        if self.class_weights is not None:
            class_indices = y_nn.argmax(dim=-1)
            weights = self.class_weights[class_indices]
            losses = losses * weights.unsqueeze(0)

        # Average over neighbors
        expected_losses = losses.mean(dim=1)

        # Add acquisition cost penalty
        n_new_features = mask.sum(dim=1) - mask_curr.sum()
        costs = expected_losses + self.acquisition_cost * \
            n_new_features.to(self.device)

        # Select best mask
        best_idx = costs.argmin().item()
        best_mask = mask[best_idx]

        # Find the new feature(s) to acquire
        new_features = (best_mask.bool() & ~observed_mask.to(self.device)).nonzero(
            as_tuple=True
        )[0]

        if len(new_features) == 0:
            # Best action is to stop (only possible if force_acquisition=False)
            return None

        # Return the first new feature to acquire
        return int(new_features[0].item())

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
            logits = self.classifier(x_input, mask)
            probs = F.softmax(logits, dim=-1)

        return probs.squeeze(0)
