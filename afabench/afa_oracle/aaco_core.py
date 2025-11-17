import torch
import logging
import torch.nn.functional as F

from afabench.common.utils import get_class_probabilities, load_classifier
from afabench.afa_oracle.mask_generator import random_mask_generator

log = logging.getLogger(__name__)


def get_knn(
    X_train,
    X_query,
    masks,
    num_neighbors,
    instance_idx=0,
    exclude_instance=True,
):
    """
    Their exact K-NN implementation

    Args:
        X_train: N x d Train Instances
        X_query: 1 x d Query Instances
        masks: d x R binary masks to try
        num_neighbors: Number of neighbors (k)

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


def load_mask_generator(dataset_name, input_dim):
    """Their exact mask generator loading logic"""
    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower in [
        "cube",
        "diabetes",
        "mnist",
        "fashionmnist",
        "physionet",
        "miniboone",
        "afacontext",
        "bank_marketing",
        "ckd",
        "actg",
    ]:
        # Paper shows this works nearly as well as 10,000 (for MNIST)
        return random_mask_generator(100, input_dim, 100)
    raise ValueError("Unsupported dataset for mask generation")


def get_initial_feature(dataset_name, n_features):
    """Their exact initial feature selection logic"""
    dataset_name = dataset_name.lower()

    if dataset_name == "cube":
        return 6
    if dataset_name in ["mnist", "fashionmnist"]:
        return 100
    # Default: select middle feature
    return n_features // 2


class AACOOracle:
    def __init__(
        self,
        k_neighbors=5,
        acquisition_cost=0.05,
        hide_val=10.0,
        dataset_name="cube",
        split="1",
        device=None,
    ):
        self.k_neighbors = k_neighbors
        self.acquisition_cost = acquisition_cost
        self.hide_val = hide_val
        self.dataset_name = dataset_name
        self.split = split
        self.classifier_artifact_name = (
            f"masked_mlp_classifier-{dataset_name}_split_{split}/"
        )
        self.classifier = None
        self.mask_generator = None
        self.X_train = None
        self.y_train = None
        self.device = device
        self.class_weights = None

    def fit(self, X_train, y_train):
        """Fit the oracle on training data"""
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        train_class_probabilities = get_class_probabilities(self.y_train)
        self.class_weights = len(train_class_probabilities) / (
            len(train_class_probabilities) * train_class_probabilities
        )

        # Load exact classifier - pass device parameter
        input_dim = X_train.shape[1]

        self.classifier = load_classifier(
            self.classifier_artifact_name, device=self.device
        )

        # Load exact mask generator
        self.mask_generator = load_mask_generator(self.dataset_name, input_dim)

        log.info(f"AACO Oracle fitted for {self.dataset_name}")
        log.info(f"Training data: {X_train.shape}")

    def select_next_feature(self, x_observed, observed_mask, instance_idx=0):
        """
        Feature selection logic
        Args:
            x_observed: 1D tensor of observed features
            observed_mask: 1D boolean tensor indicating which features are observed
            instance_idx: index of current instance (for KNN exclusion)

        Returns:
            next_feature: int index of next feature to acquire, or None if terminate

        """
        if self.classifier is None:
            raise ValueError("Oracle must be fitted first")

        feature_count = len(x_observed)

        # Convert to format: current mask as 1 x d
        mask_curr = observed_mask.float().unsqueeze(0)

        # Check if this is the first feature selection
        if not observed_mask.any():
            # Select initial feature deterministically (their approach)
            initial_feature = get_initial_feature(
                self.dataset_name, feature_count
            )
            return initial_feature

        # Get the nearest neighbors based on the observed feature mask
        x_query = x_observed.unsqueeze(0).to(self.device)  # 1 x d
        idx_nn = get_knn(
            self.X_train,
            x_query,
            mask_curr.T,
            self.k_neighbors,
            instance_idx,
            True,
        ).squeeze()

        # Generate random masks and get the next set of possible masks
        new_masks = self.mask_generator(mask_curr).to(self.device)  # R x d
        mask = torch.maximum(
            new_masks, mask_curr.repeat(new_masks.shape[0], 1)
        )
        mask[0] = mask_curr  # Ensure the current mask is included

        # Get only unique masks
        mask = mask.unique(dim=0)
        n_masks_updated = mask.shape[0]

        # Predictions based on the classifier
        x_rep = self.X_train[idx_nn].repeat(n_masks_updated, 1)
        mask_rep = torch.repeat_interleave(mask, self.k_neighbors, 0)
        idx_nn_rep = idx_nn.repeat(n_masks_updated)

        # Apply masking with hide_val
        x_masked = torch.mul(x_rep, mask_rep) - (1 - mask_rep) * self.hide_val
        x_with_mask = torch.cat([x_masked, mask_rep], -1)

        y_pred = (
            self.classifier.predict_logits(x_with_mask)
            if hasattr(self.classifier, "predict_logits")
            else self.classifier(
                x_with_mask, torch.tensor([0], device=self.device)
            )
        )

        # Compute loss - ensure tensors are on same device
        y_true_rep = self.y_train[idx_nn_rep]
        if len(y_true_rep.shape) > 1 and y_true_rep.shape[1] > 1:
            # One-hot encoded
            y_true_labels = y_true_rep.argmax(dim=1)
        else:
            y_true_labels = y_true_rep

        # Ensure y_pred is on the correct device before loss computation
        if hasattr(y_pred, "to"):
            y_pred = y_pred.to(self.device)
        else:
            y_pred = torch.tensor(y_pred, device=self.device)

        # Ensure y_true_labels is on the correct device
        y_true_labels = y_true_labels.to(self.device)

        loss = F.cross_entropy(
            y_pred, y_true_labels, weight=self.class_weights, reduction="none"
        )

        # Reshape loss to (n_masks, k_neighbors) and average over neighbors
        loss = loss.view(n_masks_updated, self.k_neighbors)
        avg_loss = loss.mean(dim=1)

        # Add acquisition cost
        mask_costs = torch.sum(mask, dim=1) - torch.sum(mask_curr)
        total_cost = avg_loss + self.acquisition_cost * mask_costs

        # Find the mask with minimum cost
        best_mask_idx = total_cost.argmin()
        best_mask = mask[best_mask_idx]

        # REMOVED: termination check - always select best mask regardless of cost improvement
        # Find which new features to acquire
        new_features = (best_mask - mask_curr.squeeze()).nonzero(
            as_tuple=True
        )[0]

        if len(new_features) == 0:
            # # If best mask equals current mask, force selection of cheapest unobserved feature
            # unobserved = (~observed_mask).nonzero(as_tuple=True)[0]
            # return unobserved[0].item() if len(unobserved) > 0 else None

            # ACO optimization returned u(x_o, o) = ∅ - no acquisitions worth the cost
            return None  # Stop action - let method terminate

        if len(new_features) == 1:
            return new_features[0].item()
        # If multiple features, select the one with lowest individual loss
        individual_losses = []
        for feat in new_features:
            temp_mask = mask_curr.clone()
            temp_mask[0, feat] = 1
            # Quick loss computation for this feature
            individual_losses.append(avg_loss[best_mask_idx])  # Simplified

        best_feat_idx = torch.tensor(individual_losses).argmin()
        return new_features[best_feat_idx].item()

    def to(self, device):
        """Move oracle to device"""
        self.device = device
        if self.X_train is not None:
            self.X_train = self.X_train.to(device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(device)
        return self
