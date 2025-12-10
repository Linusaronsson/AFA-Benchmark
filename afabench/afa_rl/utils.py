from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import wandb
from jaxtyping import Float, Integer
from tensordict import TensorDictBase
from torch import Tensor, nn
from torch.nn import functional as F

from afabench.afa_rl.custom_types import FeatureSet
from afabench.common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelection,
    FeatureMask,
    Features,
    MaskedFeatures,
)


def remove_module_prefix(state_dict: dict[str, Any]):  # noqa: ANN201
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def get_feature_set(
    masked_features: torch.Tensor, feature_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert partially observed features and their indices to the set representation, together with the number of observed features for each batch element, expected by the embedder."""
    batch_size, n_features = masked_features.shape

    # Preallocate the result tensors
    feature_set = torch.zeros(
        (batch_size, n_features, n_features + 1),
        dtype=torch.float32,
        device=masked_features.device,
    )
    lengths = torch.zeros(
        (batch_size,), dtype=torch.int64, device=masked_features.device
    )

    # Iterate over the batch
    for i in range(batch_size):
        # Get the indices of the observed features
        observed_feature_indices = feature_mask[i].nonzero(as_tuple=True)[0]
        lengths[i] = len(observed_feature_indices)

        # Create a mask for the observed features
        mask = torch.zeros(
            n_features, dtype=torch.bool, device=masked_features.device
        )
        mask[observed_feature_indices] = 1

        # Update feature_set: first column with masked feature values
        feature_set[i, : len(observed_feature_indices), 0] = masked_features[
            i, observed_feature_indices
        ]

        # Update the rest of the feature_set with one-hot encoded indices
        feature_set[i, : len(observed_feature_indices), 1:] = F.one_hot(
            observed_feature_indices, num_classes=n_features
        ).float()

    return feature_set, lengths


def shuffle_feature_set(
    feature_set: FeatureSet, lengths: Tensor
) -> FeatureSet:
    """Shuffle a feature set."""
    shuffled_feature_set = torch.zeros_like(
        feature_set, device=feature_set.device
    )
    for i in range(feature_set.shape[0]):
        shuffled_feature_set[i, : lengths[i]] = feature_set[
            i,
            torch.randperm(int(lengths[i].item()), device=feature_set.device),
        ]

    return shuffled_feature_set


def get_image_feature_set(
    masked_image: MaskedFeatures,
    feature_mask: FeatureMask,
    image_shape: tuple[int, int],
) -> FeatureSet:
    """
    Convert a partially observed image and the indices to a feature set.

    The output has shape (batch, HxW, 3), where each row in the third dimension is
    [value, row, col] for observed pixels in each batch. Unobserved pixels are
    included with value=0, row=col=0 and are padded to the end of the output.
    """
    batch_size, flat_dim = masked_image.shape
    h, w = image_shape

    # Reshape into (B, H, W)
    masked_image = masked_image.view(batch_size, h, w)
    feature_mask = feature_mask.view(batch_size, h, w)

    # Prepare a tensor to hold the result
    result = []

    for i in range(batch_size):
        # Get observed pixel indices (n_features, 2)
        obs_indices = torch.nonzero(
            feature_mask[i], as_tuple=False
        )  # shape (n_features, 2)
        values = masked_image[i, obs_indices[:, 0], obs_indices[:, 1]]

        # Adjust the row and column indices to 1-based indexing
        obs_indices = obs_indices + 1  # Make row and col 1-based

        # Stack [value, row, col] for observed pixels
        feature_set = torch.stack(
            [values, obs_indices[:, 0].float(), obs_indices[:, 1].float()],
            dim=1,
        )

        # Create a tensor of size (HxW, 3) by padding unobserved pixels with zeros
        padded_set = torch.zeros((h * w, 3), device=masked_image.device)

        # For unobserved pixels, set value=0 and row=col=0
        padded_set[:, 0] = 0  # value = 0
        padded_set[:, 1] = 0  # row = 0
        padded_set[:, 2] = 0  # col = 0

        # Copy observed pixels into the result
        padded_set[: feature_set.size(0), :] = feature_set

        result.append(padded_set)

    return torch.stack(result)


def get_2d_identity(
    feature_mask: FeatureMask,
    image_shape: tuple[int, int],
) -> Integer[Tensor, "*batch n_features 2"]:
    """Return the coordinates for each observed feature (as given by the feature mask) but with (0, 0) for unobserved features."""
    batch_size, flat_dim = feature_mask.shape
    h, w = image_shape

    # Reshape into (B, H, W)
    feature_mask = feature_mask.view(batch_size, h, w)

    # Prepare a tensor to hold the result
    result = []

    for i in range(batch_size):
        # Get observed pixel indices (n_features, 2)
        obs_indices = torch.nonzero(
            feature_mask[i], as_tuple=False
        )  # shape (n_features, 2)

        # Adjust the row and column indices to 1-based indexing
        obs_indices = obs_indices + 1  # Make row and col 1-based

        # Stack [row, col] for observed pixels
        coords = torch.stack(
            [obs_indices[:, 0].float(), obs_indices[:, 1].float()], dim=1
        )

        # Create a tensor of size (HxW, 2) by padding unobserved pixels with zeros
        padded_set = torch.zeros((h * w, 2), device=feature_mask.device)

        # For unobserved pixels, set row=col=0
        padded_set[:, 0] = 0  # row = 0
        padded_set[:, 1] = 0  # col = 0

        # Copy observed pixels into the result
        padded_set[: coords.size(0), :] = coords

        result.append(padded_set)

    return torch.stack(result)


# def get_1D_identity(
#     feature_mask: FeatureMask,
# ) -> Integer[Tensor, "*batch n_features n_features"]:
#     """
#     Converts a feature mask to a onehot representation for each feature, but with all zeros for unobserved features.
#     """
#     batch_size, feature_size = feature_mask.shape

#     feature_set = torch.zeros(
#         (batch_size, feature_size, feature_size),
#         device=feature_mask.device,
#         dtype=torch.float,
#     )

#     # Generate one-hot indices
#     one_hot_indices = torch.eye(
#         feature_size, device=feature_mask.device
#     )  # shape: (feature_size, feature_size)

#     # Expand one-hot vectors only for observed features
#     mask_expanded = feature_mask.unsqueeze(-1).expand(
#         -1, -1, feature_size
#     )  # shape: (batch, features, feature_size)
#     feature_set[:, :, :] = (
#         one_hot_indices.unsqueeze(0).expand(batch_size, -1, -1) * mask_expanded
#     )

#     return feature_set


def get_1d_identity(
    feature_mask: FeatureMask,
) -> Integer[Tensor, "*batch n_features n_features"]:
    """Convert a feature mask to a onehot representation for each feature, but with all zeros for unobserved features."""
    batch_size, feature_size = feature_mask.shape

    # One-hot vectors for each feature (feature_size, feature_size)
    one_hot = torch.eye(feature_size, device=feature_mask.device)

    # Expand to match batch: (batch, feature_size, feature_size)
    one_hot_batch = one_hot.unsqueeze(0).expand(batch_size, -1, -1)

    # Expand mask: (batch, feature_size, 1)
    mask_expanded = feature_mask.unsqueeze(-1)

    # Multiply to mask out unobserved features
    feature_set = one_hot_batch * mask_expanded

    return feature_set


def floatwrapfn(f: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function to convert all arguments to float before calling it."""

    def wrapper(*args):  # noqa: ANN002, ANN202
        return f(*[arg.float() for arg in args])

    return wrapper


def resample_invalid_actions(actions, action_mask, action_values):  # noqa: ANN001, ANN201
    resampled_actions = actions.clone()

    # Find invalid actions
    invalid_mask = ~action_mask[torch.arange(actions.shape[0]), actions]

    # Select the highest-value valid action for each invalid case
    valid_action_values = action_values.clone()
    valid_action_values[~action_mask] = float(
        "-inf"
    )  # Mask out invalid actions
    best_valid_actions = valid_action_values.argmax(dim=-1)

    resampled_actions[invalid_mask] = best_valid_actions[invalid_mask]

    return resampled_actions


def get_sequential_module_norm(module: nn.Sequential):  # noqa: ANN201
    """Calculate the average norm of all the linear layers in a sequential module."""
    weight_norms = [
        layer.weight.norm() for layer in module if isinstance(layer, nn.Linear)
    ]
    return torch.mean(torch.stack(weight_norms)).item()


def mask_data(
    features: Features, p: float
) -> tuple[MaskedFeatures, FeatureMask, Tensor]:
    """
    Given features, mask them with probability p.

    Returns the masked features, feature mask, and which rows have at least one feature.

    Args:
        features: The features to mask.
        p: The probability of each feature being masked (0).

    """
    feature_mask = torch.rand(features.shape, device=features.device) > p
    masked_features = features * feature_mask.float()

    return masked_features, feature_mask, feature_mask.any(dim=-1)


def recursive_defaultdict():  # noqa: ANN201
    return defaultdict(recursive_defaultdict)


def to_regular_dict(d: dict) -> dict:  # pyright: ignore[reportMissingTypeArgument]
    if isinstance(d, defaultdict):
        d = {k: to_regular_dict(v) for k, v in d.items()}
    return d


def check_masked_classifier_performance(
    afa_predict_fn: AFAPredictFn,
    dataset: AFADataset,
    class_weights: Float[Tensor, "n_classes"],  # noqa: ARG001, F821
) -> None:
    """Check that a masked classifier has decent performance on the dataset."""
    # model_device = next(masked_classifier.parameters()).device
    # Calculate average accuracy over the whole dataset
    with torch.no_grad():
        # Get the features and labels from the dataset
        features, labels = dataset.get_all_data()

        # Allow masked classifier to look at *all* features
        # masked_features_all = features
        # feature_mask_all = torch.ones_like(
        #     features,
        #     dtype=torch.bool,
        #     device=device
        # )
        # logits_all = masked_classifier(masked_features_all, feature_mask_all)
        # accuracy_all = (
        #     (logits_all.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
        # )

        # # Same thing, but only allow masked classifier to look at 50% of the features
        # feature_mask_half = torch.randint(0, 2, feature_mask_all.shape, device=device)
        # masked_features_half = features.clone()
        # masked_features_half[feature_mask_half == 0] = 0
        # logits_half = masked_classifier(masked_features_half, feature_mask_half)
        # accuracy_half = (
        #     (logits_half.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean()
        # )

        # Only allow classifier to look at the "optimal" feature according to AFAContext
        feature_mask_optimal = torch.zeros_like(
            features,
            dtype=torch.bool,
        )
        feature_mask_optimal[:, 0] = 1
        for i in range(feature_mask_optimal.shape[0]):
            context = features[i, 0].int().item()
            feature_mask_optimal[i, context * 3 + 1 : context * 3 + 4] = 1
        masked_features_optimal = features.clone()
        masked_features_optimal[feature_mask_optimal == 0] = 0
        probs_optimal = afa_predict_fn(
            masked_features_optimal, feature_mask_optimal
        )
        accuracy_optimal = (
            (probs_optimal.argmax(dim=-1) == labels.argmax(dim=-1))
            .float()
            .mean()
        )

        # Calculate the loss for the 50% feature case. Useful for setting acquisition costs
        # loss_half = F.cross_entropy(logits_half, labels.float(), weight=class_weights)

        # print(
        #     f"Masked classifier accuracy with all features: {accuracy_all.item() * 100:.2f}%"
        # )
        # print(
        #     f"Masked classifier accuracy with 50% features: {accuracy_half.item() * 100:.2f}%"
        # )
        print(
            f"Masked classifier accuracy with optimal features: {accuracy_optimal.item() * 100:.2f }%"
        )
        # print(f"Average cross-entropy loss with 50% features: {loss_half.item():.4f}")


def afacontext_optimal_selection(
    masked_features: MaskedFeatures, feature_mask: FeatureMask
) -> AFASelection:
    selection = torch.full(
        (masked_features.shape[0],),
        -1,
        dtype=torch.int64,
        device=masked_features.device,
    )

    # Case 1: no features are selected yet, select the first feature
    case1_mask = feature_mask.sum(dim=-1) == 0
    selection[case1_mask] = 0

    # Case 2: between 1 and 3 features are selected, select the next feature based on the context
    case2_mask = (feature_mask.sum(dim=-1) > 0) & (
        feature_mask.sum(dim=-1) < 4
    )
    case2_start_idx = masked_features[:, 0].int() * 3 + 1
    case2_end_idx = case2_start_idx + 3
    for i in range(masked_features.size(0)):
        if case2_mask[i]:
            start = int(case2_start_idx[i].item())
            end = int(case2_end_idx[i].item())

            # Find the first unselected feature in the range
            for j in range(start, end):
                if feature_mask[i, j] == 0:
                    selection[i] = j
                    break

    # Case 3: 4 or more features are selected, select a random unselected feature
    case3_mask = feature_mask.sum(dim=-1) >= 4
    for i in range(masked_features.size(0)):
        if case3_mask[i]:
            unselected_features = (~feature_mask[i]).nonzero(as_tuple=True)[0]
            if unselected_features.numel() > 0:
                selection[i] = unselected_features[0]

    return selection


def cube_simple_optimal_selection(
    masked_features: MaskedFeatures, feature_mask: FeatureMask
) -> AFASelection:
    selection = torch.full(
        (masked_features.shape[0],),
        -1,
        dtype=torch.int64,
        device=masked_features.device,
    )

    # Always select features 0, 2, and 4 in order if they are not already selected
    for i in range(masked_features.size(0)):
        for feature_idx in [0, 2, 4]:
            if feature_mask[i, feature_idx] == 0:
                selection[i] = feature_idx
                break

    return selection


def cube_simple_optimal_selection_wrapper(
    td: TensorDictBase,
) -> TensorDictBase:
    td["action"] = cube_simple_optimal_selection(
        td["masked_features"], td["feature_mask"]
    )
    return td


def cube_simple_worst_selection(
    masked_features: MaskedFeatures, feature_mask: FeatureMask
) -> AFASelection:
    selection = torch.full(
        (masked_features.shape[0],),
        -1,
        dtype=torch.int64,
        device=masked_features.device,
    )

    # Always select features 1, 3, and 5 in order if they are not already selected
    for i in range(masked_features.size(0)):
        for feature_idx in [1, 3, 5]:
            if feature_mask[i, feature_idx] == 0:
                selection[i] = feature_idx
                break

    return selection


def cube_simple_worst_selection_wrapper(td: TensorDictBase) -> TensorDictBase:
    td["action"] = cube_simple_worst_selection(
        td["masked_features"], td["feature_mask"]
    )
    return td


def module_norm(module: nn.Module) -> float:
    # Aggregate all parameters from the module and compute the norm
    return torch.norm(
        torch.cat([p.view(-1) for p in module.parameters()]), p=2
    ).item()  # L2 norm (Euclidean norm)


def get_eval_metrics(
    eval_tds: list[TensorDictBase], afa_predict_fn: AFAPredictFn
) -> dict[str, Any]:
    """Return a dictionary of metrics from a list of tensordicts collected from a torchrl environment."""
    # Assumption: td has batch size (n_agents, episode_len)
    assert eval_tds[0].batch_dims == 2, (
        "Expected eval_tds to have batch dims (n_agents, episode_len)"
    )
    eval_metrics = {}
    eval_metrics["reward_sum"] = 0.0
    eval_metrics["actions"] = []
    n_correct_samples = 0
    total_samples = 0
    for td in eval_tds:
        eval_metrics["reward_sum"] += td["next", "reward"].sum()
        eval_metrics["actions"].extend(td["action"].flatten().tolist())
        # Check whether prediction is correct
        td_end = td[:, -1]
        probs = afa_predict_fn(
            masked_features=td_end["next", "masked_features"],
            feature_mask=td_end["next", "feature_mask"],
            label=None,
            feature_shape=None,
        )
        pred_labels = probs.argmax(dim=-1)
        true_labels = td_end["label"].argmax(dim=-1)
        correct_predictions = (pred_labels == true_labels).sum().item()
        n_correct_samples += correct_predictions
        total_samples += pred_labels.numel()
    eval_metrics["accuracy"] = n_correct_samples / total_samples
    batch_size = eval_tds[0].batch_size
    eval_metrics["reward_sum"] /= len(eval_tds) * batch_size.numel()
    eval_metrics["actions"] = wandb.Histogram(
        eval_metrics["actions"], num_bins=20
    )
    return eval_metrics


def weighted_cross_entropy(
    input_probs: torch.Tensor,
    target_probs: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
    reduction: str = "none",  # or 'none', 'sum'
) -> torch.Tensor:
    """
    Compute weighted cross-entropy between predicted and target distributions.

    Args:
        input_probs: (batch_size, num_classes), predicted class probabilities.
        target_probs: (batch_size, num_classes), target probability distributions (e.g., one-hot or soft).
        weights: (num_classes,), per-class weights.
        eps: small value to avoid log(0).
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Tensor of shape (batch_size,) if reduction='none',
        else scalar loss.

    """
    # assert input_probs.ndim == 2
    # assert target_probs.ndim == 2
    # assert weights.ndim == 1
    log_input = torch.log(input_probs + eps)  # (batch_size, num_classes)
    weighted_ce = (
        -target_probs * log_input * weights.unsqueeze(0)
    )  # (batch_size, num_classes)
    loss_per_sample = weighted_ce.sum(dim=1)  # (batch_size,)

    if reduction == "mean":
        return loss_per_sample.mean()
    if reduction == "sum":
        return loss_per_sample.sum()
    if reduction == "none":
        return loss_per_sample
    msg = f"Invalid reduction: {reduction}"
    raise ValueError(msg)


str_to_activation_class_mapping: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "relu": nn.ReLU,
    "Tanh": nn.Tanh,
    "tanh": nn.Tanh,
}
