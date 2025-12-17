from collections.abc import Callable
from typing import Protocol

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)

type State = Float[
    Tensor, "*batch state_size"
]  # A state is a concatenation of feature values and feature indices
type FeatureSet = Float[
    Tensor, "*batch n_features set_size"
    # A feature set is the set version of State. Each element-index tuple becomes a vector.
]


# class MaskedClassifier(Protocol):
#     """
#     A function that returns class logits given a set of features and a feature mask.
#     """
#     def __call__(self, masked_features: MaskedFeatures, feature_mask: FeatureMask) -> Logits: ...


class AFADatasetFn(Protocol):
    """A dataset that returns new batched samples in the same format as AFADataset. move_on determines whether the dataset should return different samples next time the function is called."""

    def __call__(
        self, batch_size: torch.Size, *, move_on: bool = True
    ) -> tuple[Features, Label]: ...


type NaiveIdentity = Integer[Tensor, "*batch n_features naive_identity_size"]
type NaiveIdentityFn = Callable[[FeatureMask], NaiveIdentity]


# class NNMaskedClassifier(nn.Module, MaskedClassifier, ABC):
#     """
#     A neural network classifier that takes a set of features and a feature mask as input and returns class logits.
#     """
#
#     @abstractmethod
#     def forward(
#         self, masked_features: MaskedFeatures, feature_mask: FeatureMask
#     ) -> Logits:
#         pass
#
#     def __call__(
#         self, masked_features: MaskedFeatures, feature_mask: FeatureMask
#     ) -> Logits:
#         """
#         Calls the forward method and returns the class logits.
#         """
#         return self.forward(masked_features, feature_mask)


# A reward function will in general depend on
# - masked features at time t
# - feature mask (the features that have been observed so far) at time t
# - masked features at time x_{t+1}
# - feature mask (the features that have been observed so far) at time t+1
# - the selection (the feature that was selected) at time t
# - the ground truth features
# - the ground truth label

type AFAReward = Float[Tensor, "*batch 1"]
type AFARewardFn = Callable[
    [
        MaskedFeatures,  # current masked features
        FeatureMask,  # current feature mask
        SelectionMask,  # current selection mask
        MaskedFeatures,  # new masked features
        FeatureMask,  # new feature mask
        SelectionMask,  # new selection mask
        AFASelection,  # which selection we made
        Features,  # true features
        Label,  # true label
        Bool[Tensor, "*batch 1"],  # done key
    ],
    AFAReward,
]
