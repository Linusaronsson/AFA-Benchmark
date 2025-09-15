from abc import ABC, abstractmethod

from jaxtyping import Float
from torch import Tensor, nn

from common.custom_types import FeatureMask, Logits, MaskedFeatures

type Embedding = Float[Tensor, "*batch embedding_size"]


class Embedder(nn.Module, ABC):
    """An Embedder converts feature values and feature indices (1 if a feature is observed, 0 if not) to an embedding."""

    @abstractmethod
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding: ...

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        return super().__call__(masked_features, feature_mask)


class EmbeddingClassifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, embedding: Embedding) -> Logits: ...

    def __call__(self, embedding: Embedding) -> Logits:
        return super().__call__(embedding)
