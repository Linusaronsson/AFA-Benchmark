from typing import Protocol

from torch import Tensor


class VisionBackbone(Protocol):
    embed_dim: int
    num_features: int

    def forward_features(self, x: Tensor) -> Tensor: ...

    def forward_head(
        self, x: Tensor, *, pre_logits: bool = False
    ) -> Tensor: ...
