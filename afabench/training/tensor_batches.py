"""Vectorized in-memory batches with DataLoader-compatible indexing."""

from collections.abc import Sequence
from typing import cast, final, override

from torch import Tensor
from torch.utils.data import Dataset


@final
class TensorBatchDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        *tensors: Tensor,
    ) -> None:
        if not tensors or any(
            len(tensor) != len(tensors[0]) for tensor in tensors
        ):
            message = (
                "all tensors must have the same non-zero leading dimension"
            )
            raise ValueError(message)
        self.tensors = tensors

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

    def __getitems__(self, indices: Sequence[int]) -> tuple[Tensor, ...]:
        return tuple(tensor[list(indices)] for tensor in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors[0])


def passthrough_batch(batch: object) -> tuple[Tensor, ...]:
    """Accept a batch already materialized by ``__getitems__``."""
    return cast("tuple[Tensor, ...]", batch)
