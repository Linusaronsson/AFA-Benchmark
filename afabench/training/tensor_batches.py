"""In-memory datasets that fetch whole batches with a single gather."""

from collections.abc import Sequence
from typing import cast, final, override

from torch import Tensor
from torch.utils.data import Dataset


@final
class TensorBatchDataset(Dataset[tuple[Tensor, ...]]):
    """
    Index-aligned tensors that a DataLoader fetches batch by batch.

    DataLoader calls ``__getitems__`` with all indices of a batch, so each
    batch is one gather per tensor instead of per-instance indexing and
    stacking. Pair it with ``collate_fn=passthrough_batch``.
    """

    def __init__(self, *tensors: Tensor) -> None:
        if not tensors or any(
            len(tensor) != len(tensors[0]) for tensor in tensors
        ):
            msg = "All tensors must have the same leading dimension."
            raise ValueError(msg)
        self.tensors = tensors

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        return tuple(tensor[index] for tensor in self.tensors)

    def __getitems__(self, indices: Sequence[int]) -> tuple[Tensor, ...]:
        return tuple(tensor[list(indices)] for tensor in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors[0])


def passthrough_batch(batch: object) -> tuple[Tensor, ...]:
    """Return a batch already gathered by ``TensorBatchDataset``."""
    return cast("tuple[Tensor, ...]", batch)
