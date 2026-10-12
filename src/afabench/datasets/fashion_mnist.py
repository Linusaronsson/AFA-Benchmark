from collections.abc import Callable
from typing import final, override

from torch import Tensor
from torchvision import datasets
from torchvision.datasets import VisionDataset

from afabench.datasets.torchvision_image import TorchvisionImageDataset


@final
class FashionMNISTDataset(TorchvisionImageDataset):
    """FashionMNIST dataset wrapped to follow the AFADataset protocol."""

    def __init__(
        self,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
        root: str = "data/fashion_mnist",
    ):
        super().__init__(
            root, train=train, transform=transform, download=download
        )

    @override
    def _images(self) -> VisionDataset:
        return datasets.FashionMNIST(
            root=self.root,
            train=self.train,
            transform=self.transform,
            download=self.download,
        )
