from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import ClassVar, override

import torch
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import VisionDataset

from afabench.core.types import Features, Label
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot


class TorchvisionImageDataset(InMemoryAFADataset, ABC):
    """A torchvision image classification dataset, with every 28x28 image flattened to 784 features and 10 classes."""

    config_keys: ClassVar[tuple[str, ...]] = ("train", "root")

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([784])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([10])

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    def __init__(
        self,
        root: str,
        *,
        train: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
        download: bool = True,
    ):
        self.train: bool = train
        self.root: str = root
        # Used only to build the data, so not part of the config
        self.transform: Callable[[Tensor], Tensor] = (
            transform if transform is not None else transforms.ToTensor()
        )
        self.download: bool = download
        super().__init__()

    @abstractmethod
    def _images(self) -> VisionDataset:
        """Return the torchvision dataset of `root`, `train`, `transform` and `download`."""

    @override
    def _build(self) -> tuple[Features, Label]:
        images = self._images()
        features = torch.stack([x[0].flatten() for x in images]).float()
        labels = torch.tensor([x[1] for x in images])
        return features, one_hot(labels, self.label_shape[0])
