import copy
import tarfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Self, final, override
from urllib.parse import urlparse

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from afabench.core.types import AFADataset
from afabench.datasets.feature_costs import load_saved_feature_costs
from afabench.datasets.utils import load_generation_indices


@final
class ImagenetteDataset(Dataset[tuple[Tensor, Tensor]], AFADataset):
    """
    Imagenette dataset from the FastAI image classification benchmark.

    A subset of 10 easily classified classes from Imagenet.

    Only file lists are kept in memory. The generation indices double as
    positions in `samples`, which lists the `load_subdirs` folders in order,
    so dataset generation loads `train/` then `val/` for every split.
    """

    IMAGENETTE_URL: ClassVar[str] = (
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    )

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        subset = copy.deepcopy(self)
        subset.generation_indices = self.generation_indices[
            list(indices)
        ].clone()
        return subset

    @override
    def get_generation_indices(self) -> Tensor:
        return self.generation_indices

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([3, 224, 224])

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
        data_root: str = "data/imagenette",
        variant_dir: str = "imagenette2-320",
        load_subdirs: tuple[str, ...] = ("train", "val"),
        image_size: int = 224,
        split_role: str | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.variant_dir = variant_dir
        self.load_subdirs = load_subdirs
        self.image_size = image_size
        self.split_role = split_role  # "train" | "val" | "test"

        use_train_aug = self.split_role == "train"
        self.transform = (
            self._train_transform()
            if use_train_aug
            else self._eval_transform()
        )

        root = self._root()
        sub_datasets: list[ImageFolder] = []
        for sub in self.load_subdirs:
            d = root / sub
            if not d.exists():
                msg = f"Expected subdir '{sub}' at {d}"
                raise FileNotFoundError(msg)
            sub_datasets.append(ImageFolder(str(d), transform=None))

        self.samples = [
            path for ds in sub_datasets for (path, _) in ds.samples
        ]
        self.targets = torch.tensor(
            [y for ds in sub_datasets for (_, y) in ds.samples],
            dtype=torch.long,
        )
        self.generation_indices = torch.arange(
            len(self.samples), dtype=torch.long
        )

    def _resolve_index(self, i: int) -> int:
        if i < 0 or i >= int(self.generation_indices.numel()):
            msg = f"Index {i} out of range for dataset of length {len(self)}"
            raise IndexError(msg)
        return int(self.generation_indices[i].item())

    def _train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def _eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def _download_imagenette(self, root: Path) -> None:
        root_parent = root.parent
        root_parent.mkdir(parents=True, exist_ok=True)
        archive_path = root_parent / f"{self.variant_dir}.tgz"

        parsed = urlparse(self.IMAGENETTE_URL)
        if parsed.scheme not in ("http", "https"):
            msg = f"Invalid URL scheme in IMAGENETTE_URL: {parsed.scheme}"
            raise ValueError(msg)

        if not archive_path.exists():
            print(
                f"Downloading Imagenette from {self.IMAGENETTE_URL} to {
                    archive_path
                }"
            )
            urllib.request.urlretrieve(self.IMAGENETTE_URL, archive_path)  # noqa: S310

        print(f"Extracting {archive_path} to {root_parent}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=root_parent, filter="data")

    def _root(self) -> Path:
        root = Path(self.data_root) / self.variant_dir
        if (not root.exists()) or (not any(root.iterdir())):
            self._download_imagenette(root)

        if not root.exists():
            msg = (
                f"Imagenette folder not found at {
                    root
                } after attempted download. "
                f"Expected '{self.variant_dir}' with train/ and val/ subdirs."
            )
            raise FileNotFoundError(msg)
        return root

    @override
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        assert (  # noqa: PT018
            self.samples is not None
            and self.targets is not None
            and self.transform is not None
        ), "Dataset not properly initialized"
        idx = self._resolve_index(i)
        path = self.samples[idx]
        y = self.targets[idx]
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            x = self.transform(img_rgb)
            # Ensure x is a Tensor (transforms.ToTensor() should guarantee this)
            if not isinstance(x, Tensor):
                msg = f"Transform did not return a Tensor, got {type(x)}"
                raise TypeError(msg)
        # Convert label to one-hot tensor
        y_onehot = torch.nn.functional.one_hot(
            y, num_classes=self.label_shape[0]
        ).float()
        return x, y_onehot

    @override
    def __len__(self) -> int:
        return int(self.generation_indices.numel())

    @override
    def get_all_data(self) -> tuple[Tensor, Tensor]:
        """Return all data as tensors. Note: This loads all images into memory."""
        features = []
        labels = []
        for i in range(len(self)):
            x, y = self[i]
            features.append(x)
            labels.append(y)
        return torch.stack(features), torch.stack(labels)

    @override
    def save(self, path: Path) -> None:
        """Save only the generation indices and the dataset config, to reconstruct later from raw files on load."""
        torch.save(
            {
                "generation_indices": self.generation_indices,
                "feature_costs": self.get_feature_acquisition_costs(),
                "config": {
                    "data_root": self.data_root,
                    "variant_dir": self.variant_dir,
                    "load_subdirs": self.load_subdirs,
                    "image_size": self.image_size,
                    "split_role": self.split_role,
                },
            },
            path / "dataset.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        data = torch.load(path / "dataset.pt")
        cfg = data["config"]
        generation_indices = load_generation_indices(
            data, path / "dataset.pt", None
        )
        if "split_role" not in cfg or cfg["split_role"] is None:
            msg = "Split role not initialized!"
            raise ValueError(msg)

        # Create instance without calling __init__
        obj = cls.__new__(cls)
        obj.data_root = cfg["data_root"]
        obj.variant_dir = cfg["variant_dir"]
        obj.load_subdirs = cfg["load_subdirs"]
        obj.image_size = cfg["image_size"]
        obj.split_role = cfg["split_role"]

        use_train_aug = obj.split_role == "train"
        obj.transform = (
            obj._train_transform() if use_train_aug else obj._eval_transform()  # noqa: SLF001
        )

        root = obj._root()  # noqa: SLF001
        sub_datasets: list[ImageFolder] = []
        for sub in obj.load_subdirs:
            d = root / sub
            if not d.exists():
                msg = f"Expected subdir '{sub}' at {d}"
                raise FileNotFoundError(msg)
            sub_datasets.append(ImageFolder(str(d), transform=None))

        all_samples = [path for ds in sub_datasets for (path, _) in ds.samples]
        all_targets = torch.tensor(
            [y for ds in sub_datasets for (_, y) in ds.samples],
            dtype=torch.long,
        )

        # Apply subset filtering
        obj.samples = all_samples
        obj.targets = all_targets
        obj.generation_indices = generation_indices
        n = len(obj.samples)
        if int(generation_indices.numel()) > 0 and (
            int(generation_indices.min().item()) < 0
            or int(generation_indices.max().item()) >= n
        ):
            msg = (
                f"Loaded generation indices out of bounds: valid [0, {n - 1}]"
            )
            raise ValueError(msg)
        obj.feature_costs = load_saved_feature_costs(
            data, path / "dataset.pt", obj.feature_shape
        )

        return obj
