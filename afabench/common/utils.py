import json
import os
import random
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch import nn

from afabench.common.afa_initializers.base import AFAInitializer
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
    AFAUnmaskFn,
)
from afabench.common.registry import (
    get_afa_classifier_class,
    get_afa_dataset_class,
    get_afa_initializer,
    get_afa_method_class,
    get_afa_unmasker,
)


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_class_frequencies(labels: torch.Tensor) -> torch.Tensor:
    """Return class frequencies for labels of shape (*batch_size, n_classes)."""
    assert labels.shape[-1] > 1, f"Expected multi-class labels, got {
        labels.shape
    }"
    class_counts = labels.flatten(0, -2).float().sum(dim=0)
    return class_counts / class_counts.sum()


@contextmanager
def eval_mode(*models: nn.Module) -> Generator[None, None, None]:
    was_training = [m.training for m in models]
    try:
        for m in models:
            m.eval()
        yield
    finally:
        for m, mode in zip(models, was_training, strict=False):
            m.train(mode)


def save_method_artifact(
    method: AFAMethod,
    save_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Save an AFA method with metadata."""
    if isinstance(save_path, str):
        save_path = Path(save_path)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        method.save(temp_path)
        save_artifact(
            artifact_dir=save_path,
            files={f.name: f for f in temp_path.iterdir() if f.is_file()},
            metadata=metadata,
        )


def save_artifact(
    artifact_dir: Path,
    files: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """Save artifact files and metadata.json to directory."""
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    for dest_name, source_path in files.items():
        dest = artifact_dir / dest_name
        if source_path.is_file():
            shutil.copy2(source_path, dest)
        elif source_path.is_dir():
            shutil.copytree(source_path, dest, dirs_exist_ok=True)


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load metadata.json from artifact directory."""
    path = artifact_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"No metadata.json in {artifact_dir}")
    with open(path) as f:
        return json.load(f)


def load_pretrained_model(
    path: Path,
    device: torch.device | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load pretrained model checkpoint. Returns (model_path, metadata)."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(path)
    model_path = path / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No model.pt in {path}")
    return model_path, metadata


def load_method_artifact(
    path: Path,
    device: torch.device | None = None,
) -> tuple[AFAMethod, dict[str, Any]]:
    """Load trained AFA method from artifact path."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(path)
    method_class = get_afa_method_class(metadata["method_type"])
    return method_class.load(path, device=device), metadata


def load_dataset_artifact(
    path: Path,
    split: str,
) -> tuple[AFADataset, dict[str, Any]]:
    """Load single dataset split (train/val/test) from artifact path."""
    if isinstance(path, str):
        path = Path(path)

    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}")
    metadata = load_artifact_metadata(path)
    dataset_class = get_afa_dataset_class(metadata["dataset_type"])
    return dataset_class.load(path / f"{split}.pt"), metadata


def load_dataset_splits(
    path: Path,
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """Load all dataset splits. Returns (train, val, test, metadata)."""
    for f in ["train.pt", "val.pt", "test.pt", "metadata.json"]:
        if not (path / f).exists():
            raise FileNotFoundError(f"Missing {f} in {path}")
    metadata = load_artifact_metadata(path)
    cls = get_afa_dataset_class(metadata["dataset_type"])

    return (
        cls.load(path / "train.pt"),
        cls.load(path / "val.pt"),
        cls.load(path / "test.pt"),
        metadata,
    )


def load_classifier_artifact(
    path: Path,
    device: torch.device | None = None,
) -> AFAClassifier:
    """Load trained classifier from artifact path."""
    device = device or torch.device("cpu")
    metadata = load_artifact_metadata(path)
    classifier_class = get_afa_classifier_class(
        metadata["classifier_class_name"]
    )
    return classifier_class.load(path / "classifier.pt", device=device)


def load_unmasker(name: str, **kwargs) -> AFAUnmaskFn:
    """Load unmasker function by name."""
    return get_afa_unmasker(name, **kwargs)


def load_initializer(name: str, **kwargs) -> AFAInitializer:
    """Load initializer by name."""
    return get_afa_initializer(name, **kwargs)


def load_eval_components(
    method_artifact_path: Path,
    unmasker_name: str,
    initializer_name: str,
    dataset_artifact_path: Path,
    dataset_split: str,
    classifier_artifact_path: Path,
    device: torch.device | None = None,
    unmasker_kwargs: dict[str, Any] | None = None,
    initializer_kwargs: dict[str, Any] | None = None,
) -> tuple[AFAMethod, AFAUnmaskFn, AFAInitializer, AFADataset, AFAClassifier]:
    """Load all components for evaluation."""
    device = device or torch.device("cpu")

    return (
        load_method_artifact(method_artifact_path, device=device),
        load_unmasker(unmasker_name, **(unmasker_kwargs or {})),
        load_initializer(initializer_name, **(initializer_kwargs or {})),
        load_dataset_artifact(dataset_artifact_path, dataset_split),
        load_classifier_artifact(classifier_artifact_path, device=device),
    )
