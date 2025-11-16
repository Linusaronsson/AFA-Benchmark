import os
import json
import torch
import shutil
import random
import numpy as np

from typing import Any
from pathlib import Path
from torch import Tensor, nn
from jaxtyping import Bool, Float
from collections.abc import Generator
from contextlib import contextmanager

from afabench.common.custom_types import AFADataset, AFAMethod, AFAClassifier

from afabench.common.registry import (
    get_afa_dataset_class,
    get_afa_method_class,
    get_afa_classifier_class,
)


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def get_class_probabilities(
    labels: Bool[Tensor, "*batch n_classes"],
) -> Float[Tensor, "n_classes"]:
    """Return the class probabilities for a given set of labels."""
    class_counts = labels.float().sum(dim=0)
    class_probabilities = class_counts / class_counts.sum()
    return class_probabilities


def get_folders_with_matching_params(
    folder: Path, mapping: dict[str, Any]
) -> list[Path]:
    """Get all folders in a given folder that have a matching params.yml file."""
    matching_folders = [
        f
        for f in folder.iterdir()
        if yaml_file_matches_mapping(f / "params.yml", mapping)
    ]

    return matching_folders


@contextmanager
def eval_mode(*models: nn.Module) -> Generator[None, None, None]:
    was_training = [model.training for model in models]
    try:
        for model in models:
            model.eval()
        yield
    finally:
        for model, mode in zip(models, was_training, strict=False):
            model.train(mode)


def dict_with_prefix(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    """Return a dictionary with all keys prefixed by ."""
    return {f"{prefix}{k}": v for k, v in d.items()}


def save_artifact(
    artifact_dir: Path,
    files: dict[str, Path],
    metadata: dict[str, Any],
) -> None:
    """
    Save artifact files and metadata to local filesystem.

    Args:
        artifact_dir: Directory to save artifact (will be created)
        files: Dict mapping destination filename to source file path
        metadata: Metadata dict to save as JSON
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy files
    for dest_name, source_path in files.items():
        if source_path.is_file():
            shutil.copy2(source_path, artifact_dir / dest_name)
        elif source_path.is_dir():
            shutil.copytree(
                source_path, artifact_dir / dest_name, dirs_exist_ok=True
            )


def load_artifact_metadata(artifact_dir: Path) -> dict[str, Any]:
    """Load metadata.json from artifact directory."""
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {artifact_dir}")

    with open(metadata_path) as f:
        return json.load(f)


def get_artifact_path(
    artifact_type: str,
    artifact_name: str,
    base_dir: Path = Path("extra"),
) -> Path:
    """
    Construct path to artifact based on type and name.

    Examples:
        dataset: extra/data/cube_split_1/
        trained_method: extra/shim2018/cube_split_1_budget_3_seed_42/
        trained_classifier: extra/classifiers/masked_mlp/cube_split_1/
        pretrained_model: extra/shim2018/pretrained/cube_split_1_seed_42/
    """
    if artifact_type == "dataset":
        return base_dir / "data" / artifact_name

    elif artifact_type == "trained_classifier":
        # Format: classifier_type/dataset_variant or just name
        parts = artifact_name.split("/", 1)
        if len(parts) == 1:
            return base_dir / "classifiers" / artifact_name
        classifier_type, dataset_variant = parts
        return base_dir / "classifiers" / classifier_type / dataset_variant

    elif artifact_type == "pretrained_model":
        # Format: method/pretrained/dataset_variant
        parts = artifact_name.split("/", 1)
        method = parts[0] if len(parts) > 0 else artifact_name
        dataset_variant = parts[1] if len(parts) > 1 else ""
        return base_dir / method / "pretrained" / dataset_variant

    elif artifact_type == "trained_method":
        # Format: method/dataset_params
        parts = artifact_name.split("/", 1)
        method = parts[0] if len(parts) > 0 else artifact_name
        dataset_params = parts[1] if len(parts) > 1 else ""
        return base_dir / method / dataset_params

    else:
        raise ValueError(f"Unknown artifact type: {artifact_type}")


def parse_artifact_name(artifact_name: str) -> tuple[str, str, str]:
    """
    Parse artifact name into components.

    Returns: (method, dataset_variant, params_str)
    """
    name = artifact_name
    for prefix in ["train_", "pretrain_", "eval_"]:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    parts = name.split("-")
    if len(parts) < 2:
        return name, "", ""

    method = parts[0]
    dataset_variant = parts[1]
    params = "_".join(parts[2:]) if len(parts) > 2 else ""

    return method, dataset_variant, params


def build_artifact_name(
    method: str,
    dataset: str,
    split: int | str,
    seed: int,
    budget: int | None = None,
    cost_param: float | None = None,
    prefix: str = "train",
) -> str:
    """Build standardized artifact name."""
    parts = [f"{prefix}_{method}", f"{dataset}_split_{split}"]

    if budget is not None:
        parts.append(f"budget_{budget}")
    if cost_param is not None:
        parts.append(f"costparam_{cost_param}")

    parts.append(f"seed_{seed}")
    return "-".join(parts)


def load_dataset(
    dataset_path: str | Path,
    base_dir: Path = Path("extra"),
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """
    Load train, validation, and test datasets from local filesystem.

    Returns: (train_dataset, val_dataset, test_dataset, metadata)
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    if not dataset_path.is_absolute():
        dataset_path = get_artifact_path(
            "dataset", str(dataset_path), base_dir
        )

    # Check required files
    required_files = ["train.pt", "val.pt", "test.pt", "metadata.json"]
    artifact_filenames = [
        f.name for f in dataset_path.iterdir() if f.is_file()
    ]

    missing = set(required_files) - set(artifact_filenames)
    if missing:
        raise FileNotFoundError(
            f"Dataset at {dataset_path} missing: {missing}. Found: {
                artifact_filenames
            }"
        )

    # Load metadata and datasets
    metadata = load_artifact_metadata(dataset_path)
    dataset_class = get_afa_dataset_class(metadata["dataset_type"])

    train_dataset: AFADataset = dataset_class.load(dataset_path / "train.pt")
    val_dataset: AFADataset = dataset_class.load(dataset_path / "val.pt")
    test_dataset: AFADataset = dataset_class.load(dataset_path / "test.pt")

    return train_dataset, val_dataset, test_dataset, metadata


def load_dataset_artifact(
    artifact_name: str,
    base_dir: Path = Path("extra"),
) -> tuple[AFADataset, AFADataset, AFADataset, dict[str, Any]]:
    """
    Load datasets from filesystem. Parses old wandb-style names for compatibility.

    Args:
        artifact_name: "cube_split_1" or "afa-team/afa-benchmark/cube_split_1:latest"
    """
    # Parse old wandb-style names
    if "/" in artifact_name:
        parts = artifact_name.split("/")
        name = parts[-1].split(":")[0]  # Remove version tag
    else:
        name = artifact_name

    return load_dataset(name, base_dir)


def load_trained_method(
    artifact_name: str,
    device: torch.device | None = None,
    base_dir: Path = Path("extra"),
) -> tuple[AFADataset, AFADataset, AFADataset, AFAMethod, dict[str, Any]]:
    """
    Load trained AFA method and dataset.

    Returns: (train_dataset, val_dataset, test_dataset, method, metadata)
    """
    if device is None:
        device = torch.device("cpu")

    # Parse old wandb format
    if isinstance(artifact_name, str) and ":" in artifact_name:
        artifact_name = artifact_name.split(":")[0]
    if artifact_name.startswith("train_"):
        artifact_name = artifact_name[6:]

    method_path = get_artifact_path("trained_method", artifact_name, base_dir)

    if not method_path.exists():
        raise FileNotFoundError(f"Trained method not found: {method_path}")

    # Load metadata and method
    metadata = load_artifact_metadata(method_path)
    method_class = get_afa_method_class(metadata["method_type"])
    method = method_class.load(method_path, device=device)

    # Load dataset
    dataset_name = metadata["dataset_artifact_name"]
    if "/" in dataset_name:
        dataset_name = dataset_name.split("/")[-1].split(":")[0]

    train_dataset, val_dataset, test_dataset, _ = load_dataset(
        dataset_name, base_dir
    )

    return train_dataset, val_dataset, test_dataset, method, metadata


def load_trained_classifier(
    artifact_name: str,
    device: torch.device | None = None,
    base_dir: Path = Path("extra"),
) -> tuple[AFAClassifier, dict[str, Any]]:
    """
    Temporary patched version:
    If metadata.json is missing locally, fetch W&B artifact metadata once.
    """

    if device is None:
        device = torch.device("cpu")

    if ":" in artifact_name:
        artifact_name = artifact_name.split(":")[0]

    classifier_path = get_artifact_path(
        "trained_classifier", artifact_name, base_dir
    )

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Trained classifier not found: {classifier_path}"
        )

    # --- TEMPORARY METADATA PATCH ---------------------------------------
    metadata_file = classifier_path / "metadata.json"
    if not metadata_file.exists():
        import wandb
        import json

        api = wandb.Api()
        # assume "latest" alias for ad-hoc use
        artifact = api.artifact(f"{artifact_name}:latest")
        metadata = dict(artifact.metadata or {})

        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    # ---------------------------------------------------------------------

    metadata = load_artifact_metadata(classifier_path)
    classifier_class_name = metadata["classifier_class_name"]

    classifier_class = get_afa_classifier_class(classifier_class_name)
    classifier = classifier_class.load(
        classifier_path / "classifier.pt", device=device
    )

    return classifier, metadata


def load_pretrained_model(
    artifact_name: str,
    model_class: type,
    device: torch.device | None = None,
    base_dir: Path = Path("extra"),
) -> tuple[
    AFADataset, AFADataset, AFADataset, dict[str, Any], Any, dict[str, Any]
]:
    """
    Load pretrained model and dataset.

    Returns: (train_dataset, val_dataset, test_dataset, dataset_metadata, model, pretrain_config)
    """
    if device is None:
        device = torch.device("cpu")

    # Parse old wandb names
    if ":" in artifact_name:
        artifact_name = artifact_name.split(":")[0]

    pretrained_path = get_artifact_path(
        "pretrained_model", artifact_name, base_dir
    )

    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}"
        )

    if not (pretrained_path / "model.pt").exists():
        raise FileNotFoundError(f"model.pt not found in {pretrained_path}")

    # Load metadata and model
    metadata = load_artifact_metadata(pretrained_path)
    pretrain_config = metadata.get("pretrain_config", {})
    model = model_class.load(pretrained_path / "model.pt", map_location=device)

    # Load dataset
    dataset_name = metadata["dataset_artifact_name"]
    if "/" in dataset_name:
        dataset_name = dataset_name.split("/")[-1].split(":")[0]

    train_dataset, val_dataset, test_dataset, dataset_metadata = load_dataset(
        dataset_name, base_dir
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        dataset_metadata,
        model,
        pretrain_config,
    )


def validate_method_classifier_compatibility(
    method_metadata: dict[str, Any],
    classifier_metadata: dict[str, Any],
) -> None:
    """Validate method and classifier were trained on same dataset."""
    method_dataset = method_metadata["dataset_artifact_name"]
    classifier_dataset = classifier_metadata["dataset_artifact_name"]

    # Parse old wandb names
    if "/" in method_dataset:
        method_dataset = method_dataset.split("/")[-1].split(":")[0]
    if "/" in classifier_dataset:
        classifier_dataset = classifier_dataset.split("/")[-1].split(":")[0]

    assert method_dataset == classifier_dataset, (
        f"Method and classifier must be trained on same dataset. "
        f"Method: {method_dataset}, Classifier: {classifier_dataset}"
    )
