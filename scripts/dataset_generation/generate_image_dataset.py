import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.bundle import save_bundle
from afabench.common.config_classes import (
    DatasetGenerationConfig,
    SplitRatioConfig,
)
from afabench.common.custom_types import AFADataset
from afabench.common.registry import get_class

log = logging.getLogger(__name__)


def generate_and_save_image_split(
    dataset_class: type[AFADataset],
    split_ratio: SplitRatioConfig,
    seed_for_split: int,
    save_path: Path,
    dataset_kwargs: dict[str, Any],
    metadata_to_save: dict[str, Any],
) -> None:
    """Generate and save a single train/val/test split for a dataset with a specific seed."""
    # Create TRAIN pool dataset with the specific seed
    train_kwargs = dict(dataset_kwargs)
    train_kwargs["load_subdirs"] = ("train",)
    train_kwargs["split_role"] = "train"

    val_kwargs = dict(dataset_kwargs)
    val_kwargs["load_subdirs"] = ("train",)
    val_kwargs["split_role"] = "val"

    test_kwargs = dict(dataset_kwargs)
    test_kwargs["load_subdirs"] = ("val",)
    test_kwargs["split_role"] = "test"

    train_pool = dataset_class(**val_kwargs)

    # Calculate split sizes
    # Split ONLY into train/val from the official train pool
    total_size = len(train_pool)
    train_size = int(split_ratio.train * total_size)
    val_size = total_size - train_size

    all_indices = list(range(total_size))
    rnd = random.Random(seed_for_split)
    rnd.shuffle(all_indices)

    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size : train_size + val_size]

    train_indices_t = torch.tensor(train_indices, dtype=torch.long)
    val_indices_t = torch.tensor(val_indices, dtype=torch.long)

    # Create subset datasets using the original dataset
    train_dataset = dataset_class(**train_kwargs)
    train_dataset.indices = train_indices_t  # pyright: ignore[reportAttributeAccessIssue]
    val_dataset = dataset_class(**val_kwargs)
    val_dataset.indices = val_indices_t  # pyright: ignore[reportAttributeAccessIssue]

    # Load official val/ as the fixed test set
    test_dataset = dataset_class(**test_kwargs)

    # Create dataset directory
    save_path.mkdir(parents=True, exist_ok=True)
    train_path = save_path / "train.bundle"
    val_path = save_path / "val.bundle"
    test_path = save_path / "test.bundle"

    # Prepare metadata
    base_metadata = metadata_to_save | {
        "seed_for_split": seed_for_split,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_kwargs": dataset_kwargs,
    }
    # Save splits and metadata
    save_bundle(
        obj=train_dataset,
        path=train_path,
        metadata=base_metadata
        | {"split": "train", "effective_dataset_kwargs": train_kwargs},
    )
    save_bundle(
        obj=val_dataset,
        path=val_path,
        metadata=base_metadata
        | {"split": "val", "effective_dataset_kwargs": val_kwargs},
    )
    save_bundle(
        obj=test_dataset,
        path=test_path,
        metadata=base_metadata
        | {"split": "test", "effective_dataset_kwargs": test_kwargs},
    )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    dataset_class = get_class(cfg.dataset.class_name)

    for instance_idx, seed in zip(
        cfg.instance_indices, cfg.seeds, strict=True
    ):
        raw_kwargs = OmegaConf.to_container(cfg.dataset.kwargs, resolve=True)
        base_kwargs = cast("dict[str, Any]", raw_kwargs)
        if dataset_class.accepts_seed():
            dataset_kwargs: dict[str, Any] = base_kwargs | {"seed": seed}
        else:
            dataset_kwargs = base_kwargs
        generate_and_save_image_split(
            dataset_class=dataset_class,
            split_ratio=cfg.split_ratio,
            seed_for_split=seed,
            save_path=Path(cfg.save_path) / str(instance_idx),
            dataset_kwargs=dataset_kwargs,
            metadata_to_save={
                "instance_idx": instance_idx,
                "class_name": cfg.dataset.class_name,
            },
        )


if __name__ == "__main__":
    main()
