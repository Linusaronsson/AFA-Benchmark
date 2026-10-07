import logging
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.provenance import Split, capture_provenance
from afabench.core.registry import get_class
from afabench.core.types import AFADataset
from afabench.datasets.config import DatasetGenerationConfig, SplitRatioConfig
from afabench.datasets.utils import require_generation_order

log = logging.getLogger(__name__)


def generate_and_save_image_split(
    dataset_class: type[AFADataset],
    split_ratio: SplitRatioConfig,
    seed_for_split: int,
    save_path: Path,
    dataset_kwargs: dict[str, Any],
    metadata_to_save: dict[str, Any],
    dataset_key: str,
    dataset_realization_index: int,
    resolved_config: dict[str, Any],
) -> None:
    """
    Generate and save a single train/val/test split for a dataset with a specific seed.

    The official `train/` folder is shuffled into the train and val splits;
    the official `val/` folder is the fixed test split. Every split loads
    `train/` then `val/`, so generation indices run over both folders in
    that order and are unique across splits. `dataset_key`,
    `dataset_realization_index` and `resolved_config` go into each bundle's
    provenance record.
    """

    def pool_size(subdir: str) -> int:
        return len(
            dataset_class(
                **dataset_kwargs
                | {"load_subdirs": (subdir,), "split_role": "val"}
            )
        )

    n_train_pool = pool_size("train")
    n_test_pool = pool_size("val")

    # Split ONLY into train/val from the official train pool
    train_size = int(split_ratio.train * n_train_pool)

    all_indices = list(range(n_train_pool))
    rnd = random.Random(seed_for_split)
    rnd.shuffle(all_indices)

    generation_indices_per_split: dict[Split, list[int]] = {
        "train": all_indices[:train_size],
        "val": all_indices[train_size:],
        # Official val/ follows train/ in generation order
        "test": list(range(n_train_pool, n_train_pool + n_test_pool)),
    }
    effective_kwargs: dict[str, dict[str, Any]] = {}
    splits: dict[Split, AFADataset] = {}
    for split, generation_indices in generation_indices_per_split.items():
        # The split role selects the transform: augmentation only for train
        effective_kwargs[split] = dataset_kwargs | {
            "load_subdirs": ("train", "val"),
            "split_role": split,
        }
        dataset = dataset_class(**effective_kwargs[split])
        require_generation_order(dataset)
        splits[split] = dataset.create_subset(generation_indices)

    # Create dataset directory
    save_path.mkdir(parents=True, exist_ok=True)

    # Prepare metadata
    base_metadata = metadata_to_save | {
        "seed_for_split": seed_for_split,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_kwargs": dataset_kwargs,
    }
    # Save splits and metadata
    for split, dataset in splits.items():
        save_bundle(
            obj=dataset,
            path=save_path / f"{split}.bundle",
            metadata=base_metadata
            | {
                "split": split,
                "effective_dataset_kwargs": effective_kwargs[split],
            },
            provenance=capture_provenance(
                stage="dataset_generation",
                resolved_config=resolved_config,
                seed=seed_for_split,
                smoke_test=False,
                device="cpu",
                dataset_key=dataset_key,
                dataset_realization_index=dataset_realization_index,
                split=split,
            ),
        )


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/dataset_generation",
    config_name="config",
)
def main(cfg: DatasetGenerationConfig) -> None:
    cfg = cast("DatasetGenerationConfig", OmegaConf.to_object(cfg))
    dataset_class = get_class(cfg.dataset.class_name)
    # The dataset key is the selected `dataset` config, not the class name
    dataset_key = HydraConfig.get().runtime.choices["dataset"]

    for dataset_realization_index, seed in zip(
        cfg.dataset_realization_indices, cfg.seeds, strict=True
    ):
        base_kwargs = cfg.dataset.kwargs
        if dataset_class.accepts_seed():
            dataset_kwargs: dict[str, Any] = base_kwargs | {"seed": seed}
        else:
            dataset_kwargs = base_kwargs
        generate_and_save_image_split(
            dataset_class=dataset_class,
            split_ratio=cfg.split_ratio,
            seed_for_split=seed,
            save_path=Path(cfg.save_path) / str(dataset_realization_index),
            dataset_kwargs=dataset_kwargs,
            metadata_to_save={
                "dataset_realization_index": dataset_realization_index,
                "class_name": cfg.dataset.class_name,
            },
            dataset_key=dataset_key,
            dataset_realization_index=dataset_realization_index,
            resolved_config=asdict(cfg),
        )


if __name__ == "__main__":
    main()
