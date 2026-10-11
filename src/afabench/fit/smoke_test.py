import logging
from collections.abc import Mapping
from dataclasses import is_dataclass, replace
from typing import TYPE_CHECKING, cast

import torch

from afabench.core.types import AFADataset
from afabench.fit.contract import BaseContract

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

log = logging.getLogger(__name__)

SMOKE_TEST_BATCH_SIZE = 2
SMOKE_TEST_N_BATCHES = 2
SMOKE_TEST_N_SAMPLES = SMOKE_TEST_BATCH_SIZE * SMOKE_TEST_N_BATCHES
SMOKE_TEST_N_TRAINING_SAMPLES = 10
SMOKE_TEST_N_EPOCHS = 1


def limit[ConfigT: BaseContract](cfg: ConfigT, **fields: object) -> ConfigT:
    """
    Replace `fields` of a fit-stage config during a smoke test.

    A field given as a mapping where `cfg` holds a dataclass replaces only
    the named fields of that dataclass, at any depth. Outside a smoke test
    `cfg` is returned unchanged.
    """
    if not cfg.smoke_test:
        return cfg

    log.info("Smoke test detected.")
    return _replace_fields(cfg, fields)


def _replace_fields[ConfigT: "DataclassInstance"](
    cfg: ConfigT, fields: Mapping[str, object]
) -> ConfigT:
    replacements: dict[str, object] = {}
    for name, value in fields.items():
        current = getattr(cfg, name)
        if (
            isinstance(value, Mapping)
            and is_dataclass(current)
            and not isinstance(current, type)
        ):
            replacements[name] = _replace_fields(
                current, cast("Mapping[str, object]", value)
            )
        else:
            replacements[name] = value
    return replace(cfg, **replacements)


def eval_settings(
    *,
    smoke_test: bool,
    default_n_samples: int,
    default_batch_size: int,
) -> tuple[int, int]:
    if not smoke_test:
        return default_n_samples, default_batch_size

    log.info("Smoke test detected.")
    batch_size = training_batch_size(
        smoke_test=smoke_test,
        default_batch_size=default_batch_size,
    )
    n_samples = min(default_n_samples, batch_size * SMOKE_TEST_N_BATCHES)
    return n_samples, batch_size


def training_batch_size(*, smoke_test: bool, default_batch_size: int) -> int:
    if not smoke_test:
        return default_batch_size

    return min(default_batch_size, SMOKE_TEST_BATCH_SIZE)


def training_subset(
    X_train: torch.Tensor,  # noqa: N803
    y_train: torch.Tensor,
    *,
    smoke_test: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not smoke_test:
        return X_train, y_train

    log.info("Smoke test detected.")
    return (
        X_train[:SMOKE_TEST_N_TRAINING_SAMPLES],
        y_train[:SMOKE_TEST_N_TRAINING_SAMPLES],
    )


def limit_supervised_learning[ConfigT: BaseContract](cfg: ConfigT) -> ConfigT:
    """
    Train one epoch of a few batches during a smoke test.

    `cfg` holds a `SupervisedLearningConfig` in its `supervised_learning`
    field.
    """
    return limit(
        cfg,
        supervised_learning={
            "max_epochs": SMOKE_TEST_N_EPOCHS,
            "limit_train_batches": SMOKE_TEST_N_BATCHES,
            "limit_val_batches": SMOKE_TEST_N_BATCHES,
        },
    )


def dataset_subset[DatasetT: AFADataset](
    dataset: DatasetT, *, smoke_test: bool
) -> DatasetT:
    if not smoke_test:
        return dataset

    log.info("Smoke test detected.")
    n_samples = min(len(dataset), SMOKE_TEST_N_SAMPLES)
    return dataset.create_subset(range(n_samples))
