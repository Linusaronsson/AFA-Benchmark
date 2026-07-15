import logging

import torch

from afabench.core.types import AFADataset

log = logging.getLogger(__name__)

SMOKE_TEST_BATCH_SIZE = 2
SMOKE_TEST_N_BATCHES = 2
SMOKE_TEST_N_SAMPLES = SMOKE_TEST_BATCH_SIZE * SMOKE_TEST_N_BATCHES
SMOKE_TEST_N_TRAINING_SAMPLES = 10


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


def dataset_subset[DatasetT: AFADataset](
    dataset: DatasetT, *, smoke_test: bool
) -> DatasetT:
    if not smoke_test:
        return dataset

    log.info("Smoke test detected.")
    n_samples = min(len(dataset), SMOKE_TEST_N_SAMPLES)
    return dataset.create_subset(range(n_samples))
