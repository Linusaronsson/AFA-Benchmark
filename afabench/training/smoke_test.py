import logging

import torch

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
    batch_size = min(default_batch_size, SMOKE_TEST_BATCH_SIZE)
    n_samples = min(default_n_samples, batch_size * SMOKE_TEST_N_BATCHES)
    return n_samples, batch_size


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
