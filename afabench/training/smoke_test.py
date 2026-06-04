import logging

import torch

log = logging.getLogger(__name__)

SMOKE_TEST_BATCH_SIZE = 2
SMOKE_TEST_N_SAMPLES = 10


def eval_settings(
    *,
    smoke_test: bool,
    default_n_samples: int,
    default_batch_size: int,
) -> tuple[int, int]:
    if not smoke_test:
        return default_n_samples, default_batch_size

    log.info("Smoke test detected.")
    return SMOKE_TEST_N_SAMPLES, SMOKE_TEST_BATCH_SIZE


def training_subset(
    X_train: torch.Tensor,  # noqa: N803
    y_train: torch.Tensor,
    *,
    smoke_test: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not smoke_test:
        return X_train, y_train

    log.info("Smoke test detected.")
    return X_train[:SMOKE_TEST_N_SAMPLES], y_train[:SMOKE_TEST_N_SAMPLES]
