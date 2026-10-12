import logging
import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import wandb
from torch import nn
from wandb.sdk.wandb_run import Run

from afabench.core.types import FeatureMask, Label, MaskedFeatures

logger = logging.getLogger(__name__)


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


def initialize_wandb_run(
    cfg: dict[str, Any],
    job_type: str,
    tags: list[str],
) -> Run:
    # W&B reads the run's ID, name and group from WANDB_RUN_ID, WANDB_NAME
    # and WANDB_RUN_GROUP, which the pipeline sets (workflow/src/wandb_runs.py).
    run = wandb.init(
        config=cfg, job_type=job_type, tags=tags, dir="logs/wandb"
    )

    # Log W&B run info
    logger.info(f"W&B run initialized: {run.name} ({run.id})")
    if run.group:
        logger.info(f"W&B run group: {run.group}")
    logger.info(f"W&B run URL: {run.url}")
    return run


def flatten_afa_input(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    label: Label | None,
    feature_shape: torch.Size,
) -> tuple[MaskedFeatures, FeatureMask, Label | None]:
    """
    Flatten feature dimensions and batch dimensions.

    Since MaskedFeatures, FeatureMask and Label can have an arbitrary number of batch dimensions and MaskedFeatures/FeatureMask can have an arbitrary number of feature dimensions, this function is useful for models that assume a single batch dimension and single feature dimension.
    """
    n_feature_dims = len(feature_shape)
    # Flatten feature dimensions
    flat_masked_features = masked_features.flatten(start_dim=-n_feature_dims)
    flat_feature_mask = feature_mask.flatten(start_dim=-n_feature_dims)
    # Now everything except the last dimension should be batch dimensions, flatten them as well
    flat_masked_features = flat_masked_features.flatten(end_dim=-2)
    flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
    flat_label = label if label is None else label.flatten(end_dim=-2)

    return flat_masked_features, flat_feature_mask, flat_label
