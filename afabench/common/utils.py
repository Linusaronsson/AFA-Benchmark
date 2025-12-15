import logging
import os
import random
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from torch import nn
from wandb.sdk.wandb_run import Run

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
    cfg: Any,  # noqa: ANN401
    job_type: str,
    tags: list[str],
) -> Run:
    # Convert config to a plain dict for W&B
    config_dict = cast(
        "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
    )

    # Allow grouping of runs via environment variable `WANDB_GROUP`
    # or via a `wandb_group` key in the config dict. Environment
    # variable takes precedence.
    group = os.environ.get("WANDB_GROUP") or config_dict.get("wandb_group")

    init_kwargs: dict[str, Any] = {
        "config": config_dict,
        "job_type": job_type,
        "tags": tags,
        "dir": "extra/wandb",
    }
    if group:
        init_kwargs["group"] = group

    run = wandb.init(**init_kwargs)

    # Log W&B run info
    logger.info(f"W&B run initialized: {run.name} ({run.id})")
    if group:
        logger.info(f"W&B run group: {group}")
    logger.info(f"W&B run URL: {run.url}")
    return run
