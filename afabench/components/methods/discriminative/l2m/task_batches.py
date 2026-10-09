"""
Batches of task-prior tasks shared by L2M's two fit stages.

Both `pretrain_l2m` and `train_l2m` consume tasks only through
`sample_task`, assembled here into batches with the context set first and
queries last. Each query gets one random acquisition mask over its
retrospectively available features, the affordable reading of the paper's
Algorithms 1 and 2 (arXiv:2510.12624), which visit every mask size.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool

from afabench.components.methods.discriminative.l2m.models import (
    TaskFeatures,
    TaskLabels,
    TaskMask,
)
from afabench.components.methods.discriminative.l2m.task_sampler import (
    FeatureSource,
    sample_task,
)
from afabench.core.types import Features

type AvailableFeatures = Bool[torch.Tensor, "tasks sequence n_features"]
type QueryAvailableFeatures = Bool[torch.Tensor, "tasks queries n_features"]
type AcquisitionMask = Bool[torch.Tensor, "tasks queries n_features"]

# Task seeds are drawn from one generator per run, so the held-out tasks
# and the training tasks are distinct draws from the task prior.
TASK_SEED_BOUND = 2**62


@dataclass(frozen=True, kw_only=True)
class TaskBatch:
    """
    Tasks of one step, with the context set first and queries last.

    `mask` holds the retrospective missingness mask of context instances
    and the random acquisition mask of queries; `available` holds the
    retrospective missingness mask of every instance.
    """

    features: TaskFeatures
    mask: TaskMask
    labels: TaskLabels
    available: AvailableFeatures
    n_context: int


def parse_feature_source(value: str) -> FeatureSource:
    if value in ("real", "synthetic"):
        return value
    msg = f"feature_source must be 'real' or 'synthetic'; got {value!r}"
    raise ValueError(msg)


def draw_task_batch(
    n_tasks: int,
    generator: torch.Generator,
    *,
    feature_source: FeatureSource,
    feature_pool: Features | None,
    n_features: int,
    label_shape: torch.Size,
    sequence_length: int,
    missingness_cap: float,
    keep_one_unacquired: bool = False,
) -> TaskBatch:
    """
    Draw `n_tasks` tasks with one context size, uniform on 1 to N - 1.

    With `keep_one_unacquired`, a query never has every available feature
    acquired, so the policy stage always has a selection to make.
    """
    tasks = [
        sample_task(
            feature_source,
            n_features=n_features,
            sequence_length=sequence_length,
            label_shape=label_shape,
            missingness_cap=missingness_cap,
            feature_pool=feature_pool,
            seed=int(torch.randint(TASK_SEED_BOUND, (), generator=generator)),
        )
        for _ in range(n_tasks)
    ]
    available = torch.stack([task.feature_mask for task in tasks]).bool()
    n_context = int(torch.randint(1, sequence_length, (), generator=generator))
    return TaskBatch(
        features=torch.stack([task.features for task in tasks]),
        mask=torch.cat(
            (
                available[:, :n_context],
                random_acquisition_mask(
                    available[:, n_context:],
                    generator,
                    keep_one_unacquired=keep_one_unacquired,
                ),
            ),
            dim=1,
        ),
        labels=torch.stack([task.labels for task in tasks]),
        available=available,
        n_context=n_context,
    )


def random_acquisition_mask(
    available: QueryAvailableFeatures,
    generator: torch.Generator,
    *,
    keep_one_unacquired: bool = False,
) -> AcquisitionMask:
    """
    Acquire a uniformly random subset of each query's available features.

    The subset size is uniform from none to all available features, or to
    all but one with `keep_one_unacquired`.
    """
    scores = torch.rand(available.shape, generator=generator)
    # Available features rank first, in random order.
    ranks = scores.masked_fill(~available, 2.0).argsort(-1).argsort(-1)
    n_available = available.sum(dim=-1, keepdim=True)
    n_sizes = n_available if keep_one_unacquired else n_available + 1
    sizes = (
        torch.rand(n_available.shape, generator=generator) * n_sizes
    ).floor()
    return ranks < sizes
