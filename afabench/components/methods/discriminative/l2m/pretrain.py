"""
L2M pretraining stage: paper Algorithm 1 (arXiv:2510.12624).

The built-in classifier of `L2MModel` learns to predict the labels of
queries from a context set, on tasks drawn from the task prior with
`sample_task`. Of the train split only the features are used, as the
pretraining pool of the real feature source; its labels are discarded.

Choices where the paper is ambiguous, or where this port departs from it:

- Algorithm 1 visits every acquisition mask size of every query. Each
  query here gets one random acquisition mask instead, so a task costs one
  forward pass rather than one per mask size: a size drawn uniformly from
  none to all of the query's retrospectively available features, then a
  uniformly random subset of that size.
- The paper sums the loss over context sizes using training-only target
  points. Here each step draws one context size, uniform on 1 to
  `sequence_length - 1` and shared by the step's tasks, and the remaining
  instances are queries. The loss is the cross-entropy averaged over the
  step's queries.
- The checkpoint is chosen on a fixed set of held-out tasks from the same
  task prior, as in the paper, which does not state how many.
- Features are not normalized within each task, unlike the paper, because
  `L2MAFAMethod` does not normalize them at evaluation either.
- The learning rate warms up linearly, then decays linearly towards zero.
"""

import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial

import torch
from jaxtyping import Bool
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m.config import (
    L2MPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
    TaskFeatures,
    TaskLabels,
    TaskMask,
)
from afabench.components.methods.discriminative.l2m.task_sampler import (
    FeatureSource,
    sample_task,
)
from afabench.core.types import Features
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)

type AvailableFeatures = Bool[torch.Tensor, "tasks queries n_features"]
type AcquisitionMask = Bool[torch.Tensor, "tasks queries n_features"]

# Task seeds are drawn from one generator per run, so the held-out tasks
# and the training tasks are distinct draws from the task prior.
_TASK_SEED_BOUND = 2**62


@dataclass(frozen=True, kw_only=True)
class _TaskBatch:
    """
    Tasks of one step, with the context set first and queries last.

    `mask` holds the retrospective missingness mask of context instances
    and the random acquisition mask of queries.
    """

    features: TaskFeatures
    mask: TaskMask
    labels: TaskLabels
    n_context: int


def pretrain_l2m(
    cfg: L2MPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> L2MModel:
    feature_source = _feature_source(cfg.feature_source)
    if cfg.sequence_length < 2:
        msg = (
            f"sequence_length={cfg.sequence_length} must be at least 2, "
            "for one context instance and one query"
        )
        raise ValueError(msg)
    if cfg.n_steps < 1:
        msg = f"n_steps={cfg.n_steps} must be at least 1"
        raise ValueError(msg)
    device = torch.device(cfg.device)

    train_dataset = inputs.train_dataset()
    n_features = train_dataset.feature_shape.numel()
    features, _ = train_dataset.get_all_data()
    draw_batch = partial(
        _draw_task_batch,
        feature_source=feature_source,
        feature_pool=features.reshape(len(features), n_features)
        if feature_source == "real"
        else None,
        n_features=n_features,
        label_shape=train_dataset.label_shape,
        sequence_length=cfg.sequence_length,
        missingness_cap=cfg.missingness_cap,
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    validation_batches = [
        draw_batch(
            min(cfg.batch_size, cfg.n_validation_tasks - start), generator
        )
        for start in range(0, cfg.n_validation_tasks, cfg.batch_size)
    ]

    model = L2MModel(
        n_features,
        train_dataset.label_shape.numel(),
        **asdict(cfg.architecture),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            _learning_rate_factor,
            warmup_steps=cfg.warmup_steps,
            n_steps=cfg.n_steps,
        ),
    )

    best_loss = torch.inf
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, cfg.n_steps + 1):
        model.train()
        loss = _query_loss(model, draw_batch(cfg.batch_size, generator))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % cfg.checkpoint_interval and step != cfg.n_steps:
            continue

        model.eval()
        with torch.no_grad():
            validation_loss = sum(
                float(_query_loss(model, batch))
                for batch in validation_batches
            ) / len(validation_batches)
        log.info(
            "L2M pretraining step %d: train loss %.4f, validation loss %.4f",
            step,
            float(loss),
            validation_loss,
        )
        if metric_logger is not None:
            metric_logger(
                {
                    "l2m_pretrain/step": float(step),
                    "l2m_pretrain/train_loss": float(loss),
                    "l2m_pretrain/val_loss": validation_loss,
                }
            )
        if best_state is None or validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

    # The last step always checkpoints, and there is at least one step.
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.cpu().eval()


def _feature_source(value: str) -> FeatureSource:
    if value in ("real", "synthetic"):
        return value
    msg = f"feature_source must be 'real' or 'synthetic'; got {value!r}"
    raise ValueError(msg)


def _draw_task_batch(
    n_tasks: int,
    generator: torch.Generator,
    *,
    feature_source: FeatureSource,
    feature_pool: Features | None,
    n_features: int,
    label_shape: torch.Size,
    sequence_length: int,
    missingness_cap: float,
) -> _TaskBatch:
    tasks = [
        sample_task(
            feature_source,
            n_features=n_features,
            sequence_length=sequence_length,
            label_shape=label_shape,
            missingness_cap=missingness_cap,
            feature_pool=feature_pool,
            seed=int(torch.randint(_TASK_SEED_BOUND, (), generator=generator)),
        )
        for _ in range(n_tasks)
    ]
    available = torch.stack([task.feature_mask for task in tasks]).bool()
    n_context = int(torch.randint(1, sequence_length, (), generator=generator))
    return _TaskBatch(
        features=torch.stack([task.features for task in tasks]),
        mask=torch.cat(
            (
                available[:, :n_context],
                _random_acquisition_mask(available[:, n_context:], generator),
            ),
            dim=1,
        ),
        labels=torch.stack([task.labels for task in tasks]),
        n_context=n_context,
    )


def _random_acquisition_mask(
    available: AvailableFeatures, generator: torch.Generator
) -> AcquisitionMask:
    scores = torch.rand(available.shape, generator=generator)
    # Available features rank first, in random order.
    ranks = scores.masked_fill(~available, 2.0).argsort(-1).argsort(-1)
    n_available = available.sum(dim=-1, keepdim=True)
    sizes = (
        torch.rand(n_available.shape, generator=generator) * (n_available + 1)
    ).floor()
    return ranks < sizes


def _query_loss(model: L2MModel, batch: _TaskBatch) -> torch.Tensor:
    logits, _ = model(
        batch.features.to(model.device),
        batch.mask.to(model.device),
        batch.labels.to(model.device),
        n_context=batch.n_context,
    )
    targets = batch.labels[:, batch.n_context :].argmax(dim=-1)
    return F.cross_entropy(
        logits.flatten(0, 1), targets.flatten().to(model.device)
    )


def _learning_rate_factor(
    step: int, *, warmup_steps: int, n_steps: int
) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return (n_steps - step) / max(1, n_steps - warmup_steps)
