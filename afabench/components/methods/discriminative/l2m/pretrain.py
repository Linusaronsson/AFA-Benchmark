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
- The paper sums the loss over context set sizes using training-only target
  points. Here each step draws one context set size, uniform on 1 to
  `sequence_length - 1` and shared by the step's tasks, and the remaining
  instances are queries. The loss is the cross-entropy averaged over the
  step's queries.
- The checkpoint is chosen on a fixed set of held-out tasks from the same
  task prior, as in the paper, which does not state how many.
- Features are not normalized within each task, unlike the paper. The
  paper does not say which statistics normalize the context set and
  queries at evaluation, where queries are only partly observed, and the
  MiniBooNE loader already z-normalizes every feature over the dataset.
- The learning rate warms up linearly, then decays linearly towards zero.
"""

from collections.abc import Callable
from dataclasses import asdict
from functools import partial

import torch
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m.config import (
    L2MPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m.fit_loop import (
    fit_on_task_prior,
)
from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.methods.discriminative.l2m.task_batches import (
    TaskBatch,
)
from afabench.fit.inputs import FitInputs


def pretrain_l2m(
    cfg: L2MPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> L2MModel:
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    model = L2MModel(
        train_dataset.feature_shape.numel(),
        train_dataset.label_shape.numel(),
        **asdict(cfg.architecture),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    fit_on_task_prior(
        model,
        optimizer,
        cfg,
        stage="pretrain",
        train_dataset=train_dataset,
        generator=torch.Generator().manual_seed(cfg.seed),
        training_loss=partial(_query_loss, model),
        validation_loss=partial(_query_loss, model),
        keep_one_unacquired=False,
        scheduler=torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(
                _learning_rate_factor,
                warmup_steps=cfg.warmup_steps,
                n_steps=cfg.n_steps,
            ),
        ),
        metric_logger=metric_logger,
    )
    return model.cpu().eval()


def _query_loss(model: L2MModel, batch: TaskBatch) -> torch.Tensor:
    batch = batch.to(model.device)
    logits, _ = model(
        batch.features,
        batch.mask,
        batch.labels,
        context_set_size=batch.context_set_size,
    )
    targets = batch.labels[:, batch.context_set_size :].argmax(dim=-1)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def _learning_rate_factor(
    step: int, *, warmup_steps: int, n_steps: int
) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return (n_steps - step) / max(1, n_steps - warmup_steps)
