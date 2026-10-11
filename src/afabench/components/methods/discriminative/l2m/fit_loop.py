"""
The fit loop shared by L2M's two fit stages.

Both `pretrain_l2m` and `train_l2m` train on batches of tasks from the
task prior, drawn with `draw_task_batch`. Of the train split only the
features are used, as the pretraining pool of the real feature source.
The state kept is the one with the lowest loss on a fixed set of held-out
tasks from the same prior, checked every `checkpoint_interval` steps and at
the last step.
"""

import logging
from collections.abc import Callable
from functools import partial
from typing import Protocol

import torch

from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.methods.discriminative.l2m.task_batches import (
    TaskBatch,
    draw_task_batch,
)
from afabench.components.methods.discriminative.l2m.task_sampler import (
    FeatureSource,
)
from afabench.core.types import AFADataset

log = logging.getLogger(__name__)


class TaskPriorFitConfig(Protocol):
    """The fields of `L2MPretrainingConfig` and `L2MTrainingConfig` used here."""

    @property
    def feature_source(self) -> FeatureSource: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def sequence_length(self) -> int: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def checkpoint_interval(self) -> int: ...
    @property
    def n_validation_tasks(self) -> int: ...
    @property
    def missingness_cap(self) -> float: ...


def fit_on_task_prior(
    model: L2MModel,
    optimizer: torch.optim.Optimizer,
    cfg: TaskPriorFitConfig,
    *,
    stage: str,
    train_dataset: AFADataset,
    generator: torch.Generator,
    training_loss: Callable[[TaskBatch], torch.Tensor],
    validation_loss: Callable[[TaskBatch], torch.Tensor],
    keep_one_unacquired: bool,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    metric_logger: Callable[[dict[str, float]], None] | None,
) -> None:
    """
    Fit `model` in place, then load its best checkpoint.

    `stage` names the log lines and the `l2m_<stage>/` metrics.
    `generator` draws the held-out tasks, then every step's tasks.
    """
    if cfg.n_steps < 1:
        msg = f"n_steps={cfg.n_steps} must be at least 1"
        raise ValueError(msg)
    n_features = train_dataset.feature_shape.numel()
    features, _ = train_dataset.get_all_data()
    draw_batch = partial(
        draw_task_batch,
        feature_source=cfg.feature_source,
        feature_pool=features.reshape(len(features), n_features)
        if cfg.feature_source is FeatureSource.real
        else None,
        n_features=n_features,
        label_shape=train_dataset.label_shape,
        sequence_length=cfg.sequence_length,
        missingness_cap=cfg.missingness_cap,
        keep_one_unacquired=keep_one_unacquired,
    )
    validation_batches = [
        draw_batch(
            min(cfg.batch_size, cfg.n_validation_tasks - start), generator
        )
        for start in range(0, cfg.n_validation_tasks, cfg.batch_size)
    ]

    best_loss = torch.inf
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, cfg.n_steps + 1):
        model.train()
        loss = training_loss(draw_batch(cfg.batch_size, generator))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % cfg.checkpoint_interval and step != cfg.n_steps:
            continue

        model.eval()
        with torch.no_grad():
            mean_validation_loss = sum(
                float(validation_loss(batch)) for batch in validation_batches
            ) / len(validation_batches)
        log.info(
            "L2M %s step %d: train loss %.4f, validation loss %.4f",
            stage,
            step,
            float(loss),
            mean_validation_loss,
        )
        if metric_logger is not None:
            metric_logger(
                {
                    f"l2m_{stage}/step": float(step),
                    f"l2m_{stage}/train_loss": float(loss),
                    f"l2m_{stage}/val_loss": mean_validation_loss,
                }
            )
        if best_state is None or mean_validation_loss < best_loss:
            best_loss = mean_validation_loss
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }

    # The last step always checkpoints, and there is at least one step.
    assert best_state is not None
    model.load_state_dict(best_state)
