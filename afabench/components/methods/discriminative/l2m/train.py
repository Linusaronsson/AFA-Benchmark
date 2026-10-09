"""
L2M training stage: paper Algorithm 2 (arXiv:2510.12624), then the context set.

The pretrained `L2MModel` from `pretrain_l2m` is loaded and its policy head
trained with straight-through Gumbel-softmax at a fixed temperature, while
the backbone and the built-in classifier are fine-tuned at a lower learning
rate (paper Appendix A.5.6). Tasks come from the task prior through
`sample_task`; of the train split only the features are used, as the
pretraining pool of the real feature source. Afterwards the context set is
drawn from the validation split with the contract seed and packaged with
the model as an `L2MAFAMethod`.

Reading of Algorithm 2, where the paper is ambiguous:

- The state advances by a random available feature (line 9), not by the
  policy's action: each query instance gets one random feature mask over
  its retrospectively available features, leaving at least one unacquired,
  as `pretrain_l2m` does for Algorithm 1. The straight-through action only
  builds the one-step loss (line 8): the feature mask plus the one-hot
  action is the input of a second forward pass, whose classifier
  cross-entropy on the query instance's label is the loss. Gradients reach
  the policy through the relaxed action in the mask.
- The policy is blocked on retrospectively missing features (paper
  Definition 4.1) and, the safe choice the paper leaves unstated, on
  features already acquired.
- The one-step loss is the only loss; the classifier's loss on the current
  state is not added.
- Query instances whose available features are all acquired have no
  transition and are left out of the loss.
- The paper states no checkpoint rule for the policy. The checkpoint is
  chosen every `checkpoint_interval` steps on a fixed set of held-out tasks
  from the task prior, by the one-step loss of the argmax action.
- Learning rates are constant, as the paper states for this stage.
"""

from collections.abc import Callable
from functools import partial

import torch
from jaxtyping import Float
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
    require_direct_unmasker,
)
from afabench.components.methods.discriminative.l2m.config import (
    L2MTrainingConfig,
)
from afabench.components.methods.discriminative.l2m.fit_loop import (
    fit_on_task_prior,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
    PolicyLogits,
)
from afabench.components.methods.discriminative.l2m.task_batches import (
    TaskBatch,
)
from afabench.core.types import Features, Label
from afabench.fit.inputs import FitInputs

type QueryActions = Float[torch.Tensor, "tasks query_instances n_features"]


def train_l2m(
    cfg: L2MTrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> L2MAFAMethod:
    _check_config(cfg, inputs)
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    val_dataset = inputs.val_dataset()
    n_features = train_dataset.feature_shape.numel()

    model = inputs.pretrained_model(L2MModel).to(device)
    if (model.n_features, model.n_classes) != (
        n_features,
        train_dataset.label_shape.numel(),
    ):
        msg = (
            f"The pretrained model has {model.n_features} features and "
            f"{model.n_classes} classes; the train split has {n_features} "
            f"and {train_dataset.label_shape.numel()}"
        )
        raise ValueError(msg)
    # The Gumbel noise is drawn from the generator of the tasks.
    generator = torch.Generator().manual_seed(cfg.seed)
    fit_on_task_prior(
        model,
        _two_rate_optimizer(
            model, policy_lr=cfg.policy_lr, backbone_lr=cfg.backbone_lr
        ),
        cfg,
        stage="train",
        train_dataset=train_dataset,
        generator=generator,
        training_loss=partial(
            _one_step_loss,
            model,
            temperature=cfg.temperature,
            generator=generator,
        ),
        validation_loss=partial(_one_step_loss, model, temperature=None),
        keep_one_unacquired=True,
        scheduler=None,
        metric_logger=metric_logger,
    )
    context_features, context_labels = _draw_context_set(
        val_dataset.get_all_data(),
        n_features=n_features,
        context_set_size=cfg.context_set_size,
        seed=cfg.seed,
    )
    return L2MAFAMethod(
        model.cpu().eval(),
        context_features,
        context_labels,
        unmasker=cfg.unmasker,
    )


def _check_config(cfg: L2MTrainingConfig, inputs: FitInputs) -> None:
    require_direct_unmasker(cfg.unmasker)
    if cfg.context_set_size < 1:
        msg = f"context_set_size={cfg.context_set_size} must be at least 1"
        raise ValueError(msg)
    n_validation = len(inputs.val_dataset())
    if n_validation < cfg.context_set_size:
        msg = (
            f"The validation split has {n_validation} instances, fewer "
            f"than context_set_size={cfg.context_set_size}"
        )
        raise ValueError(msg)
    n_selections = inputs.unmasker().get_n_selections(
        inputs.train_dataset().feature_shape
    )
    # L2M has no stop action, so the budget must leave a selection for
    # every act call of the hard-budget setting.
    if cfg.hard_budget is None or not 0 < cfg.hard_budget < n_selections:
        msg = (
            f"hard_budget={cfg.hard_budget} must be between 1 and "
            f"the number of selections minus one ({n_selections - 1})"
        )
        raise ValueError(msg)


def _two_rate_optimizer(
    model: L2MModel, *, policy_lr: float, backbone_lr: float
) -> torch.optim.Optimizer:
    """Adam: the policy head at `policy_lr`, the rest at `backbone_lr`."""
    policy_parameters = list(model.policy_head.parameters())
    policy_parameter_ids = {id(parameter) for parameter in policy_parameters}
    return torch.optim.Adam(
        [
            {"params": policy_parameters, "lr": policy_lr},
            {
                "params": [
                    parameter
                    for parameter in model.parameters()
                    if id(parameter) not in policy_parameter_ids
                ],
                "lr": backbone_lr,
            },
        ]
    )


def _one_step_loss(
    model: L2MModel,
    batch: TaskBatch,
    *,
    temperature: float | None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Classifier cross-entropy after acquiring the policy's action.

    With a temperature the action is a straight-through Gumbel-softmax
    sample; without one it is the argmax, for validation.
    """
    batch = batch.to(model.device)
    features, mask, labels = batch.features, batch.mask, batch.labels
    context_set_size = batch.context_set_size
    _, policy_logits = model(
        features, mask, labels, context_set_size=context_set_size
    )
    selectable = (
        batch.available[:, context_set_size:] & ~mask[:, context_set_size:]
    )
    has_selection = selectable.any(dim=-1)
    # Query instances without a selection get a harmless uniform policy and
    # are weighted out of the loss below.
    blocked_logits = policy_logits.masked_fill(
        ~(selectable | ~has_selection.unsqueeze(-1)), -torch.inf
    )
    if temperature is None:
        action = F.one_hot(blocked_logits.argmax(dim=-1), model.n_features).to(
            features.dtype
        )
    else:
        action = _straight_through_gumbel_softmax(
            blocked_logits, temperature=temperature, generator=generator
        )
    acquired_mask = torch.cat(
        (
            mask[:, :context_set_size].to(features.dtype),
            mask[:, context_set_size:].to(features.dtype) + action,
        ),
        dim=1,
    )
    classifier_logits, _ = model(
        features, acquired_mask, labels, context_set_size=context_set_size
    )
    query_losses = F.cross_entropy(
        classifier_logits.flatten(0, 1),
        labels[:, context_set_size:].argmax(dim=-1).flatten(),
        reduction="none",
    )
    weights = has_selection.flatten().to(query_losses.dtype)
    return (query_losses * weights).sum() / weights.sum().clamp(min=1.0)


def _straight_through_gumbel_softmax(
    logits: PolicyLogits,
    *,
    temperature: float,
    generator: torch.Generator | None,
) -> QueryActions:
    """One-hot in the forward pass, relaxed softmax in the backward pass."""
    gumbel_noise = (
        -torch.empty(logits.shape, dtype=logits.dtype)
        .exponential_(generator=generator)
        .log()
    )
    relaxed = F.softmax(
        (F.log_softmax(logits, dim=-1) + gumbel_noise.to(logits.device))
        / temperature,
        dim=-1,
    )
    hard = F.one_hot(relaxed.argmax(dim=-1), logits.shape[-1]).to(
        relaxed.dtype
    )
    return hard - relaxed.detach() + relaxed


def _draw_context_set(
    validation_data: tuple[Features, Label],
    *,
    n_features: int,
    context_set_size: int,
    seed: int,
) -> tuple[Features, Label]:
    """Draw the context set from the validation split with the contract seed."""
    features, labels = validation_data
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)[
        :context_set_size
    ]
    return (
        features.reshape(len(features), n_features)[indices].clone(),
        labels.reshape(len(labels), -1)[indices].clone(),
    )
