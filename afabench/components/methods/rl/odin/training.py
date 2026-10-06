"""Train the ODIN AFA method from the `pvae` pretrained model."""

import logging
from dataclasses import replace
from typing import Any, override

import torch
from torch.nn import functional as F

from afabench.components.methods.rl.common.afa_methods import RLAFAMethod
from afabench.components.methods.rl.common.agent_interface import Agent
from afabench.components.methods.rl.common.custom_types import AFARewardFn
from afabench.components.methods.rl.common.training import (
    RLTrainer,
    limit_training_loop,
)
from afabench.components.methods.rl.odin.agents import ODINAgent
from afabench.components.methods.rl.odin.config import ODINTrainConfig
from afabench.components.methods.rl.odin.models import (
    ODINAFAClassifier,
    ODINPretrainingModel,
)
from afabench.components.methods.rl.odin.reward import get_odin_reward_fn
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.types import AFAMethod, Features, Label
from afabench.datasets.wrappers import ExtendedAFADataset
from afabench.training.inputs import TrainingInputs
from afabench.training.metric_logger import MetricLogger

log = logging.getLogger(__name__)


def train_odin(
    cfg: ODINTrainConfig, inputs: TrainingInputs, metric_logger: MetricLogger
) -> AFAMethod:
    cfg = replace(
        cfg,
        rl_training_loop=limit_training_loop(
            cfg.rl_training_loop, smoke_test=cfg.smoke_test
        ),
    )
    trainer = ODINRLTrainer(cfg, inputs, metric_logger)
    return trainer.train(cfg=cfg.rl_training_loop)


def generate_data_batched(
    pretrained_model: ODINPretrainingModel,
    samples: int,
    batch_size: int,
) -> tuple[Features, Label]:
    """Generate synthetic data using the generative model in batches. Data is placed on cpu, since there might be a lot of it."""
    generated_flat_features = torch.zeros(
        samples, pretrained_model.n_output_features
    )
    generated_labels = torch.zeros(samples, pretrained_model.n_classes)
    n_full_batches = samples // batch_size
    n_samples_rest = samples % batch_size
    # Add full batches
    batch_plan = [
        (i * batch_size, (i + 1) * batch_size, batch_size)
        for i in range(n_full_batches)
    ]
    # Add remainder batch
    if n_samples_rest > 0:
        batch_plan.append(
            (n_full_batches * batch_size, samples, n_samples_rest)
        )

    for start, end, curr_batch_size in batch_plan:
        _z, flat_batch, label_batch = pretrained_model.generate_data(
            n_samples=curr_batch_size
        )
        generated_flat_features[start:end, :] = flat_batch.cpu()
        generated_labels[start:end, :] = F.one_hot(
            label_batch.argmax(-1),
            num_classes=label_batch.shape[-1],
        ).cpu()
    return generated_flat_features, generated_labels


class ODINRLTrainer(RLTrainer):
    pretrained_model: ODINPretrainingModel
    extended_train_dataset: ExtendedAFADataset | Any
    typed_cfg: ODINTrainConfig

    def __init__(
        self,
        cfg: ODINTrainConfig,
        inputs: TrainingInputs,
        metric_logger: MetricLogger,
    ) -> None:
        self.typed_cfg = cfg
        super().__init__(cfg, inputs, cfg.mdp, metric_logger)

    @override
    def _setup_subclass_specific_state(self) -> None:
        """Load pretrained model and generate synthetic data if needed."""
        pretrained_model = self.inputs.pretrained_model(TorchModelBundle).model
        if not isinstance(pretrained_model, ODINPretrainingModel):
            msg = (
                "ODIN expects the pvae pretrained model "
                f"(ODINPretrainingModel), got {type(pretrained_model).__name__}."
            )
            raise TypeError(msg)
        pretrained_model.eval()
        self.pretrained_model = pretrained_model.to(self.device)

        # odin unique step: generate additional data using generative model
        if self.typed_cfg.additional_generation_fraction > 0.0:
            n_artificial_samples = int(
                self.typed_cfg.additional_generation_fraction
                * len(self.train_dataset)
            )
            additional_features, additional_labels = generate_data_batched(
                pretrained_model=self.pretrained_model,
                samples=n_artificial_samples,
                batch_size=self.typed_cfg.generation_batch_size,
            )
            self.extended_train_dataset = ExtendedAFADataset(
                base_dataset=self.train_dataset,
                additional_features=additional_features.view(
                    (-1, *self.train_dataset.feature_shape)
                ),
                additional_labels=additional_labels,
            )
        else:
            self.extended_train_dataset = self.train_dataset

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        return get_odin_reward_fn(
            pretrained_model=self.pretrained_model,
            weights=self.class_weights,
            selection_costs=(
                0
                if self.typed_cfg.soft_budget_param is None
                else self.typed_cfg.soft_budget_param
            )
            * self.normalized_selection_costs.to(self.device),
            n_feature_dims=self._n_feature_dims,
        )

    @override
    def _get_agent(self) -> Agent:
        return ODINAgent(
            cfg=self.typed_cfg.agent,
            pointnet=self.pretrained_model.partial_vae.pointnet,
            encoder=self.pretrained_model.partial_vae.encoder,
            action_spec=self.train_env.action_spec,
            latent_size=self.pretrained_model.latent_size,
            action_mask_key="allowed_action_mask",
            frames_per_batch=self.typed_cfg.rl_training_loop.frames_per_batch,
            module_device=self.device,
            n_feature_dims=len(self.extended_train_dataset.feature_shape),
        )

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        return RLAFAMethod(
            self.agent.get_exploitative_policy().to(device),
            ODINAFAClassifier(self.pretrained_model, device=device),
            device,
        )

    @override
    def _create_envs(self) -> None:
        """Create environments using the extended training dataset."""
        self.train_env = self._get_env_from_dataset(  # pyright: ignore[reportUnannotatedClassAttribute]
            self.extended_train_dataset
        )
        self.eval_env = self._get_env_from_dataset(self.val_dataset)  # pyright: ignore[reportUnannotatedClassAttribute]
