"""Train the JAFA AFA method from the `jafa` pretrained model."""

import logging
from typing import Any, override

import torch
from tensordict import TensorDictBase
from torch import optim
from torch.nn import functional as F

from afabench.components.methods.rl.common.afa_methods import RLAFAMethod
from afabench.components.methods.rl.common.agent_interface import Agent
from afabench.components.methods.rl.common.custom_types import AFARewardFn
from afabench.components.methods.rl.common.training import RLTrainer
from afabench.components.methods.rl.jafa.agents import JAFAAgent
from afabench.components.methods.rl.jafa.config import JAFATrainConfig
from afabench.components.methods.rl.jafa.models import (
    JAFAAFAClassifier,
    LitJAFAEmbedderClassifier,
)
from afabench.components.methods.rl.jafa.reward import get_jafa_reward_fn
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.types import AFAMethod
from afabench.fit.inputs import FitInputs
from afabench.fit.metric_logger import MetricLogger

log = logging.getLogger(__name__)


def train_jafa(
    cfg: JAFATrainConfig, inputs: FitInputs, metric_logger: MetricLogger
) -> AFAMethod:
    trainer = JAFARLTrainer(cfg, inputs, metric_logger)
    return trainer.train(cfg=cfg.rl_training_loop)


class JAFARLTrainer(RLTrainer):
    pretrained_model: LitJAFAEmbedderClassifier
    pretrained_model_optim: torch.optim.Adam
    replay_buffer_device: torch.device
    activate_joint_training_after_batch: int
    typed_cfg: JAFATrainConfig

    def __init__(
        self,
        cfg: JAFATrainConfig,
        inputs: FitInputs,
        metric_logger: MetricLogger,
    ) -> None:
        self.typed_cfg = cfg
        super().__init__(cfg, inputs, cfg.mdp, metric_logger)

        self.activate_joint_training_after_batch = int(
            self.typed_cfg.rl_training_loop.n_batches
            * self.typed_cfg.activate_joint_training_after_fraction
        )

    @override
    def _setup_subclass_specific_state(self) -> None:
        self.replay_buffer_device = self.device
        pretrained_model = self.inputs.pretrained_model(TorchModelBundle).model
        if not isinstance(pretrained_model, LitJAFAEmbedderClassifier):
            msg = (
                "JAFA expects the jafa pretrained model "
                "(LitJAFAEmbedderClassifier), got "
                f"{type(pretrained_model).__name__}."
            )
            raise TypeError(msg)
        pretrained_model.eval()
        self.pretrained_model = pretrained_model.to(self.device)
        self.pretrained_model_optim = optim.Adam(
            self.pretrained_model.parameters(),
            lr=self.typed_cfg.pretrained_model_lr,
        )

    @override
    def _get_reward_fn(self) -> AFARewardFn:
        return get_jafa_reward_fn(
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
        is_hard_budget_mode = self.typed_cfg.soft_budget_param is None
        return JAFAAgent(
            cfg=self.typed_cfg.agent,
            embedder=self.pretrained_model.embedder,
            embedding_size=self.pretrained_model.embedder.encoder.output_size,
            action_spec=self.train_env.action_spec,
            action_mask_key="allowed_action_mask",
            module_device=self.device,
            n_feature_dims=len(self.train_dataset.feature_shape),
            n_batches=self.typed_cfg.rl_training_loop.n_batches,
            allow_stop_action=not is_hard_budget_mode,
        )

    @override
    def _get_afa_method(self, device: torch.device) -> AFAMethod:
        return RLAFAMethod(
            self.agent.get_exploitative_policy().to(device),
            JAFAAFAClassifier(self.pretrained_model, device=device),
            device,
        )

    @override
    def _post_process_batch(
        self, td: TensorDictBase, batch_idx: int
    ) -> dict[str, Any]:
        assert td.batch_dims == 2, "Expected two batch dimensions"

        if batch_idx >= self.activate_joint_training_after_batch:
            if batch_idx == self.activate_joint_training_after_batch:
                log.info(
                    "Activating joint training of classifier and embedder"
                )
            self.pretrained_model.train()
            self.pretrained_model_optim.zero_grad()

            n_feature_dims = len(self.train_dataset.feature_shape)

            # Flatten feature dims
            flat_masked_features = td["next", "masked_features"].flatten(
                start_dim=-n_feature_dims
            )
            flat_feature_mask = td["next", "feature_mask"].flatten(
                start_dim=-n_feature_dims
            )
            assert flat_masked_features.ndim == td["next", "label"].ndim, (
                "Label should be 1D"
            )

            # Flatten batch dims
            flat_masked_features = flat_masked_features.flatten(end_dim=-2)
            flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
            flat_label = td["next", "label"].flatten(end_dim=-2)

            _, logits_next = self.pretrained_model(
                flat_masked_features, flat_feature_mask
            )
            class_loss_next = F.cross_entropy(
                logits_next,
                flat_label,
                weight=self.class_weights,
            )
            class_loss_next.mean().backward()

            self.pretrained_model_optim.step()
            self.pretrained_model.eval()

            return {"avg_class_loss": class_loss_next.mean().cpu().item()}
        return {}

    @override
    def _pre_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.eval_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def _post_eval(self) -> None:
        self.agent.egreedy_tdmodule._spec = self.train_env.action_spec  # noqa: SLF001  # pyright: ignore[reportAttributeAccessIssue]
