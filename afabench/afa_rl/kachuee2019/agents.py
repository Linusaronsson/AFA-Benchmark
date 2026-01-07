from typing import Any, final, override

import torch
from rl_helpers import module_norm
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import Tensor, nn, optim
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorSpec,
)
from torchrl.modules import EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.kachuee2019.models import Kachuee2019PQModule
from afabench.common.config_classes import Kachuee2019AgentConfig
from afabench.common.custom_types import MaskedFeatures


@final
class Kachuee2019ActionValueModule(nn.Module):
    def __init__(self, pq_module: Kachuee2019PQModule, n_feature_dims: int):
        super().__init__()

        self.pq_module = pq_module
        self.n_feature_dims = n_feature_dims

    @override
    def forward(
        self, masked_features: MaskedFeatures, action_mask: Tensor
    ) -> Tensor:
        # Flatten feature dimensions
        flat_masked_features = masked_features.flatten(
            start_dim=-self.n_feature_dims
        )
        # Flatten batch dimensions
        flat_masked_features = flat_masked_features.flatten(end_dim=-2)
        # pq_module.forward ensures that gradients are not backpropagated to the P network
        _class_logits, qvalues = self.pq_module.forward(flat_masked_features)

        # Unflatten batch dimensions
        qvalues = qvalues.unflatten(
            0, masked_features.shape[: -self.n_feature_dims]
        )

        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


@final
class Kachuee2019Agent(Agent):
    def __init__(
        self,
        cfg: Kachuee2019AgentConfig,
        pq_module: Kachuee2019PQModule,
        action_spec: TensorSpec,
        action_mask_key: str,
        module_device: torch.device,  # device to place nn.Modules on
        replay_buffer_device: torch.device,  # device to place replay buffer on
        n_feature_dims: int,  # how many dimensions the feature shape needs. Used to flatten the features before they are passed to the pq module
        n_batches: int,  # total number of batches expected during training, needed for eps annealing
    ):
        self.cfg = cfg
        self.pq_module = pq_module.to(module_device)
        self.action_spec = action_spec
        self.action_mask_key = action_mask_key
        self.module_device = module_device
        self.replay_buffer_device = replay_buffer_device
        self.n_feature_dims = n_feature_dims

        self.action_value_module = Kachuee2019ActionValueModule(
            pq_module=self.pq_module, n_feature_dims=self.n_feature_dims
        ).to(self.module_device)

        self.action_value_tdmodule = TensorDictModule(
            module=self.action_value_module,
            in_keys=["masked_features", self.action_mask_key],
            out_keys=["action_value"],
        )

        self.greedy_tdmodule = QValueModule(
            spec=self.action_spec,
            action_mask_key=self.action_mask_key,
            action_value_key="action_value",
            out_keys=["action", "action_value", "chosen_action_value"],
        ).to(self.module_device)

        self.greedy_policy_tdmodule = TensorDictSequential(
            [self.action_value_tdmodule, self.greedy_tdmodule]
        )

        eps_annealing_num_steps = int(
            n_batches * self.cfg.eps_annealing_fraction
        )
        self.egreedy_tdmodule = EGreedyModule(
            spec=self.action_spec,
            action_key="action",
            action_mask_key=self.action_mask_key,
            annealing_num_steps=eps_annealing_num_steps,
            eps_init=self.cfg.eps_init,
            eps_end=self.cfg.eps_end,
        ).to(self.module_device)

        self.egreedy_policy_tdmodule = TensorDictSequential(
            [self.greedy_policy_tdmodule, self.egreedy_tdmodule]
        )

        self.loss_tdmodule = DQNLoss(
            value_network=self.greedy_policy_tdmodule,
            loss_function=self.cfg.loss_function,
            delay_value=self.cfg.delay_value,
            double_dqn=self.cfg.double_dqn,
            action_space=self.action_spec,
        ).to(self.module_device)

        self.loss_tdmodule.make_value_estimator(
            ValueEstimators.TDLambda,
            gamma=self.cfg.gamma,
            lmbda=self.cfg.lmbda,
        )

        if self.cfg.delay_value:
            self.target_net_updater = SoftUpdate(
                self.loss_tdmodule, eps=1 - self.cfg.update_tau
            )
        else:
            self.target_net_updater = None

        self.optimizer = optim.Adam(
            self.loss_tdmodule.parameters(), lr=self.cfg.lr
        )

        self.replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(
                max_size=self.cfg.replay_buffer_size,
                device=self.replay_buffer_device,
            ),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.cfg.replay_buffer_batch_size,
        )

    @override
    def get_exploitative_policy(self) -> TensorDictModuleBase:
        return self.greedy_policy_tdmodule

    @override
    def get_exploratory_policy(self) -> TensorDictModuleBase:
        return self.egreedy_policy_tdmodule

    @override
    def get_policy(self) -> TensorDictModuleBase:
        return self.get_exploratory_policy()

    @override
    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, Any]:
        return {}

    @override
    def get_cheap_info(self) -> dict[str, Any]:
        return {
            "eps": self.egreedy_tdmodule.eps.item(),  # pyright: ignore[reportCallIssue]
            "replay_buffer_count": len(self.replay_buffer),
        }

    @override
    def get_expensive_info(self) -> dict[str, Any]:
        return {
            "p_net_norm": module_norm(self.pq_module.layers_p),
            "q_net_norm": module_norm(self.pq_module.layers_q),
        }

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        self.replay_buffer.extend(td)

        # Initialize total loss dictionary
        total_loss_dict = {"loss": 0.0}
        td_errors = []

        for _ in range(self.cfg.num_epochs):
            sampled_td = self.replay_buffer.sample()
            if self.replay_buffer_device != self.module_device:
                sampled_td = sampled_td.to(self.module_device)

            self.optimizer.zero_grad()
            loss_td: TensorDictBase = self.loss_tdmodule(sampled_td)
            loss_tensor: Tensor = loss_td["loss"]
            loss_tensor.backward()
            nn.utils.clip_grad_norm_(
                self.loss_tdmodule.parameters(),
                max_norm=self.cfg.max_grad_norm,
            )
            self.optimizer.step()
            # Update target network
            if self.target_net_updater is not None:
                self.target_net_updater.step()

            td_errors.append(sampled_td["td_error"])

            # Accumulate losses
            total_loss_dict["loss"] += loss_td["loss"].item()

        # Anneal epsilon for epsilon greedy exploration
        self.egreedy_tdmodule.step()

        # Compute average loss
        process_dict = {
            k: v / self.cfg.num_epochs for k, v in total_loss_dict.items()
        }
        process_dict["td_error"] = torch.mean(torch.stack(td_errors)).item()

        return process_dict

    @override
    def get_module_device(self) -> torch.device:
        return self.module_device

    @override
    def set_module_device(self, device: torch.device) -> None:
        self.module_device = device

        # Send modules to device
        self.action_value_module = self.action_value_module.to(
            self.module_device
        )
        self.greedy_tdmodule = self.greedy_tdmodule.to(self.module_device)
        self.egreedy_tdmodule = self.egreedy_tdmodule.to(self.module_device)

    @override
    def get_replay_buffer_device(self) -> torch.device:
        return self.replay_buffer_device

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        msg = "Changing replay buffer device is not yet supported."
        raise ValueError(msg)
