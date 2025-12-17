from typing import Any, final, override

import torch
from afabench.afa_rl.agent_interface import Agent
from afabench.afa_rl.utils import module_norm
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from torch import Tensor, nn, optim
from torchrl.data import (
    TensorSpec,
)
from torchrl.modules import MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, ValueEstimators

from afabench.afa_rl.shim2018.models import Shim2018Embedder
from afabench.common.config_classes import Shim2018AgentConfig
from afabench.common.custom_types import FeatureMask, MaskedFeatures


@final
class Shim2018ActionValueModule(nn.Module):
    def __init__(
        self,
        embedder: Shim2018Embedder,
        embedding_size: int,
        action_size: int,
        num_cells: tuple[int, ...],
        dropout: float,
        n_feature_dims: int,
    ):
        super().__init__()
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_size = action_size
        self.num_cells = num_cells
        self.dropout = dropout
        self.n_feature_dims = n_feature_dims

        self.net = MLP(
            in_features=self.embedding_size,
            out_features=self.action_size,
            num_cells=self.num_cells,
            dropout=self.dropout,
            activation_class=nn.ReLU,
        )

    @override
    def forward(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        action_mask: Tensor,
    ) -> Tensor:
        # Flatten feature dimensions
        flat_masked_features = masked_features.flatten(
            start_dim=-self.n_feature_dims
        )
        flat_feature_mask = feature_mask.flatten(
            start_dim=-self.n_feature_dims
        )
        # Flatten batch dimensions
        flat_masked_features = flat_masked_features.flatten(end_dim=-2)
        flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
        # We do not want to update the embedder weights using the Q-values, this is done separately in the training loop
        with torch.no_grad():
            embedding = self.embedder(flat_masked_features, flat_feature_mask)
        qvalues = self.net(embedding)
        # Unflatten batch dimensions
        qvalues = qvalues.unflatten(
            0, masked_features.shape[: -self.n_feature_dims]
        )

        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


@final
class Shim2018DummyActionValueModule(nn.Module):
    def __init__(
        self,
        in_size: int,
        action_size: int,
        num_cells: tuple[int, ...],
        dropout: float,
        n_feature_dims: int,
    ):
        super().__init__()
        self.in_size = in_size
        self.action_size = action_size
        self.num_cells = num_cells
        self.dropout = dropout
        self.n_feature_dims = n_feature_dims

        self.net = MLP(
            in_features=self.in_size,
            out_features=self.action_size,
            num_cells=self.num_cells,
            dropout=self.dropout,
            activation_class=nn.ReLU,
        )

    @override
    def forward(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        action_mask: Tensor,
    ) -> Tensor:
        # Flatten feature dimensions
        flat_masked_features = masked_features.flatten(
            start_dim=-self.n_feature_dims
        )
        flat_feature_mask = feature_mask.flatten(
            start_dim=-self.n_feature_dims
        )
        # Flatten batch dimensions
        flat_masked_features = flat_masked_features.flatten(end_dim=-2)
        flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)
        qvalues = self.net(
            torch.cat([flat_masked_features, flat_feature_mask], dim=-1)
        )
        # Unflatten batch dimensions
        qvalues = qvalues.unflatten(
            0, masked_features.shape[: -self.n_feature_dims]
        )

        # By setting the Q-values of invalid actions to -inf, we prevent them from being selected greedily.
        qvalues[~action_mask] = float("-inf")
        return qvalues


@final
class Shim2018Agent(Agent):
    def __init__(
        self,
        cfg: Shim2018AgentConfig,
        embedder: Shim2018Embedder,
        embedding_size: int,  # size of the embedding produced by `embedder`
        action_spec: TensorSpec,
        action_mask_key: str,
        batch_size: int,  # expected batch size received in `process_batch`
        module_device: torch.device,  # device to place nn.Modules on
        n_feature_dims: int,  # how many dimensions the feature shape needs. Used to flatten the features before they are passed to the embedder
        n_batches: int,  # total number of batches expected during training
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_spec = action_spec
        self.action_mask_key = action_mask_key
        self.batch_size = batch_size
        self.module_device = module_device
        self.n_feature_dims = n_feature_dims

        self.action_value_module = Shim2018ActionValueModule(
            embedder=self.embedder,
            embedding_size=self.embedding_size,
            action_size=self.action_spec.n,  # pyright: ignore[reportAttributeAccessIssue]
            num_cells=tuple(self.cfg.action_value_num_cells),
            dropout=self.cfg.action_value_dropout,
            n_feature_dims=self.n_feature_dims,
        ).to(self.module_device)

        self.action_value_tdmodule = TensorDictModule(
            module=self.action_value_module,
            in_keys=["masked_features", "feature_mask", self.action_mask_key],
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

        # The shim2018 method does not use a replay buffer

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
    def get_cheap_info(self) -> dict[str, Any]:
        return {"eps": self.egreedy_tdmodule.eps.item()}  # pyright: ignore[reportCallIssue]

    @override
    def get_expensive_info(self) -> dict[str, Any]:
        return {
            "value net norm": module_norm(self.action_value_module.net),
        }

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        # Initialize total loss dictionary
        total_loss_dict = {"loss": 0.0}
        td_errors = []

        for _ in range(self.cfg.num_epochs):
            td_copy = td.clone()
            self.optimizer.zero_grad()
            loss_td: TensorDictBase = self.loss_tdmodule(td_copy)
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

            td_errors.append(td_copy["td_error"])

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
    def get_replay_buffer_device(self) -> None:
        return None

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        error_msg = (
            "set_replay_buffer_device not yet supported for Shim2018Agent"
        )
        raise ValueError(error_msg)


@final
class DummyShim2018Agent(Agent):
    def __init__(
        self,
        cfg: Shim2018AgentConfig,
        embedder: Shim2018Embedder,
        embedding_size: int,  # size of the embedding produced by `embedder`
        action_spec: TensorSpec,
        action_mask_key: str,
        batch_size: int,  # expected batch size received in `process_batch`
        module_device: torch.device,  # device to place nn.Modules on
        n_feature_dims: int,  # how many dimensions the feature shape needs. Used to flatten the features before they are passed to the embedder
        n_batches: int,  # total number of batches expected during training
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.embedding_size = embedding_size
        self.action_spec = action_spec
        self.action_mask_key = action_mask_key
        self.batch_size = batch_size
        self.module_device = module_device
        self.n_feature_dims = n_feature_dims

        self.action_value_module = Shim2018DummyActionValueModule(
            in_size=40,
            action_size=self.action_spec.n,  # pyright: ignore[reportAttributeAccessIssue]
            num_cells=tuple(self.cfg.action_value_num_cells),
            dropout=self.cfg.action_value_dropout,
            n_feature_dims=self.n_feature_dims,
        ).to(self.module_device)

        self.action_value_tdmodule = TensorDictModule(
            module=self.action_value_module,
            in_keys=["masked_features", "feature_mask", self.action_mask_key],
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

        # The shim2018 method does not use a replay buffer

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
    def get_cheap_info(self) -> dict[str, Any]:
        return {"eps": self.egreedy_tdmodule.eps.item()}  # pyright: ignore[reportCallIssue]

    @override
    def get_expensive_info(self) -> dict[str, Any]:
        return {
            "value net norm": module_norm(self.action_value_module.net),
        }

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        # assert td.batch_size == torch.Size((self.batch_size,)), (
        #     "Batch size mismatch"
        # )

        # td = tds.flatten(start_dim=0, end_dim=1)

        # Initialize total loss dictionary
        total_loss_dict = {"loss": 0.0}
        td_errors = []

        for _ in range(self.cfg.num_epochs):
            td_copy = td.clone()
            self.optimizer.zero_grad()
            loss_td: TensorDictBase = self.loss_tdmodule(td_copy)
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

            td_errors.append(td_copy["td_error"])

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
    def get_replay_buffer_device(self) -> None:
        return None

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        error_msg = (
            "set_replay_buffer_device not yet supported for Shim2018Agent"
        )
        raise ValueError(error_msg)
