from typing import Any, final, override

import torch
from jaxtyping import Bool
from tensordict import TensorDictBase
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor, nn, optim
from torch.distributions import Categorical
from torchrl.data import (
    TensorSpec,
)
from torchrl.modules import (
    MLP,
    ProbabilisticActor,
)
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.common.utils import module_norm
from afabench.afa_rl.zannone2019.models import PointNet
from afabench.common.config_classes import Zannone2019AgentConfig
from afabench.common.custom_types import FeatureMask, MaskedFeatures


@final
class Zannone2019ValueModule(nn.Module):
    def __init__(
        self, latent_size: int, num_cells: tuple[int, ...], dropout: float
    ):
        super().__init__()
        self.latent_size = latent_size
        self.num_cells = num_cells
        self.dropout = dropout

        self.net = MLP(
            in_features=latent_size,
            out_features=1,
            num_cells=self.num_cells,
            dropout=self.dropout,
        )

    @override
    def forward(self, mu: Tensor) -> torch.Tensor:
        return self.net(mu)


@final
class Zannone2019PolicyModule(nn.Module):
    def __init__(
        self,
        latent_size: int,
        n_actions: int,
        num_cells: tuple[int, ...],
        dropout: float,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.n_actions = n_actions
        self.num_cells = num_cells
        self.dropout = dropout

        self.net = MLP(
            in_features=latent_size,
            out_features=n_actions,
            num_cells=self.num_cells,
            dropout=self.dropout,
        )

    @override
    def forward(
        self,
        mu: Tensor,
        action_mask: Bool[Tensor, "batch n_actions"],
    ) -> torch.Tensor:
        action_logits = self.net(mu)
        # By setting the logits of invalid actions to -inf, we prevent them from being selected.
        action_logits[~action_mask] = float("-inf")
        return action_logits


@final
class Zannone2019CommonModule(nn.Module):
    def __init__(
        self, pointnet: PointNet, encoder: nn.Module, n_feature_dims: int
    ):
        super().__init__()
        self.pointnet = pointnet
        self.encoder = encoder
        self.n_feature_dims = n_feature_dims

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> torch.Tensor:
        # pointnet is trained on and expects 1D features
        # First flatten feature dims
        flat_masked_features = masked_features.flatten(
            start_dim=-self.n_feature_dims
        )
        flat_feature_mask = feature_mask.flatten(
            start_dim=-self.n_feature_dims
        )
        batch_dims = flat_masked_features.shape[:-1]
        # Then flatten batch dimension
        flat_masked_features = flat_masked_features.flatten(end_dim=-2)
        flat_feature_mask = flat_feature_mask.flatten(end_dim=-2)

        # The pointnet is trained to accept labels appended to the features
        # Since we don't know the label, simple append a vector of zeros
        device = flat_masked_features.device
        flat_batch_size = flat_masked_features.shape[0]
        n_normal_features = flat_masked_features.shape[1]
        n_classes = self.pointnet.n_features - n_normal_features
        augmented_masked_features = torch.cat(
            [
                flat_masked_features,
                torch.zeros((flat_batch_size, n_classes), device=device),
            ],
            dim=-1,
        )
        augmented_feature_mask = torch.cat(
            [
                flat_feature_mask,
                torch.zeros((flat_batch_size, n_classes), device=device),
            ],
            dim=-1,
        )
        pointnet_output = self.pointnet(
            augmented_masked_features, augmented_feature_mask
        )
        encoding = self.encoder(pointnet_output)
        mu = encoding[:, : encoding.shape[1] // 2]

        # Unflatten batch dimensions
        mu = mu.unflatten(0, batch_dims)
        return mu


@final
class Zannone2019Agent(Agent):
    def __init__(
        self,
        cfg: Zannone2019AgentConfig,
        pointnet: PointNet,
        encoder: nn.Module,
        action_spec: TensorSpec,
        latent_size: int,
        action_mask_key: str,
        frames_per_batch: int,
        module_device: torch.device,
        n_feature_dims: int,
    ):
        self.cfg = cfg
        self.pointnet = pointnet
        self.pointnet.requires_grad_(False)
        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.action_spec = action_spec
        self.latent_size = latent_size
        self.action_mask_key = action_mask_key
        self.frames_per_batch = frames_per_batch
        self.module_device = module_device
        self.n_feature_dims = n_feature_dims

        self.common_module = Zannone2019CommonModule(
            pointnet=self.pointnet,
            encoder=self.encoder,
            n_feature_dims=self.n_feature_dims,
        ).to(self.module_device)
        self.common_tdmodule = TensorDictModule(
            module=self.common_module,
            in_keys=["masked_features", "feature_mask"],
            out_keys=["mu"],
        )
        self.policy_head = Zannone2019PolicyModule(
            latent_size=self.latent_size,
            n_actions=self.action_spec.n,  # pyright: ignore[reportAttributeAccessIssue]
            num_cells=tuple(self.cfg.policy_num_cells),
            dropout=self.cfg.policy_dropout,
        ).to(self.module_device)
        self.policy_tdmodule = TensorDictSequential(
            [
                self.common_tdmodule,
                TensorDictModule(
                    self.policy_head,
                    in_keys=["mu", self.action_mask_key],
                    out_keys=["logits"],
                ),
            ]
        )

        self.probabilistic_policy_tdmodule = ProbabilisticActor(
            module=self.policy_tdmodule,
            spec=self.action_spec,
            in_keys=["logits"],
            distribution_class=Categorical,
            return_log_prob=True,
        )

        self.value_head = Zannone2019ValueModule(
            latent_size=self.latent_size,
            num_cells=tuple(self.cfg.value_num_cells),
            dropout=self.cfg.value_dropout,
        ).to(self.module_device)

        self.state_value_tdmodule = TensorDictSequential(
            [
                self.common_tdmodule,
                TensorDictModule(
                    self.value_head,
                    in_keys=["mu"],
                    out_keys=["state_value"],
                ),
            ]
        )

        self.loss_tdmodule = ClipPPOLoss(
            actor_network=self.probabilistic_policy_tdmodule,
            critic_network=self.state_value_tdmodule,
            clip_epsilon=self.cfg.clip_epsilon,
            entropy_bonus=self.cfg.entropy_bonus,
            entropy_coef=self.cfg.entropy_coef,
            critic_coef=self.cfg.critic_coef,
            loss_critic_type=self.cfg.loss_critic_type,
        ).to(self.module_device)
        self.loss_tdmodule.make_value_estimator(
            ValueEstimators.TDLambda,
            gamma=self.cfg.gamma,
            lmbda=self.cfg.lmbda,
        )
        self.loss_keys = ["loss_objective", "loss_critic"] + (
            ["loss_entropy"] if self.cfg.entropy_bonus else []
        )
        self.optimizer = optim.Adam(
            self.loss_tdmodule.parameters(), lr=self.cfg.lr
        )

    @override
    def get_exploitative_policy(self) -> ProbabilisticActor:
        # No distinction between "exploitative" and "exploratory" modules
        # User has to set ExplorationType
        return self.probabilistic_policy_tdmodule

    @override
    def get_exploratory_policy(self) -> ProbabilisticActor:
        # No distinction between "exploitative" and "exploratory" modules
        # User has to set ExplorationType
        return self.probabilistic_policy_tdmodule

    @override
    def get_policy(self) -> ProbabilisticActor:
        return self.probabilistic_policy_tdmodule

    @override
    def process_batch(self, td: TensorDictBase) -> dict[str, Any]:
        # Initialize total loss dictionary
        total_loss_dict = dict.fromkeys(self.loss_keys + ["loss"], 0.0)

        # Perform multiple epochs of training
        for _ in range(self.cfg.num_epochs):
            td_copy = td.clone()
            self.optimizer.zero_grad()
            loss_td: TensorDictBase = self.loss_tdmodule(td_copy)
            loss_tensor: Tensor = sum(
                (loss_td[k] for k in self.loss_keys),
                torch.tensor(0.0, device=td.device),
            )
            loss_tensor.backward()
            nn.utils.clip_grad_norm_(
                self.loss_tdmodule.parameters(),
                max_norm=self.cfg.max_grad_norm,
            )
            self.optimizer.step()

            # Accumulate losses
            for k in self.loss_keys:
                total_loss_dict[k] += loss_td[k].item()
                total_loss_dict["loss"] += loss_td[k].item()

        # Compute average loss
        process_dict = {
            k: v / self.cfg.num_epochs for k, v in total_loss_dict.items()
        }

        return process_dict

    @override
    def get_module_device(self) -> torch.device:
        return torch.device(self.module_device)

    @override
    def set_module_device(self, device: torch.device) -> None:
        self.module_device = device

        self.common_tdmodule = self.common_tdmodule.to(self.module_device)
        self.policy_head = self.policy_head.to(self.module_device)
        self.value_head = self.value_head.to(self.module_device)

    @override
    def get_replay_buffer_device(self) -> None:
        return None

    @override
    def set_replay_buffer_device(self, device: torch.device) -> None:
        msg = "set_replay_buffer_device not yet supported for Zannone2019Agent"
        raise ValueError(msg)

    @override
    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, Any]:
        return {
            "common_module_norm": module_norm(self.common_module),
            "value_head_norm": module_norm(self.value_head),
            "policy_head_norm": module_norm(self.policy_head),
        }
