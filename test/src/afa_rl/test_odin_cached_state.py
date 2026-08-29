import torch
from torch import nn
from torchrl.data import Categorical

from afabench.components.methods.rl.odin.agents import ODINAgent
from afabench.components.methods.rl.odin.config import ODINAgentConfig
from afabench.components.methods.rl.odin.models import PointNet, PointNetType


def test_odin_ppo_reuses_collected_state_encoding() -> None:
    pointnet = PointNet(
        identity_size=2,
        n_features=4,
        feature_map_encoder=nn.Linear(3, 3),
        pointnet_type=PointNetType.POINTNET,
        max_embedding_norm=1.0,
    )
    agent = ODINAgent(
        cfg=ODINAgentConfig(
            gamma=1.0,
            lmbda=0.75,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coef=0.01,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
            num_epochs=1,
            lr=1e-3,
            max_grad_norm=1.0,
            value_num_cells=[],
            value_dropout=0.0,
            policy_num_cells=[],
            policy_dropout=0.0,
        ),
        pointnet=pointnet,
        encoder=nn.Linear(3, 4),
        action_spec=Categorical(3),
        latent_size=2,
        action_mask_key="allowed_action_mask",
        frames_per_batch=8,
        module_device=torch.device("cpu"),
        n_feature_dims=1,
    )

    assert any(
        module is agent.common_module
        for module in agent.probabilistic_policy_tdmodule.modules()
    )
    assert not any(
        module is agent.common_module
        for module in agent.loss_tdmodule.modules()
    )
    assert "mu" in agent.loss_tdmodule.in_keys
    assert ("next", "mu") in agent.loss_tdmodule.in_keys
