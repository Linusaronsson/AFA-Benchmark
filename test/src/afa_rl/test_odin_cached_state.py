import pytest
import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictSequential
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from afabench.components.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.components.methods.rl.common.afa_env import AFAEnv
from afabench.components.methods.rl.common.dataset_utils import (
    get_afa_dataset_fn,
)
from afabench.components.methods.rl.common.reward_functions import (
    get_fixed_reward_reward_fn,
)
from afabench.components.methods.rl.odin.agents import ODINAgent
from afabench.components.methods.rl.odin.config import ODINAgentConfig
from afabench.components.methods.rl.odin.models import PointNet, PointNetType
from afabench.components.unmaskers.direct_unmasker import DirectUnmasker

N_FEATURES = 3
N_CLASSES = 2
LATENT_SIZE = 2
FEATURE_MAP_SIZE = 4


def _make_env() -> AFAEnv:
    features = torch.randn(6, N_FEATURES)
    labels = nn.functional.one_hot(torch.tensor([0, 1, 0, 1, 1, 0]), N_CLASSES)
    return AFAEnv(
        dataset_fn=get_afa_dataset_fn(features, labels),
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=1.0, reward_otherwise=-0.1
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((N_FEATURES,)),
        n_selections=N_FEATURES,
        n_classes=N_CLASSES,
        hard_budget=None,
        initialize_fn=FixedRandomInitializer(
            num_initial_features=0
        ).initialize,
        unmask_fn=DirectUnmasker().unmask,
        seed=0,
    )


def _make_agent(env: AFAEnv) -> ODINAgent:
    identity_size = 2
    pointnet = PointNet(
        identity_size=identity_size,
        n_features=N_FEATURES + N_CLASSES,
        feature_map_encoder=nn.Linear(identity_size + 1, FEATURE_MAP_SIZE),
        pointnet_type=PointNetType.POINTNET,
        max_embedding_norm=1.0,
    )
    agent = ODINAgent(
        cfg=ODINAgentConfig(
            gamma=0.9,
            lmbda=0.75,
            clip_epsilon=0.2,
            entropy_bonus=True,
            entropy_coef=0.01,
            critic_coef=1.0,
            loss_critic_type="smooth_l1",
            num_epochs=1,
            lr=1e-3,
            max_grad_norm=1.0,
            value_num_cells=[8],
            value_dropout=0.0,
            policy_num_cells=[8],
            policy_dropout=0.0,
        ),
        pointnet=pointnet,
        encoder=nn.Linear(FEATURE_MAP_SIZE, 2 * LATENT_SIZE),
        action_spec=env.action_spec,
        latent_size=LATENT_SIZE,
        action_mask_key="allowed_action_mask",
        frames_per_batch=8,
        module_device=torch.device("cpu"),
        n_feature_dims=1,
    )
    # The pretrained modules are loaded in eval mode by the training script
    agent.common_module.eval()
    return agent


def _recompute_losses(
    agent: ODINAgent, td: TensorDictBase
) -> dict[str, float]:
    """Compute PPO losses with the actor and critic each re-encoding the state."""
    critic = TensorDictSequential(
        [agent.common_tdmodule, agent.value_head_tdmodule]
    )
    loss_module = ClipPPOLoss(
        actor_network=agent.probabilistic_policy_tdmodule,
        critic_network=critic,
        clip_epsilon=agent.cfg.clip_epsilon,
        entropy_bonus=agent.cfg.entropy_bonus,
        entropy_coef=agent.cfg.entropy_coef,
        critic_coef=agent.cfg.critic_coef,
        loss_critic_type=agent.cfg.loss_critic_type,
    )
    loss_module.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=agent.cfg.gamma,
        lmbda=agent.cfg.lmbda,
    )
    recompute_td = td.exclude("mu", ("next", "mu"))
    with torch.no_grad():
        loss_td = loss_module(recompute_td)
    return {k: loss_td[k].item() for k in agent.loss_keys}


def test_odin_process_batch_losses_match_recomputed_state_encodings() -> None:
    torch.manual_seed(0)
    env = _make_env()
    agent = _make_agent(env)
    collector = SyncDataCollector(
        env,
        agent.get_exploratory_policy(),
        frames_per_batch=8,
        total_frames=8,
        device=torch.device("cpu"),
    )
    td = next(iter(collector))
    collector.shutdown()

    expected_losses = _recompute_losses(agent, td.clone())
    process_dict = agent.process_batch(td.clone())

    for key, expected_loss in expected_losses.items():
        assert process_dict[key] == pytest.approx(expected_loss, abs=1e-6)
    assert process_dict["loss"] == pytest.approx(
        sum(expected_losses.values()), abs=1e-6
    )
