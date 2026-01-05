import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from rl_helpers import dict_with_prefix
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.common.custom_types import AFARewardFn
from afabench.afa_rl.common.dataset_utils import get_afa_dataset_fn
from afabench.afa_rl.common.utils import (
    get_eval_metrics,
)

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.common.bundle import load_bundle
from afabench.common.config_classes import (
    AFARLTrainingLoopConfig,
    InitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFADataset,
    AFAInitializeFn,
    AFAInitializer,
    AFAPredictFn,
    AFAUnmasker,
    AFAUnmaskFn,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import get_class_frequencies

log = logging.getLogger(__name__)


def should_evaluate_at_batch(
    batch_idx: int, n_batches: int, eval_n_times: int | None
) -> bool:
    """
    Determine if evaluation should be performed at the current batch.

    Args:
        batch_idx: Current batch index (0-based)
        n_batches: Total number of batches in training
        eval_n_times: Total number of evaluations desired across training

    Returns:
        True if evaluation should be performed at this batch, False otherwise
    """
    if eval_n_times is None or eval_n_times <= 0 or batch_idx == 0:
        return False

    eval_interval = n_batches // eval_n_times
    return eval_interval > 0 and batch_idx % eval_interval == 0


# class PretrainedModel(Protocol):
#     pass


def afa_rl_training_loop(
    cfg: AFARLTrainingLoopConfig,
    train_env: AFAEnv,
    eval_env: AFAEnv,
    agent: Agent,
    afa_predict_fn: AFAPredictFn,
    device: torch.device,
    *,
    feature_shape: torch.Size | None = None,
    log_fn: Callable[[dict[str, Any]], None] | None = None,
    post_process_batch_callback: Callable[[TensorDict, int], dict[str, Any]]
    | None = None,
    pre_eval_callback: Callable[[], None] | None = None,
    post_eval_callback: Callable[[], None] | None = None,
) -> None:
    """Train an RL agent training in an AFA MDP."""
    if log_fn is None:
        log_fn = lambda _dict: None  # noqa: E731
    if pre_eval_callback is None:
        pre_eval_callback = lambda: None  # noqa: E731
    if post_eval_callback is None:
        post_eval_callback = lambda: None  # noqa: E731

    log.info("Creating data collector")
    collector = SyncDataCollector(
        train_env,
        agent.get_exploratory_policy(),
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.n_batches * cfg.frames_per_batch,
        device=device,
    )
    log.info("Data collector created")

    # Training loop
    log.info(f"Starting training loop for {cfg.n_batches} batches")
    for batch_idx, td in tqdm(
        enumerate(collector), total=cfg.n_batches, desc="Training agent..."
    ):
        evaluate_this_batch = should_evaluate_at_batch(
            batch_idx, cfg.n_batches, cfg.eval_n_times
        )

        # In case we have multiple different devices
        collector.update_policy_weights_()

        # Environment specific logging
        train_env_batch_info = train_env.get_batch_info(td)

        # Learning happens here
        agent_process_batch_info = agent.process_batch(td)

        # Some methods do stuff like joint training, do that here
        if post_process_batch_callback is not None:
            post_process_info = post_process_batch_callback(td, batch_idx)
        else:
            post_process_info = {}

        dict_to_log = dict_with_prefix(
            "train/",
            dict_with_prefix("train_env_batch_info.", train_env_batch_info)
            | dict_with_prefix(
                "agent_process_batch_info.", agent_process_batch_info
            )
            | dict_with_prefix("agent_cheap_info.", agent.get_cheap_info())
            | dict_with_prefix("post_process_info.", post_process_info),
        )

        if evaluate_this_batch:
            log.info(f"Running evaluation at batch {batch_idx}")

            # Some methods (like shim2018) need to change their action spec to point to the eval environment
            pre_eval_callback()

            with (
                torch.no_grad(),
                set_exploration_type(ExplorationType.DETERMINISTIC),
            ):
                td_evals = [
                    eval_env.rollout(
                        cfg.eval_max_steps, agent.get_exploitative_policy()
                    ).squeeze(0)
                    for _ in tqdm(
                        range(cfg.n_eval_episodes), desc="Evaluating"
                    )
                ]

            # Environment specific logging of rollouts
            eval_env_rollout_info = eval_env.get_rollout_info(td_evals)

            # Agent specific logging of rollouts
            agent_rollout_info = agent.get_rollout_info(td_evals)

            # With a predictor we can perform classification at every step of the episode.
            metrics_eval = get_eval_metrics(
                td_evals, afa_predict_fn, feature_shape=feature_shape
            )

            dict_to_log = dict_to_log | dict_with_prefix(
                "eval/",
                dict_with_prefix(
                    "eval_env_rollout_info.", eval_env_rollout_info
                )
                | dict_with_prefix("agent_rollout_info.", agent_rollout_info)
                | dict_with_prefix(
                    "agent_expensive_info.", agent.get_expensive_info()
                )
                | dict_with_prefix("metrics.", metrics_eval),
            )

            # Some methods might need resetting here
            post_eval_callback()

            log.info(f"Evaluation completed at batch {batch_idx}")

        # Log everything
        log_fn(dict_to_log)


def afa_rl_training_prep(
    train_dataset_bundle_path: Path,
    val_dataset_bundle_path: Path,
    initializer_cfg: InitializerConfig,
    unmasker_cfg: UnmaskerConfig,
) -> tuple[AFADataset, AFADataset, AFAInitializer, AFAUnmasker, torch.Tensor]:
    train_dataset, _train_dataset_manifest = load_bundle(
        train_dataset_bundle_path,
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    val_dataset, _val_dataset_manifest = load_bundle(
        val_dataset_bundle_path,
    )
    val_dataset = cast("AFADataset", cast("object", val_dataset))

    initializer = get_afa_initializer_from_config(initializer_cfg)

    unmasker = get_afa_unmasker_from_config(unmasker_cfg)

    # Also calculate class weights
    _train_features, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)
    class_weights = 1 / train_class_probabilities
    class_weights = class_weights / class_weights.sum()

    return train_dataset, val_dataset, initializer, unmasker, class_weights


def create_afa_envs(
    train_dataset: AFADataset,
    val_dataset: AFADataset,
    reward_fn: AFARewardFn,
    n_agents: int,
    n_selections: int,
    hard_budget: int | None,
    initialize_fn: AFAInitializeFn,
    unmask_fn: AFAUnmaskFn,
    force_hard_budget: bool,
    device: torch.device,
    seed: int | None,
) -> tuple[AFAEnv, AFAEnv]:
    train_features, train_labels = train_dataset.get_all_data()
    val_features, val_labels = val_dataset.get_all_data()

    train_dataset_fn = get_afa_dataset_fn(
        train_features, train_labels, device=device
    )
    val_dataset_fn = get_afa_dataset_fn(
        val_features, val_labels, device=device
    )
    assert len(train_dataset.label_shape) == 1, (
        "Expected 1D label shape (n_classes). Instead got {train_dataset.label_shape}"
    )
    n_classes = train_dataset.label_shape[-1]
    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((n_agents,)),
        feature_shape=train_dataset.feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=hard_budget,
        initialize_fn=initialize_fn,
        unmask_fn=unmask_fn,
        force_hard_budget=force_hard_budget,
        seed=seed,
    )
    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((n_agents,)),
        feature_shape=val_dataset.feature_shape,
        n_selections=n_selections,
        n_classes=n_classes,
        hard_budget=hard_budget,
        initialize_fn=initialize_fn,
        unmask_fn=unmask_fn,
        force_hard_budget=force_hard_budget,
        seed=seed,
    )
    return train_env, eval_env
