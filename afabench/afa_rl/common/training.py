import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from rl_helpers import dict_with_prefix
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, check_env_specs, set_exploration_type
from tqdm import tqdm

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.agent_interface import Agent
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
    AFAInitializer,
    AFAPredictFn,
    AFAUnmasker,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import set_seed

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


class PretrainedModel(Protocol):
    pass


def afa_rl_training_loop(
    cfg: AFARLTrainingLoopConfig,
    train_env: AFAEnv,
    eval_env: AFAEnv,
    agent: Agent,
    post_process_batch_callback: Callable[
        [TensorDict, int], None
    ],  # called with the the train tensordict and the current batch idx. If your method has special logic like joint training, do that here.
    train_log_fn: Callable[
        [TensorDict, dict[str, Any]], None
    ],  # called with the env tensordict and process_batch dict
    afa_predict_fn: AFAPredictFn,  # what to make class predictions with to evaluate method's performance
    device: torch.device,
    *,
    log_fn: Callable[[dict[str, Any]], None]
    | None = None,  # some function that we use for logging, typically wandb.log
    pre_eval_callback: Callable[[AFAEnv, AFAEnv], None]
    | None = None,  # called with (train_env, eval_env) before evaluation
    post_eval_callback: Callable[[AFAEnv, AFAEnv, list[TensorDict]], None]
    | None = None,  # called with (train_env, eval_env, td_evals) after evaluation
) -> None:
    """Train an RL agent training in an AFA MDP."""
    if log_fn is None:
        log_fn = lambda _dict: None  # noqa: E731
    if pre_eval_callback is None:
        pre_eval_callback = lambda _train_env, _eval_env: None  # noqa: E731
    if post_eval_callback is None:
        post_eval_callback = lambda _train_env, _eval_env, _td_evals: None  # noqa: E731

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
        # In case we have multiple different devices
        collector.update_policy_weights_()

        # Learning happens here
        process_batch_td = agent.process_batch(td)

        # Some methods do stuff like joint training, do that here
        post_process_batch_callback(td, batch_idx)

        train_log_fn(td, process_batch_td)

        if should_evaluate_at_batch(
            batch_idx, cfg.n_batches, cfg.eval_n_times
        ):
            log.info(f"Running evaluation at batch {batch_idx}")

            # Some methods (like shim2018) need a reference to the evaluation environment
            pre_eval_callback(train_env, eval_env)

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

            metrics_eval = get_eval_metrics(td_evals, afa_predict_fn)

            log_fn(
                dict_with_prefix(
                    "eval/", dict_with_prefix("metrics.", metrics_eval)
                )
            )

            post_eval_callback(train_env, eval_env, td_evals)

            log.info(f"Evaluation completed at batch {batch_idx}")


def afa_rl_training_prep(
    train_dataset_bundle_path: Path,
    val_dataset_bundle_path: Path,
    initializer_cfg: InitializerConfig,
    unmasker_cfg: UnmaskerConfig,
    *,
    seed: int | None = None,
) -> tuple[AFADataset, AFADataset, AFAInitializer, AFAUnmasker]:
    set_seed(seed)
    torch.set_float32_matmul_precision("medium")

    log.info("Loading datasets...")
    train_dataset, _train_dataset_manifest = load_bundle(
        train_dataset_bundle_path,
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    val_dataset, _val_dataset_manifest = load_bundle(
        val_dataset_bundle_path,
    )
    val_dataset = cast("AFADataset", cast("object", val_dataset))
    log.info("Datasets loaded.")

    # Create initializer
    log.info("Creating initializer...")
    initializer = get_afa_initializer_from_config(initializer_cfg)
    log.info("Initializer created.")

    # Create unmasker
    log.info("Creating unmasker...")
    unmasker = get_afa_unmasker_from_config(unmasker_cfg)
    log.info("Unmasker created.")

    # MDP expects special dataset functions
    log.info("Creating dataset functions for environments...")
    train_features, train_labels = train_dataset.get_all_data()
    n_classes = train_labels.shape[-1]
    train_dataset_fn = get_afa_dataset_fn(
        train_features, train_labels, device=device
    )
    val_features, val_labels = val_dataset.get_all_data()
    val_dataset_fn = get_afa_dataset_fn(
        val_features, val_labels, device=device
    )
    log.info("Dataset functions created.")

    log.info("Creating training environment")
    train_env = AFAEnv(
        dataset_fn=train_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_shape=train_dataset.feature_shape,
        n_selections=unmasker.get_n_selections(
            feature_shape=train_dataset.feature_shape
        ),
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
        initialize_fn=initializer.initialize,
        unmask_fn=unmasker.unmask,
        force_hard_budget=cfg.force_hard_budget,
        seed=cfg.env_seed,
    )
    check_env_specs(train_env)
    log.info("Training environment created and validated")

    log.info("Creating evaluation environment")
    eval_env = AFAEnv(
        dataset_fn=val_dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((cfg.n_agents,)),
        feature_shape=val_dataset.feature_shape,
        n_selections=unmasker.get_n_selections(
            feature_shape=val_dataset.feature_shape
        ),
        n_classes=n_classes,
        hard_budget=cfg.hard_budget,
        initialize_fn=initializer.initialize,
        unmask_fn=unmasker.unmask,
        force_hard_budget=cfg.force_hard_budget,
        seed=cfg.env_seed,
    )
    log.info("Evaluation environment created")

    return train_dataset, val_dataset, initializer, unmasker
