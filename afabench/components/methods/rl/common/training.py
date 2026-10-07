"""
Shared training loop of the RL methods (JAFA, ODIN, OL).

`RLTrainer` builds the AFA environments from the training contract's inputs
and runs the collector loop; subclasses supply the reward function, the
agent and the resulting AFA method. The script that calls it owns seeding,
the metric logger and saving (`afabench.fit.run`).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any

import torch
import wandb
from rl_helpers import dict_with_prefix
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

from afabench.components.methods.rl.common.afa_env import AFAEnv
from afabench.components.methods.rl.common.agent_interface import Agent
from afabench.components.methods.rl.common.config import (
    AFAMDPConfig,
    AFARLTrainingLoopConfig,
)
from afabench.components.methods.rl.common.custom_types import AFARewardFn
from afabench.components.methods.rl.common.dataset_utils import (
    get_afa_dataset_fn,
)
from afabench.components.methods.rl.common.utils import (
    get_eval_metrics,
)
from afabench.core.types import (
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)
from afabench.core.utils import get_class_frequencies
from afabench.fit.contract import TrainingContract
from afabench.fit.inputs import FitInputs
from afabench.fit.metric_logger import MetricLogger
from afabench.fit.smoke_test import SMOKE_TEST_N_BATCHES

log = logging.getLogger(__name__)


def limit_training_loop(
    cfg: AFARLTrainingLoopConfig, *, smoke_test: bool
) -> AFARLTrainingLoopConfig:
    """Train for only a few batches during a smoke test."""
    if not smoke_test:
        return cfg

    log.info("Smoke test detected.")
    return replace(cfg, n_batches=SMOKE_TEST_N_BATCHES)


def _should_disable_collector_cuda_sync(device: torch.device) -> bool:
    """Disable collector CUDA sync for CPU-only runs."""
    return device.type == "cpu"


class RLTrainer(ABC):
    contract: TrainingContract
    inputs: FitInputs
    mdp_cfg: AFAMDPConfig
    metric_logger: MetricLogger
    device: torch.device
    train_dataset: AFADataset
    val_dataset: AFADataset
    class_weights: torch.Tensor
    unmasker: AFAUnmasker
    initializer: AFAInitializer
    reward_fn: AFARewardFn
    agent: Agent
    unnormalized_selection_costs: torch.Tensor
    normalized_selection_costs: torch.Tensor
    train_env: AFAEnv
    eval_env: AFAEnv

    def __init__(
        self,
        contract: TrainingContract,
        inputs: FitInputs,
        mdp_cfg: AFAMDPConfig,
        metric_logger: MetricLogger,
    ):
        self.contract = contract
        self.inputs = inputs
        self.mdp_cfg = mdp_cfg
        self.metric_logger = metric_logger
        self.device = torch.device(contract.device)

        self._create_datasets()
        self._calculate_class_weights()
        self.unmasker = self.inputs.unmasker()
        self.initializer = self.inputs.initializer()
        self._create_selection_costs()
        self._setup_subclass_specific_state()
        self.reward_fn = self._get_reward_fn()
        self._create_envs()
        self.agent = self._get_agent()

    def _setup_subclass_specific_state(self) -> None:
        return None

    def _create_datasets(self) -> None:
        self.train_dataset = self._with_1d_labels(self.inputs.train_dataset())
        self.val_dataset = self._with_1d_labels(self.inputs.val_dataset())

    @staticmethod
    def _with_1d_labels(dataset: AFADataset) -> AFADataset:
        if len(dataset.label_shape) != 1:
            msg = (
                "Expected a 1D label shape (n_classes), got "
                f"label_shape={dataset.label_shape}"
            )
            raise ValueError(msg)
        return dataset

    def _calculate_class_weights(self) -> None:
        _train_features, train_labels = self.train_dataset.get_all_data()
        train_class_probabilities = get_class_frequencies(train_labels)
        class_weights = 1 / train_class_probabilities
        class_weights = class_weights / class_weights.sum()
        self.class_weights = class_weights.to(self.device)

    @abstractmethod
    def _get_reward_fn(self) -> AFARewardFn: ...

    def _create_selection_costs(self) -> None:
        self.unnormalized_selection_costs = self.unmasker.get_selection_costs(
            feature_costs=self.train_dataset.get_feature_acquisition_costs()
        ).to(self.device)
        self.normalized_selection_costs = (
            self.unnormalized_selection_costs
            / self.unnormalized_selection_costs.sum()
        ) * len(self.unnormalized_selection_costs)

    def _create_envs(self) -> None:
        self.train_env = self._get_env_from_dataset(self.train_dataset)
        self.eval_env = self._get_env_from_dataset(self.val_dataset)

    def _get_env_from_dataset(self, dataset: AFADataset) -> AFAEnv:
        features, labels = dataset.get_all_data()
        dataset_fn = get_afa_dataset_fn(features, labels, device=self.device)
        env = AFAEnv(
            dataset_fn=dataset_fn,
            reward_fn=self.reward_fn,
            device=self.device,
            batch_size=torch.Size((self.mdp_cfg.n_agents,)),
            feature_shape=dataset.feature_shape,
            n_selections=self._n_selections,
            n_classes=self._n_classes,
            hard_budget=self.contract.hard_budget,
            initialize_fn=self.initializer.initialize,
            unmask_fn=self.unmasker.unmask,
            force_hard_budget=self.mdp_cfg.force_hard_budget,
            seed=self.contract.seed,
            selection_costs=self.unnormalized_selection_costs.tolist(),
        )
        return env

    @property
    def _n_selections(self) -> int:
        return len(self.unnormalized_selection_costs)

    @property
    def _n_classes(self) -> int:
        return self.train_dataset.label_shape.numel()

    @abstractmethod
    def _get_agent(self) -> Agent: ...

    def train(self, cfg: AFARLTrainingLoopConfig) -> AFAMethod:
        """Run the training loop and return the trained AFA method on the CPU."""
        collector = SyncDataCollector(
            self.train_env,
            self.agent.get_exploratory_policy(),
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.n_batches * cfg.frames_per_batch,
            device=self.device,
            no_cuda_sync=_should_disable_collector_cuda_sync(self.device),
        )

        for batch_idx, td in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            self._single_collector_step(collector, td, batch_idx, cfg)

        return self._get_afa_method(device=torch.device("cpu"))

    def _single_collector_step(
        self,
        collector: SyncDataCollector,
        td: TensorDictBase,
        batch_idx: int,
        cfg: AFARLTrainingLoopConfig,
    ) -> None:
        # In case we have multiple different devices
        collector.update_policy_weights_()

        train_dict_to_log = self._train_step(td, batch_idx=batch_idx)

        if self._should_evaluate_at_batch(
            batch_idx, cfg.n_batches, cfg.eval_n_times
        ):
            eval_dict_to_log = self._eval_step(
                batch_idx=batch_idx,
                eval_max_steps=cfg.eval_max_steps,
                n_eval_episodes=cfg.n_eval_episodes,
            )
        else:
            eval_dict_to_log = {}

        self.metric_logger.log(train_dict_to_log | eval_dict_to_log)

    def _train_step(
        self, td: TensorDictBase, batch_idx: int
    ) -> dict[str, Any]:
        # Environment specific logging
        train_env_batch_info = self.train_env.get_batch_info(td)

        # Learning happens here
        agent_process_batch_info = self.agent.process_batch(td)

        # Some methods do stuff like joint training, do that here
        post_process_info = self._post_process_batch(td, batch_idx=batch_idx)

        train_dict_to_log = self._get_train_dict_to_log(
            td=td,
            train_env_batch_info=train_env_batch_info,
            agent_process_batch_info=agent_process_batch_info,
            post_process_info=post_process_info,
        )
        return train_dict_to_log

    def _eval_step(
        self,
        batch_idx: int,
        eval_max_steps: int,
        n_eval_episodes: int,
    ) -> dict[str, Any]:
        log.info(f"Running evaluation at batch {batch_idx}")

        # Some methods (like jafa) need to change their action spec to point to the eval environment
        self._pre_eval()

        with (
            torch.no_grad(),
            set_exploration_type(ExplorationType.DETERMINISTIC),
        ):
            td_evals = [
                self.eval_env.rollout(
                    eval_max_steps, self.agent.get_exploitative_policy()
                ).squeeze(0)
                # .cpu()
                for _ in tqdm(range(n_eval_episodes), desc="Evaluating")
            ]

        eval_dict_to_log = self._get_eval_dict_to_log_from_rollouts(td_evals)

        # Some methods might need resetting here
        self._post_eval()

        log.info(f"Evaluation completed at batch {batch_idx}")

        return eval_dict_to_log

    def _get_eval_dict_to_log_from_rollouts(
        self, td_evals: list[TensorDictBase]
    ) -> dict[str, Any]:
        # A cpu copy is passed to logging functions
        td_evals_cpu = [td_eval.cpu() for td_eval in td_evals]

        # Environment specific logging of rollouts
        eval_env_rollout_info = self.eval_env.get_rollout_info(td_evals_cpu)

        # Agent specific logging of rollouts
        agent_rollout_info = self.agent.get_rollout_info(td_evals_cpu)

        # With a predictor we can perform classification at every step of the episode. GPU copy is used here
        metrics_eval = get_eval_metrics(
            eval_tds=td_evals,
            afa_predict_fn=self._get_afa_method(device=self.device).predict,
            feature_shape=self._feature_shape,
        )

        eval_dict_to_log = dict_with_prefix(
            "eval/",
            dict_with_prefix("eval_env_rollout_info.", eval_env_rollout_info)
            | dict_with_prefix("agent_rollout_info.", agent_rollout_info)
            | dict_with_prefix(
                "agent_expensive_info.", self.agent.get_expensive_info()
            )
            | dict_with_prefix("metrics.", metrics_eval),
        )
        return eval_dict_to_log

    @property
    def _feature_shape(self) -> torch.Size:
        return self.train_dataset.feature_shape

    @abstractmethod
    def _get_afa_method(self, device: torch.device) -> AFAMethod: ...

    def _post_process_batch(
        self,
        td: TensorDictBase,  # noqa: ARG002
        batch_idx: int,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {}

    def _get_train_dict_to_log(
        self,
        td: TensorDictBase,
        train_env_batch_info: dict[str, Any],
        agent_process_batch_info: dict[str, Any],
        post_process_info: dict[str, Any],
    ) -> dict[str, Any]:
        train_dict_to_log = dict_with_prefix(
            "train/",
            dict_with_prefix("train_env_batch_info.", train_env_batch_info)
            | dict_with_prefix(
                "agent_process_batch_info.", agent_process_batch_info
            )
            | dict_with_prefix(
                "agent_cheap_info.", self.agent.get_cheap_info()
            )
            | dict_with_prefix("post_process_info.", post_process_info)
            | {"action_distribution": wandb.Histogram(td["action"].cpu())},
        )
        return train_dict_to_log

    def _should_evaluate_at_batch(
        self, batch_idx: int, n_batches: int, eval_n_times: int | None
    ) -> bool:
        if eval_n_times is None or eval_n_times <= 0 or batch_idx == 0:
            return False

        eval_interval = n_batches // eval_n_times
        return eval_interval > 0 and batch_idx % eval_interval == 0

    def _pre_eval(self) -> None:
        return None

    def _post_eval(self) -> None:
        return None

    @property
    def _n_feature_dims(self) -> int:
        return len(self.train_dataset.feature_shape)
