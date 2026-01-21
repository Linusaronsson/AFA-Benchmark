import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import wandb
from rl_helpers import dict_with_prefix
from tensordict import TensorDictBase
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.agent_interface import Agent
from afabench.afa_rl.common.custom_types import AFARewardFn
from afabench.afa_rl.common.dataset_utils import get_afa_dataset_fn
from afabench.afa_rl.common.utils import (
    get_eval_metrics,
)

# from afabench.afa_rl.reward_functions import get_range_based_reward_fn
# from afabench.afa_rl.shim2018.reward import get_shim2018_reward_fn
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import (
    AFAMDPConfig,
    AFARLTrainingLoopConfig,
    InitializerConfig,
    UnmaskerConfig,
)
from afabench.common.custom_types import (
    AFADataset,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
)
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import get_class_frequencies, initialize_wandb_run

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)


class RLTrainer(ABC):
    train_dataset_bundle_path: Path
    val_dataset_bundle_path: Path
    initializer_cfg: InitializerConfig
    unmasker_cfg: UnmaskerConfig
    mdp_cfg: AFAMDPConfig
    n_agents: int
    seed: int | None
    device: torch.device
    cfg: dict[str, Any]
    use_wandb: bool
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
    run: Run | None

    def __init__(
        self,
        train_dataset_bundle_path: Path,
        val_dataset_bundle_path: Path,
        initializer_cfg: InitializerConfig,
        unmasker_cfg: UnmaskerConfig,
        mdp_cfg: AFAMDPConfig,
        n_agents: int,
        seed: int | None,
        device: torch.device,
        cfg: dict[str, Any],  # used only for logging
        *,
        use_wandb: bool = False,
    ):
        self.train_dataset_bundle_path = train_dataset_bundle_path
        self.val_dataset_bundle_path = val_dataset_bundle_path
        self.initializer_cfg = initializer_cfg
        self.unmasker_cfg = unmasker_cfg
        self.mdp_cfg = mdp_cfg
        self.n_agents = n_agents
        self.seed = seed
        self.device = device
        self.cfg = cfg
        self.use_wandb = use_wandb

        if self.use_wandb:
            self.run = initialize_wandb_run(
                cfg=self.cfg, job_type="training", tags=self._get_tags()
            )
        else:
            self.run = None
        self._create_log_fn()
        self._create_datasets()
        self._calculate_class_weights()
        self._create_unmasker()
        self._create_initializer()
        self.reward_fn = self._get_reward_fn()
        self._create_selection_costs()
        self._create_envs()
        self.agent = self._get_agent()

    def _create_log_fn(self) -> None:
        if self.run is not None:
            self.log_fn = self.run.log
        else:
            self.log_fn: Callable[[dict[str, Any]], None] = lambda _d: None

    @abstractmethod
    def _get_tags(self) -> list[str]: ...

    def _create_datasets(self) -> None:
        self.train_dataset = self._load_dataset_from_bundle(
            self.train_dataset_bundle_path
        )
        self.val_dataset = self._load_dataset_from_bundle(
            self.val_dataset_bundle_path
        )

    @staticmethod
    def _load_dataset_from_bundle(dataset_bundle_path: Path) -> AFADataset:
        dataset, _dataset_manifest = load_bundle(
            dataset_bundle_path,
        )
        dataset = cast("AFADataset", cast("object", dataset))
        assert len(dataset.label_shape) == 1, (
            "Expected 1D label shape (n_classes). Instead got {train_dataset.label_shape}"
        )
        return dataset

    def _calculate_class_weights(self) -> None:
        _train_features, train_labels = self.train_dataset.get_all_data()
        train_class_probabilities = get_class_frequencies(train_labels)
        class_weights = 1 / train_class_probabilities
        self.class_weights = class_weights / class_weights.sum()

    def _create_unmasker(self) -> None:
        self.unmasker = get_afa_unmasker_from_config(self.unmasker_cfg)

    def _create_initializer(self) -> None:
        self.initializer = get_afa_initializer_from_config(
            self.initializer_cfg
        )

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
            batch_size=torch.Size((self.n_agents,)),
            feature_shape=dataset.feature_shape,
            n_selections=self._n_selections,
            n_classes=self._n_classes,
            hard_budget=self.mdp_cfg.hard_budget,
            initialize_fn=self.initializer.initialize,
            unmask_fn=self.unmasker.unmask,
            force_hard_budget=self.mdp_cfg.force_hard_budget,
            seed=self.seed,
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

    def train(self, cfg: AFARLTrainingLoopConfig) -> None:
        collector = SyncDataCollector(
            self.train_env,
            self.agent.get_exploratory_policy(),
            frames_per_batch=cfg.frames_per_batch,
            total_frames=cfg.n_batches * cfg.frames_per_batch,
            device=self.device,
        )

        for batch_idx, td in tqdm(
            enumerate(collector), total=cfg.n_batches, desc="Training agent..."
        ):
            self._single_collector_step(collector, td, batch_idx, cfg)

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

        self.log_fn(train_dict_to_log | eval_dict_to_log)

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

        # Some methods (like shim2018) need to change their action spec to point to the eval environment
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

    def save(self, save_path: Path) -> None:
        save_bundle(
            obj=self._get_afa_method(device=torch.device("cpu")),
            path=Path(save_path),
            metadata={"config": self.cfg},
        )

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
