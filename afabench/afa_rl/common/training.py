from __future__ import annotations

import gc
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import wandb
from rl_helpers import dict_with_prefix
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from tqdm import tqdm

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.dataset_utils import get_afa_dataset_fn
from afabench.afa_rl.common.utils import (
    get_eval_metrics,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.custom_types import AFAInitializer
from afabench.common.initializers.utils import get_afa_initializer_from_config
from afabench.common.unmaskers import AFAContextUnmasker, CubeNMARUnmasker
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import get_class_frequencies, initialize_wandb_run

if TYPE_CHECKING:
    from collections.abc import Callable

    from tensordict import TensorDictBase
    from wandb.sdk.wandb_run import Run

    from afabench.afa_rl.common.agent_interface import Agent
    from afabench.afa_rl.common.custom_types import AFARewardFn
    from afabench.common.config_classes import (
        AFAMDPConfig,
        AFARLTrainingLoopConfig,
        InitializerConfig,
        UnmaskerConfig,
    )
    from afabench.common.custom_types import (
        AFADataset,
        AFAInitializeFn,
        AFAMethod,
        AFAUnmasker,
        FeatureMask,
        SelectionMask,
    )

log = logging.getLogger(__name__)

type EnvMaskMode = Literal["default", "train_restricted", "eval_cold"]


def _initializer_has_training_support_restriction(
    initializer: AFAInitializer,
) -> bool:
    return (
        type(initializer).get_training_forbidden_mask
        is not AFAInitializer.get_training_forbidden_mask
    )


def _adapt_forbidden_mask_to_selection_space(
    forbidden_mask: torch.Tensor,
    *,
    n_selection_choices: int,
    feature_shape: torch.Size,
    unmasker: AFAUnmasker,
) -> torch.Tensor:
    """Convert feature-space forbidden masks to selection-space masks."""
    if forbidden_mask.shape[-1] == n_selection_choices:
        return forbidden_mask

    n_features = math.prod(feature_shape)
    if forbidden_mask.shape[-1] != n_features:
        msg = (
            "Initializer forbidden mask has incompatible shape. "
            f"Expected trailing dim {n_selection_choices} (selection space) or "
            f"{n_features} (feature space), got {forbidden_mask.shape[-1]}."
        )
        raise ValueError(msg)

    def collapse_grouped_context_mask(
        n_contexts: int,
        *,
        unmasker_name: str,
    ) -> torch.Tensor:
        expected_n_selections = 1 + (n_features - n_contexts)
        if n_selection_choices != expected_n_selections:
            msg = (
                f"Unexpected selection-space size for {unmasker_name}. "
                f"Expected {expected_n_selections}, got {n_selection_choices}."
            )
            raise ValueError(msg)

        flat_forbidden = forbidden_mask.reshape(-1, n_features)
        sel_forbidden = torch.zeros(
            (flat_forbidden.shape[0], n_selection_choices),
            dtype=torch.bool,
            device=forbidden_mask.device,
        )
        sel_forbidden[:, 0] = flat_forbidden[:, :n_contexts].any(dim=1)
        sel_forbidden[:, 1:] = flat_forbidden[:, n_contexts:]
        batch_shape = forbidden_mask.shape[:-1]
        return sel_forbidden.reshape(*batch_shape, n_selection_choices)

    if isinstance(unmasker, AFAContextUnmasker):
        return collapse_grouped_context_mask(
            unmasker.n_contexts,
            unmasker_name="AFAContextUnmasker",
        )

    if isinstance(unmasker, CubeNMARUnmasker):
        return collapse_grouped_context_mask(
            unmasker.n_contexts,
            unmasker_name="CubeNMARUnmasker",
        )

    if forbidden_mask.any():
        msg = (
            "Cannot convert feature-level forbidden mask to selection space for "
            f"unmasker {type(unmasker).__name__}."
        )
        raise ValueError(msg)

    return torch.zeros(
        (*forbidden_mask.shape[:-1], n_selection_choices),
        dtype=torch.bool,
        device=forbidden_mask.device,
    )


def _build_env_mask_fns(
    initializer: AFAInitializer,
    unmasker: AFAUnmasker,
    *,
    n_selection_choices: int,
    mode: EnvMaskMode,
) -> tuple[
    AFAInitializeFn,
    Callable[[FeatureMask, torch.Size], SelectionMask] | None,
]:
    """
    Build environment mask functions for RL under default or train-missing semantics.

    `train_restricted` matches the thesis objective: sample missingness once,
    start from a cold state, and block training-forbidden actions.
    `eval_cold` restores the standard cold-start evaluation protocol.
    """
    if mode == "eval_cold":

        def cold_initialize(
            features: torch.Tensor,
            label: torch.Tensor,
            feature_shape: torch.Size | None = None,
        ) -> torch.Tensor:
            del label
            assert feature_shape is not None, (
                "feature_shape must be provided for cold initialization"
            )
            batch_shape = features.shape[: -len(feature_shape)]
            return torch.zeros(
                batch_shape + feature_shape,
                dtype=torch.bool,
                device=features.device,
            )

        return cold_initialize, None

    cached_observed_mask: torch.Tensor | None = None
    maybe_forbidden_selection_mask_fn = getattr(
        initializer, "get_forbidden_selection_mask", None
    )
    typed_forbidden_selection_mask_fn = (
        cast(
            "Callable[[FeatureMask, torch.Size], SelectionMask]",
            maybe_forbidden_selection_mask_fn,
        )
        if callable(maybe_forbidden_selection_mask_fn)
        else None
    )

    def initialize_fn(
        features: torch.Tensor,
        label: torch.Tensor,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        nonlocal cached_observed_mask
        observed_mask = initializer.initialize(
            features=features,
            label=label,
            feature_shape=feature_shape,
        ).bool()
        cached_observed_mask = observed_mask
        if mode == "train_restricted":
            return torch.zeros_like(observed_mask, dtype=torch.bool)
        return observed_mask

    forbidden_selection_mask_fn: (
        Callable[[FeatureMask, torch.Size], SelectionMask] | None
    ) = None
    if (
        mode == "train_restricted"
        or typed_forbidden_selection_mask_fn is not None
    ):

        def build_forbidden_selection_mask(
            observed_mask: torch.Tensor,
            feature_shape: torch.Size,
        ) -> torch.Tensor:
            source_observed_mask = (
                observed_mask
                if cached_observed_mask is None
                else cached_observed_mask
            )
            batch_shape = source_observed_mask.shape[: -len(feature_shape)]
            combined = torch.zeros(
                batch_shape + torch.Size((n_selection_choices,)),
                dtype=torch.bool,
                device=source_observed_mask.device,
            )

            if mode == "train_restricted":
                training_forbidden = initializer.get_training_forbidden_mask(
                    source_observed_mask
                ).bool()
                combined |= _adapt_forbidden_mask_to_selection_space(
                    training_forbidden,
                    n_selection_choices=n_selection_choices,
                    feature_shape=feature_shape,
                    unmasker=unmasker,
                )

            if typed_forbidden_selection_mask_fn is not None:
                raw_forbidden = typed_forbidden_selection_mask_fn(
                    source_observed_mask,
                    feature_shape,
                ).bool()
                combined |= _adapt_forbidden_mask_to_selection_space(
                    raw_forbidden,
                    n_selection_choices=n_selection_choices,
                    feature_shape=feature_shape,
                    unmasker=unmasker,
                )

            return combined

        forbidden_selection_mask_fn = build_forbidden_selection_mask

    return initialize_fn, forbidden_selection_mask_fn


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
        self._create_selection_costs()
        self._setup_subclass_specific_state()
        self.reward_fn = self._get_reward_fn()
        self._create_envs()
        self.agent = self._get_agent()

    def _create_log_fn(self) -> None:
        if self.run is not None:
            self.log_fn = self.run.log
        else:
            self.log_fn: Callable[[dict[str, Any]], None] = lambda _d: None

    def _setup_subclass_specific_state(self) -> None:
        return None

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
        class_weights = class_weights / class_weights.sum()
        self.class_weights = class_weights.to(self.device)

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
        has_train_restriction = _initializer_has_training_support_restriction(
            self.initializer
        )
        if has_train_restriction:
            log.info(
                "Initializer %s uses restricted-support RL training. "
                "Train env starts cold and blocks training-forbidden actions; "
                "eval env uses standard cold-start semantics.",
                type(self.initializer).__name__,
            )

        self.train_env = self._get_env_from_dataset(
            self.train_dataset,
            mode="train_restricted" if has_train_restriction else "default",
        )
        self.eval_env = self._get_env_from_dataset(
            self.val_dataset,
            mode="eval_cold" if has_train_restriction else "default",
        )

    def _get_env_from_dataset(
        self,
        dataset: AFADataset,
        *,
        mode: EnvMaskMode,
    ) -> AFAEnv:
        features, labels = dataset.get_all_data()
        dataset_fn = get_afa_dataset_fn(features, labels, device=self.device)
        initialize_fn, forbidden_selection_mask_fn = _build_env_mask_fns(
            self.initializer,
            self.unmasker,
            n_selection_choices=self._n_selections,
            mode=mode,
        )
        env = AFAEnv(
            dataset_fn=dataset_fn,
            reward_fn=self.reward_fn,
            device=self.device,
            batch_size=torch.Size((self.n_agents,)),
            feature_shape=dataset.feature_shape,
            n_selections=self._n_selections,
            n_classes=self._n_classes,
            hard_budget=self.mdp_cfg.hard_budget,
            initialize_fn=initialize_fn,
            unmask_fn=self.unmasker.unmask,
            forbidden_selection_mask_fn=forbidden_selection_mask_fn,
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
