from __future__ import annotations

from typing import TYPE_CHECKING, Any, final, override

import torch
import wandb
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from afabench.afa_rl.common.custom_types import (
        AFADatasetFn,
        AFARewardFn,
    )
    from afabench.common.custom_types import (
        AFAInitializeFn,
        AFAUnmaskFn,
        FeatureMask,
        Features,
        Label,
        SelectionMask,
    )


@final
class AFAEnv(EnvBase):
    """
    A dynamic-length MDP for active feature acquisition (AFA).

    The episode length is at most `hard_budget`, and the agent can choose to stop earlier.
    """

    @property
    @override
    def batch_locked(self) -> bool:
        return False

    @batch_locked.setter
    def batch_locked(self, value: bool) -> None:
        # AFAEnv doesn't support batch locking, so we ignore the setter
        pass

    def __init__(
        self,
        dataset_fn: AFADatasetFn,  # a function that returns data in batches when called
        reward_fn: AFARewardFn,
        device: torch.device | None,
        batch_size: torch.Size,
        feature_shape: torch.Size,
        n_selections: int,  # action dim = n_selections + 1 since we have a stop action as well
        n_classes: int,
        hard_budget: float
        | None,  # accumulated selection cost allowed before the episode ends. If None, no limit.
        initialize_fn: AFAInitializeFn,
        unmask_fn: AFAUnmaskFn,
        forbidden_selection_mask_fn: (
            Callable[[FeatureMask, torch.Size], SelectionMask] | None
        ) = None,
        *,
        force_hard_budget: bool = False,  # if True and hard_budget is set, never allow the stop action
        seed: int | None = None,
        selection_costs: Sequence[float]
        | None = None,  # How much each sequence costs. If None, assume unit cost (1).
    ):
        # Do not allow empty batch sizes
        assert batch_size != torch.Size(()), "Batch size must be non-empty"
        assert len(batch_size) == 1, "Batch size must be 1D"
        super().__init__(device=device, batch_size=batch_size)

        self.dataset_fn = dataset_fn
        self.reward_fn = reward_fn
        self.feature_shape = feature_shape
        self.n_selections = n_selections
        self.n_classes = n_classes
        self.has_hard_budget = hard_budget is not None
        if hard_budget is None:
            # If hard budget is not set, always allow agent to stop
            self.hard_budget = self.n_selections
            self.allow_stop_action = True
        else:
            # If hard budget is set, stop action is only allowed if force_hard_budget is false
            self.hard_budget = hard_budget
            self.allow_stop_action = not force_hard_budget
        self.force_hard_budget = force_hard_budget
        self.initialize_fn = initialize_fn
        self.unmask_fn = unmask_fn
        self.forbidden_selection_mask_fn = forbidden_selection_mask_fn
        self.seed = seed
        if selection_costs is None:
            self.selection_costs = torch.ones(
                (self.n_selections,), device=self.device
            )
        else:
            self.selection_costs = torch.tensor(
                selection_costs, device=self.device
            )

        self.rng = torch.Generator()
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

        self._make_spec()

    def _make_spec(self) -> None:
        self.observation_spec = Composite(
            # For binary tensorspecs, torchrl now forces us to specify how large the last dimension is, I'm not sure why.
            feature_mask=Binary(
                n=self.feature_shape[-1],
                shape=self.batch_size + self.feature_shape,
                dtype=torch.bool,
            ),
            performed_action_mask=Binary(
                n=self.n_selections + 1,
                shape=self.batch_size + torch.Size((self.n_selections + 1,)),
                dtype=torch.bool,
            ),
            # "action" does include the stop action
            allowed_action_mask=Binary(
                n=self.n_selections + 1,
                shape=self.batch_size + torch.Size((self.n_selections + 1,)),
                dtype=torch.bool,
            ),
            # "selections" does not include the stop action
            performed_selection_mask=Binary(
                n=self.n_selections,
                shape=self.batch_size + torch.Size((self.n_selections,)),
                dtype=torch.bool,
            ),
            masked_features=Unbounded(
                shape=self.batch_size + self.feature_shape,
                dtype=torch.float32,
            ),
            # hidden from the agent
            features=Unbounded(
                shape=self.batch_size + self.feature_shape,
                dtype=torch.float32,
            ),
            label=Unbounded(
                shape=self.batch_size + (self.n_classes,),
                dtype=torch.float32,
            ),
            accumulated_cost=Unbounded(
                shape=self.batch_size, dtype=torch.float32
            ),
            batch_size=self.batch_size,
        )
        # One action per feature + stop action
        self.action_spec = Categorical(
            n=self.n_selections + 1,
            shape=self.batch_size + torch.Size(()),
            dtype=torch.int64,
        )
        self.reward_spec = Unbounded(
            shape=self.batch_size + torch.Size((1,)), dtype=torch.float32
        )
        self.done_spec = Binary(
            n=1, shape=self.batch_size + torch.Size((1,)), dtype=torch.bool
        )

    @override
    def _reset(
        self, tensordict: TensorDictBase | None, **_: dict[str, Any]
    ) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict(
                {}, batch_size=self.batch_size, device=self.device
            )

        # Get a batch from the dataset
        features, label = self.dataset_fn(tensordict.batch_size)
        features: Features = features.to(tensordict.device)
        label: Label = label.to(tensordict.device)

        # Initialize features
        initial_feature_mask = self.initialize_fn(
            features=features, label=label, feature_shape=self.feature_shape
        )

        initial_masked_features = features.clone()
        initial_masked_features[~initial_feature_mask] = 0.0
        initial_selection_mask = torch.zeros(
            tensordict.batch_size + torch.Size((self.n_selections,)),
            dtype=torch.bool,
            device=tensordict.device,
        )
        if self.forbidden_selection_mask_fn is not None:
            initial_selection_mask = self.forbidden_selection_mask_fn(
                initial_feature_mask, self.feature_shape
            ).to(tensordict.device)
            assert initial_selection_mask.shape == (
                tensordict.batch_size + torch.Size((self.n_selections,))
            ), (
                "forbidden_selection_mask_fn must return selection-space mask "
                f"with shape {tensordict.batch_size + torch.Size((self.n_selections,))}, "
                f"got {initial_selection_mask.shape}."
            )
        initial_allowed_action_mask = torch.ones(
            tensordict.batch_size + torch.Size((self.n_selections + 1,)),
            dtype=torch.bool,
            device=tensordict.device,
        )
        initial_allowed_action_mask[:, 1:] = ~initial_selection_mask

        td = TensorDict(
            {
                "feature_mask": initial_feature_mask,
                "performed_action_mask": torch.zeros(
                    tensordict.batch_size
                    + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "allowed_action_mask": initial_allowed_action_mask,
                "performed_selection_mask": initial_selection_mask,
                "masked_features": initial_masked_features,
                "features": features,
                "label": label,
                "accumulated_cost": torch.zeros(
                    tensordict.batch_size,
                    dtype=torch.float32,
                    device=tensordict.device,
                ),
            },
            batch_size=tensordict.batch_size,
            device=tensordict.device,
        )

        # If stop action is not allowed, disable it in the action mask
        if not self.allow_stop_action:
            td["allowed_action_mask"][:, 0] = False

        return td

    @override
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        batch_numel = tensordict.batch_size.numel()
        batch_indices = torch.arange(batch_numel, device=tensordict.device)
        action = tensordict["action"].clone()
        action_values = (
            action.squeeze(-1)
            if action.ndim > 1 and action.shape[-1] == 1
            else action
        )

        # Match evaluator semantics: if a proposed selection would exceed the
        # hard budget, terminate without executing the acquisition.
        budget_exceeded = torch.zeros_like(action_values, dtype=torch.bool)
        if self.has_hard_budget:
            candidate_selection_mask = action_values != 0
            if candidate_selection_mask.any():
                proposed_costs = self.selection_costs[
                    (action_values[candidate_selection_mask] - 1).long()
                ]
                would_exceed = (
                    tensordict["accumulated_cost"][candidate_selection_mask]
                    + proposed_costs
                    > self.hard_budget
                )
                if would_exceed.any():
                    candidate_indices = batch_indices[candidate_selection_mask]
                    budget_exceeded[candidate_indices[would_exceed]] = True

        # Acquire new features only for selections that are both non-stop and
        # within budget.
        valid_selection_mask = (action_values != 0) & ~budget_exceeded
        new_feature_mask = tensordict["feature_mask"].clone()
        if valid_selection_mask.any():
            new_feature_mask_valid = self.unmask_fn(
                masked_features=tensordict["masked_features"][
                    valid_selection_mask
                ],
                feature_mask=tensordict["feature_mask"][valid_selection_mask],
                features=tensordict["features"][valid_selection_mask],
                afa_selection=(
                    action_values[valid_selection_mask] - 1
                ).unsqueeze(-1),
                selection_mask=tensordict["performed_selection_mask"][
                    valid_selection_mask
                ],
                label=tensordict["label"][valid_selection_mask],
                feature_shape=self.feature_shape,
            )
            new_feature_mask[valid_selection_mask] = new_feature_mask_valid

        new_masked_features = tensordict["features"].clone()
        new_masked_features[~new_feature_mask] = 0.0

        # Add up costs
        new_accumulated_cost = tensordict["accumulated_cost"].clone()
        if valid_selection_mask.any():
            new_accumulated_cost[valid_selection_mask] += self.selection_costs[
                (action_values[valid_selection_mask] - 1).long()
            ]

        # Update masks
        new_performed_action_mask = tensordict["performed_action_mask"].clone()
        new_allowed_action_mask = tensordict["allowed_action_mask"].clone()
        new_performed_selection_mask = tensordict[
            "performed_selection_mask"
        ].clone()

        stop_mask = action_values == 0
        if stop_mask.any():
            new_performed_action_mask[batch_indices[stop_mask], 0] = True

        # For valid selection actions, update selection mask and disable that action
        if valid_selection_mask.any():
            non_stop_indices = batch_indices[valid_selection_mask]
            selections = (action_values[valid_selection_mask] - 1).long()
            new_performed_action_mask[
                non_stop_indices, action_values[valid_selection_mask].long()
            ] = True
            new_performed_selection_mask[non_stop_indices, selections] = True
            new_allowed_action_mask[
                non_stop_indices, action_values[valid_selection_mask].long()
            ] = False

        # If stop action is not allowed, ensure it stays disabled
        if not self.allow_stop_action:
            new_allowed_action_mask[:, 0] = False

        # Done if we choose to stop, propose an over-budget selection, or all
        # selection actions are exhausted.
        # Check if all selection actions (actions 1 through n_selections) are disabled
        selection_actions_available = new_allowed_action_mask[:, 1:].any(
            dim=-1
        )
        done = (
            stop_mask.unsqueeze(-1)
            | budget_exceeded.unsqueeze(-1)
            | (~selection_actions_available).unsqueeze(-1)
        )

        # Always calculate a possible reward
        with torch.no_grad():
            reward = self.reward_fn(
                tensordict["masked_features"],
                tensordict["feature_mask"],
                tensordict["performed_selection_mask"],
                new_masked_features,
                new_feature_mask,
                new_performed_selection_mask,
                action,
                tensordict["features"],
                tensordict["label"],
                done,
            )

        r = TensorDict(
            {
                "performed_action_mask": new_performed_action_mask,
                "allowed_action_mask": new_allowed_action_mask,
                "performed_selection_mask": new_performed_selection_mask,
                "feature_mask": new_feature_mask,
                "masked_features": new_masked_features,
                "done": done,
                "reward": reward,
                # features and label are not cloned since they stay the same
                "features": tensordict["features"],
                "label": tensordict["label"],
                "accumulated_cost": new_accumulated_cost,
            },
            batch_size=tensordict.batch_size,
        )
        return r

    @override
    def _set_seed(self, seed: int | None) -> None:
        rng = torch.manual_seed(seed)
        self.rng = rng

    def get_batch_info(self, td: TensorDictBase) -> dict[str, Any]:
        """Return a wandb-loggable dictionary from a tensordict collected during training. Should only contain method-agnostic info."""
        # TODO:
        return {
            "avg_reward": td["next", "reward"].mean().item(),
            # Average number of features selected when we stop
            "fraction observed at stop time": td["next", "feature_mask"][
                td["next", "done"].squeeze(-1)
            ]
            .float()
            .mean()
            .cpu()
            .item(),
        }

    def get_rollout_info(
        self, rollout_tds: list[TensorDictBase]
    ) -> dict[str, Any]:
        """Return a wandb-loggable dictionary from a lits of tensordicts collected during evaluation rollouts. Should only contain method-agnostic info."""
        # Every rollout td has shape (n_agents, episode_len)
        flat_td = torch.cat(rollout_tds, dim=-1).flatten()  # pyright: ignore[reportArgumentType, reportCallIssue]
        return {"action": wandb.Histogram(flat_td["action"].cpu())}
