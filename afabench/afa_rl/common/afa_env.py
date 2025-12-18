from typing import TYPE_CHECKING, Any, final, override

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

import wandb
from afabench.afa_rl.common.custom_types import (
    AFADatasetFn,
    AFARewardFn,
)
from afabench.common.custom_types import AFAInitializeFn, AFAUnmaskFn

if TYPE_CHECKING:
    from afabench.common.custom_types import (
        Features,
        Label,
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
        device: torch.device,
        batch_size: torch.Size,
        feature_shape: torch.Size,
        n_selections: int,  # action dim = n_selections + 1 since we have a stop action as well
        n_classes: int,
        hard_budget: int
        | None,  # how many selections can be performed before the episode ends. If None, no limit.
        initialize_fn: AFAInitializeFn,
        unmask_fn: AFAUnmaskFn,
        *,
        force_hard_budget: bool = False,  # if True and hard_budget is set, never allow the stop action
        seed: int | None = None,
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
        self.seed = seed

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

        td = TensorDict(
            {
                "feature_mask": initial_feature_mask,
                "performed_action_mask": torch.zeros(
                    tensordict.batch_size
                    + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "allowed_action_mask": torch.ones(
                    tensordict.batch_size
                    + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "performed_selection_mask": torch.zeros(
                    tensordict.batch_size + torch.Size((self.n_selections,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "masked_features": initial_masked_features,
                "features": features,
                "label": label,
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
        # new_feature_mask: FeatureMask = tensordict["feature_mask"].clone()
        # new_masked_features: MaskedFeatures = tensordict[
        #     "masked_features"
        # ].clone()

        batch_numel = tensordict.batch_size.numel()
        batch_indices = torch.arange(batch_numel, device=tensordict.device)

        # Acquire new features from unmasker
        new_feature_mask = self.unmask_fn(
            masked_features=tensordict["masked_features"],
            feature_mask=tensordict["feature_mask"],
            features=tensordict["features"],
            afa_selection=tensordict["action"].unsqueeze(-1),
            selection_mask=tensordict["performed_selection_mask"],
            label=tensordict["label"],
            feature_shape=self.feature_shape,
        )

        new_masked_features = tensordict["features"].clone()
        new_masked_features[~new_feature_mask] = 0.0

        # Update masks
        action_idx = tensordict["action"].squeeze(-1)
        new_performed_action_mask = tensordict["performed_action_mask"].clone()
        new_performed_action_mask[batch_indices, action_idx] = True
        new_allowed_action_mask = tensordict["allowed_action_mask"].clone()
        new_performed_selection_mask = tensordict[
            "performed_selection_mask"
        ].clone()

        # For non-stop actions, update selection mask and disable that action
        non_stop_mask = action_idx > 0
        if non_stop_mask.any():
            non_stop_indices = batch_indices[non_stop_mask]
            selection_indices = (
                action_idx[non_stop_mask] - 1
            )  # Convert to 0-based selection index
            new_performed_selection_mask[
                non_stop_indices, selection_indices
            ] = True
            new_allowed_action_mask[
                non_stop_indices, action_idx[non_stop_mask]
            ] = False

        # If stop action is not allowed, ensure it stays disabled
        if not self.allow_stop_action:
            new_allowed_action_mask[:, 0] = False

        # Done if we exceed the hard budget, have chosen all the actions, choose to stop (action 0),
        # or all selection actions are exhausted
        selections_taken = new_performed_selection_mask.sum(-1)
        # Check if all selection actions (actions 1 through n_selections) are disabled
        selection_actions_available = new_allowed_action_mask[:, 1:].any(
            dim=-1
        )
        done = (
            ((selections_taken >= self.hard_budget).unsqueeze(-1))
            | (tensordict["action"] == 0).unsqueeze(-1)
            | (~selection_actions_available).unsqueeze(-1)
        )

        # Always calculate a possible reward
        reward = self.reward_fn(
            tensordict["masked_features"],
            tensordict["feature_mask"],
            tensordict["performed_selection_mask"],
            new_masked_features,
            new_feature_mask,
            new_performed_selection_mask,
            tensordict["action"],
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
        return {"action": wandb.Histogram(flat_td["action"])}
