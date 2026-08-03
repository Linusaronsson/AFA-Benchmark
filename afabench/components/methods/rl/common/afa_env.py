from collections.abc import Sequence
from typing import Any, cast, final, override

import torch
import wandb
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Binary, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from afabench.components.methods.rl.common.custom_types import (
    AFADatasetFn,
    AFAFeatureRestorationFn,
    AFARewardFn,
)
from afabench.core.types import (
    AFAInitializeFn,
    AFAUnmaskFn,
    Features,
    Label,
)
from afabench.missing_values.stepwise import restore_acquired_features


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
        *,
        force_hard_budget: bool = False,  # if True and hard_budget is set, never allow the stop action
        seed: int | None = None,
        selection_costs: Sequence[float]
        | None = None,  # How much each sequence costs. If None, assume unit cost (1).
        feature_restoration_fn: AFAFeatureRestorationFn | None = None,
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
        self.feature_restoration_fn = feature_restoration_fn
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
            source_availability=Binary(
                n=self.feature_shape[-1],
                shape=self.batch_size + self.feature_shape,
                dtype=torch.bool,
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

    def _draw_dataset_batch(
        self,
        tensordict: TensorDictBase,
    ) -> tuple[Features, Label, torch.Tensor | None, torch.Tensor]:
        # TorchRL calls _reset whenever *any* sub-env is done, passing a mask of
        # which ones, and then keeps only those entries of what we return. Draw
        # exactly that many rows: drawing a full batch every time advanced the
        # dataset pointer far faster than samples were actually consumed (~4x
        # under per-sample availability), so most of the dataset was skipped.
        batch_numel = tensordict.batch_size.numel()
        reset_mask = tensordict.get("_reset", None)
        reset_idx = (
            None
            if reset_mask is None
            else reset_mask.reshape(batch_numel).nonzero(as_tuple=True)[0]
        )
        n_draw = batch_numel if reset_idx is None else int(reset_idx.numel())

        dataset_batch = self.dataset_fn(torch.Size((n_draw,)))
        features, label = dataset_batch[:2]
        selection_availability = (
            dataset_batch[2] if len(dataset_batch) >= 3 else None
        )
        source_availability = (
            dataset_batch[3] if len(dataset_batch) == 4 else None
        )
        features: Features = features.to(tensordict.device)
        label: Label = label.to(tensordict.device)
        if selection_availability is not None:
            selection_availability = selection_availability.to(
                tensordict.device
            )
        if source_availability is not None:
            source_availability = source_availability.to(tensordict.device)

        if reset_idx is not None:
            # Scatter the drawn rows back to full batch shape. Entries outside
            # the reset mask are never read, so their value does not matter.
            def _scatter(src: torch.Tensor) -> torch.Tensor:
                out = src.new_zeros((batch_numel, *src.shape[1:]))
                out[reset_idx] = src
                return out

            features = _scatter(features)
            label = _scatter(label)
            if selection_availability is not None:
                selection_availability = _scatter(selection_availability)
            if source_availability is not None:
                source_availability = _scatter(source_availability)

        if self.feature_restoration_fn is not None:
            if source_availability is None:
                msg = "Feature restoration requires source availability."
                raise ValueError(msg)
            features = features.clone()
            features[~source_availability] = 0.0
        elif source_availability is None:
            source_availability = torch.ones_like(features, dtype=torch.bool)
        return features, label, selection_availability, source_availability

    @override
    def _reset(
        self, tensordict: TensorDictBase | None, **_: dict[str, Any]
    ) -> TensorDict:
        if tensordict is None:
            tensordict = TensorDict(
                {}, batch_size=self.batch_size, device=self.device
            )

        (
            features,
            label,
            selection_availability,
            source_availability,
        ) = self._draw_dataset_batch(tensordict)

        # Initialize features
        initial_feature_mask = self.initialize_fn(
            features=features, label=label, feature_shape=self.feature_shape
        ).to(device=features.device, dtype=torch.bool)
        features = restore_acquired_features(
            features,
            torch.zeros_like(initial_feature_mask),
            initial_feature_mask,
            source_availability,
            self.feature_restoration_fn,
        )

        initial_masked_features = features.clone()
        initial_masked_features[~initial_feature_mask] = 0.0

        allowed_action_mask = torch.ones(
            tensordict.batch_size + torch.Size((self.n_selections + 1,)),
            dtype=torch.bool,
            device=tensordict.device,
        )
        if selection_availability is not None:
            allowed_action_mask[:, 1:] = selection_availability.to(
                device=tensordict.device,
                dtype=torch.bool,
            )

        td = TensorDict(
            {
                "feature_mask": initial_feature_mask,
                "performed_action_mask": torch.zeros(
                    tensordict.batch_size
                    + torch.Size((self.n_selections + 1,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "allowed_action_mask": allowed_action_mask,
                "performed_selection_mask": torch.zeros(
                    tensordict.batch_size + torch.Size((self.n_selections,)),
                    dtype=torch.bool,
                    device=tensordict.device,
                ),
                "masked_features": initial_masked_features,
                "features": features,
                "source_availability": source_availability,
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
            allowed_action_mask = cast(
                "torch.Tensor", td.get("allowed_action_mask")
            )
            no_selection_available = ~allowed_action_mask[:, 1:].any(dim=1)
            allowed_action_mask[:, 0] = no_selection_available

        return td

    @override
    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        batch_numel = tensordict.batch_size.numel()
        batch_indices = torch.arange(batch_numel, device=tensordict.device)
        action = cast("torch.Tensor", tensordict.get("action"))

        # Acquire new features from unmasker if we don't choose the stop action
        no_stop_mask = action != 0
        new_feature_mask_no_stop = self.unmask_fn(
            masked_features=tensordict["masked_features"][no_stop_mask],
            feature_mask=tensordict["feature_mask"][no_stop_mask],
            features=tensordict["features"][no_stop_mask],
            afa_selection=(action - 1)[no_stop_mask].unsqueeze(-1),
            selection_mask=tensordict["performed_selection_mask"][
                no_stop_mask
            ],
            label=tensordict["label"][no_stop_mask],
            feature_shape=self.feature_shape,
        )
        new_feature_mask = tensordict["feature_mask"].clone()
        new_feature_mask[no_stop_mask] = new_feature_mask_no_stop

        new_features = restore_acquired_features(
            tensordict["features"],
            tensordict["feature_mask"],
            new_feature_mask,
            tensordict["source_availability"],
            self.feature_restoration_fn,
        )
        new_masked_features = new_features.clone()
        new_masked_features[~new_feature_mask] = 0.0

        # Add up costs
        new_accumulated_cost = tensordict["accumulated_cost"].clone()
        new_accumulated_cost[no_stop_mask] += self.selection_costs[
            (action - 1)[no_stop_mask]
        ]

        # Update masks
        new_performed_action_mask = tensordict["performed_action_mask"].clone()
        new_performed_action_mask[batch_indices, action] = True
        new_allowed_action_mask = tensordict["allowed_action_mask"].clone()
        new_performed_selection_mask = tensordict[
            "performed_selection_mask"
        ].clone()

        # For non-stop actions, update selection mask and disable that action
        if no_stop_mask.any():
            non_stop_indices = batch_indices[no_stop_mask]
            selections = action[no_stop_mask] - 1
            new_performed_selection_mask[non_stop_indices, selections] = True
            new_allowed_action_mask[non_stop_indices, action[no_stop_mask]] = (
                False
            )

        # If stop action is not allowed, ensure it stays disabled
        if not self.allow_stop_action:
            new_allowed_action_mask[:, 0] = False

        # Done if we **exceed** the hard budget, have chosen all the actions, choose to stop (action 0),
        # or all selection actions are exhausted
        # Check if all selection actions (actions 1 through n_selections) are disabled
        selection_actions_available = cast(
            "torch.Tensor",
            new_allowed_action_mask[:, 1:].any(dim=-1),
        )
        done = (
            ((new_accumulated_cost > self.hard_budget).unsqueeze(-1))
            | (action == 0).unsqueeze(-1)
            | torch.logical_not(selection_actions_available).unsqueeze(-1)
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
                new_features,
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
                "features": new_features,
                "source_availability": tensordict["source_availability"],
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
