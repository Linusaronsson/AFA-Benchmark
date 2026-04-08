from __future__ import annotations

from typing import final

import torch

from afabench.afa_rl.common.afa_env import AFAEnv
from afabench.afa_rl.common.dataset_utils import get_afa_dataset_fn
from afabench.afa_rl.common.reward_functions import get_fixed_reward_reward_fn
from afabench.afa_rl.common.training import (
    _build_env_mask_fns,
    _initializer_has_training_support_restriction,
)
from afabench.common.initializers.manual_initializer import ManualInitializer
from afabench.common.initializers.missingness_initializer import (
    MissingnessInitializer,
)
from afabench.common.unmaskers.cube_nm_unmasker import (
    CubeNMUnmasker,
)
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker


@final
class _ToyRestrictedInitializer:
    _observed_mask: torch.Tensor
    _training_forbidden_mask: torch.Tensor

    def __init__(
        self,
        observed_mask: torch.Tensor,
        training_forbidden_mask: torch.Tensor,
    ) -> None:
        self._observed_mask = observed_mask.bool()
        self._training_forbidden_mask = training_forbidden_mask.bool()

    def set_seed(self, seed: int | None) -> None:
        del seed

    def initialize(
        self,
        features: torch.Tensor,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del features, label
        assert feature_shape is not None
        return self._observed_mask.clone()

    def get_training_forbidden_mask(
        self,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        del observed_mask
        return self._training_forbidden_mask.clone()


@final
class _ToyWarmStartInitializer:
    _observed_mask: torch.Tensor

    def __init__(self, observed_mask: torch.Tensor) -> None:
        self._observed_mask = observed_mask.bool()

    def set_seed(self, seed: int | None) -> None:
        del seed

    def initialize(
        self,
        features: torch.Tensor,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del features, label
        assert feature_shape is not None
        return self._observed_mask.clone()

    def get_training_forbidden_mask(
        self,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(observed_mask, dtype=torch.bool)


def test_initializer_support_restriction_detection() -> None:
    assert _initializer_has_training_support_restriction(
        MissingnessInitializer(mechanism="mcar", p=0.3)
    )
    assert not _initializer_has_training_support_restriction(
        ManualInitializer(flat_feature_indices=[0, 2])
    )


def test_train_restricted_env_starts_cold_and_blocks_forbidden_actions() -> (
    None
):
    observed_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ],
        dtype=torch.bool,
    )
    training_forbidden_mask = ~observed_mask
    initializer = _ToyRestrictedInitializer(
        observed_mask=observed_mask,
        training_forbidden_mask=training_forbidden_mask,
    )

    initialize_fn, forbidden_selection_mask_fn = _build_env_mask_fns(
        initializer,
        DirectUnmasker(),
        n_selection_choices=4,
        mode="train_restricted",
    )
    assert forbidden_selection_mask_fn is not None

    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )
    all_labels = torch.tensor([[1, 0], [0, 1]])
    env = AFAEnv(
        dataset_fn=get_afa_dataset_fn(all_features, all_labels, shuffle=False),
        reward_fn=get_fixed_reward_reward_fn(
            reward_for_stop=0.0,
            reward_otherwise=-1.0,
        ),
        device=torch.device("cpu"),
        batch_size=torch.Size((2,)),
        feature_shape=torch.Size((4,)),
        n_selections=4,
        n_classes=2,
        hard_budget=4.0,
        initialize_fn=initialize_fn,
        unmask_fn=DirectUnmasker().unmask,
        forbidden_selection_mask_fn=forbidden_selection_mask_fn,
        seed=123,
    )

    td = env.reset()

    expected_allowed_action_mask = torch.tensor(
        [
            [True, True, False, True, False],
            [True, False, True, False, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(
        td["feature_mask"],
        torch.zeros_like(observed_mask, dtype=torch.bool),
    )
    assert torch.equal(
        td["masked_features"],
        torch.zeros_like(all_features, dtype=torch.float32),
    )
    assert torch.equal(
        td["performed_selection_mask"],
        training_forbidden_mask,
    )
    assert torch.equal(
        td["allowed_action_mask"],
        expected_allowed_action_mask,
    )


def test_train_restricted_masks_adapt_to_grouped_selection_space() -> None:
    training_forbidden_mask = torch.tensor(
        [[True, False, False, True, False, True]],
        dtype=torch.bool,
    )
    observed_mask = ~training_forbidden_mask
    initializer = _ToyRestrictedInitializer(
        observed_mask=observed_mask,
        training_forbidden_mask=training_forbidden_mask,
    )
    n_contexts = 2
    unmasker = CubeNMUnmasker(n_contexts=n_contexts)
    n_selection_choices = unmasker.get_n_selections(torch.Size((6,)))

    initialize_fn, forbidden_selection_mask_fn = _build_env_mask_fns(
        initializer,
        unmasker,
        n_selection_choices=n_selection_choices,
        mode="train_restricted",
    )
    assert forbidden_selection_mask_fn is not None

    cold_mask = initialize_fn(
        features=torch.zeros((1, 6), dtype=torch.float32),
        label=torch.zeros((1, 1), dtype=torch.float32),
        feature_shape=torch.Size((6,)),
    )
    selection_forbidden = forbidden_selection_mask_fn(
        cold_mask,
        torch.Size((6,)),
    )

    expected = torch.tensor([[True, False, True, False, True]])
    assert torch.equal(cold_mask, torch.zeros_like(observed_mask))
    assert torch.equal(selection_forbidden, expected)


def test_eval_cold_mode_restores_standard_cold_start() -> None:
    initializer = _ToyRestrictedInitializer(
        observed_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
        training_forbidden_mask=torch.tensor(
            [[False, True, False]], dtype=torch.bool
        ),
    )
    initialize_fn, forbidden_selection_mask_fn = _build_env_mask_fns(
        initializer,
        DirectUnmasker(),
        n_selection_choices=3,
        mode="eval_cold",
    )

    cold_mask = initialize_fn(
        features=torch.ones((1, 3), dtype=torch.float32),
        label=torch.zeros((1, 1), dtype=torch.float32),
        feature_shape=torch.Size((3,)),
    )

    assert torch.equal(cold_mask, torch.zeros((1, 3), dtype=torch.bool))
    assert forbidden_selection_mask_fn is None


def test_default_mode_preserves_warm_start_for_nonrestricted_initializer() -> (
    None
):
    observed_mask = torch.tensor(
        [[True, False, True, False]], dtype=torch.bool
    )
    initialize_fn, forbidden_selection_mask_fn = _build_env_mask_fns(
        _ToyWarmStartInitializer(observed_mask),
        DirectUnmasker(),
        n_selection_choices=4,
        mode="default",
    )

    returned_mask = initialize_fn(
        features=torch.ones((1, 4), dtype=torch.float32),
        label=torch.zeros((1, 1), dtype=torch.float32),
        feature_shape=torch.Size((4,)),
    )

    assert torch.equal(returned_mask, observed_mask)
    assert forbidden_selection_mask_fn is None
