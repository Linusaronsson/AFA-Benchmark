from typing import TYPE_CHECKING, cast

import torch

from afabench.afa_oracle.afa_methods import AACOAFAMethod

if TYPE_CHECKING:
    from afabench.afa_oracle.aaco_core import AACOOracle


class DummyOracle:
    def __init__(self, next_selection: int | None):
        self.next_selection: int | None = next_selection
        self.last_selection_mask: torch.Tensor | None = None
        self.last_selection_to_feature_mask: torch.Tensor | None = None
        self.last_selection_costs: torch.Tensor | None = None

    def to(self, _device: torch.device) -> "DummyOracle":
        return self

    def set_classifier(self, classifier) -> None:  # noqa: ANN001
        pass

    def select_next_selection(
        self,
        x_observed: torch.Tensor,  # noqa: ARG002
        observed_mask: torch.Tensor,  # noqa: ARG002
        selection_mask: torch.Tensor,
        selection_to_feature_mask: torch.Tensor,
        selection_costs: torch.Tensor | None = None,
        instance_idx: int = 0,  # noqa: ARG002
        *,
        force_acquisition: bool = False,  # noqa: ARG002
        exclude_instance: bool = True,  # noqa: ARG002
    ) -> int | None:
        self.last_selection_mask = selection_mask.clone()
        self.last_selection_to_feature_mask = selection_to_feature_mask.clone()
        self.last_selection_costs = (
            None if selection_costs is None else selection_costs.clone()
        )
        return self.next_selection

    def select_next_feature(
        self,
        x_observed: torch.Tensor,  # noqa: ARG002
        observed_mask: torch.Tensor,  # noqa: ARG002
        instance_idx: int = 0,  # noqa: ARG002
        *,
        force_acquisition: bool = False,  # noqa: ARG002
        exclude_instance: bool = True,  # noqa: ARG002
        feature_shape: torch.Size | None = None,  # noqa: ARG002
        selection_size: int | None = None,  # noqa: ARG002
        selection_costs: torch.Tensor | None = None,  # noqa: ARG002
    ) -> int | None:
        msg = "Feature-space oracle path should not be used in this test."
        raise AssertionError(msg)

    def predict_with_mask(
        self,
        x_observed: torch.Tensor,  # noqa: ARG002
        observed_mask: torch.Tensor,  # noqa: ARG002
    ) -> torch.Tensor:
        return torch.tensor([1.0, 0.0])


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    masked_features = torch.zeros((1, 55), dtype=torch.float32)
    feature_mask = torch.zeros((1, 55), dtype=torch.bool)
    feature_mask[0, 0] = True
    selection_mask = torch.zeros((1, 51), dtype=torch.bool)
    return masked_features, feature_mask, selection_mask


def test_cube_nm_selection_space_path_maps_to_actions() -> None:
    oracle = DummyOracle(next_selection=0)
    method = AACOAFAMethod(
        aaco_oracle=cast("AACOOracle", cast("object", oracle)),
        dataset_name="cube_nm",
        unmasker_class_name="CubeNMUnmasker",
        unmasker_kwargs={"n_contexts": 5},
        _selection_size=51,
        _selection_costs=torch.ones(51),
    )

    masked_features, feature_mask, selection_mask = _make_inputs()
    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([55]),
    )

    assert action.shape == (1, 1)
    assert int(action.item()) == 1

    assert oracle.last_selection_mask is not None
    assert oracle.last_selection_mask.shape == (51,)

    assert oracle.last_selection_to_feature_mask is not None
    mask_table = oracle.last_selection_to_feature_mask
    assert mask_table.shape == (51, 55)
    assert mask_table[0, :5].all()
    assert int(mask_table[0, 5:].sum().item()) == 0
    assert bool(mask_table[1, 5].item())
    assert int(mask_table[1].sum().item()) == 1

    assert oracle.last_selection_costs is not None
    assert oracle.last_selection_costs.shape == (51,)


def test_cube_nm_selection_space_stop_maps_to_zero_action() -> None:
    oracle = DummyOracle(next_selection=None)
    method = AACOAFAMethod(
        aaco_oracle=cast("AACOOracle", cast("object", oracle)),
        dataset_name="cube_nm",
        unmasker_class_name="CubeNMUnmasker",
        unmasker_kwargs={"n_contexts": 5},
        _selection_size=51,
    )

    masked_features, feature_mask, selection_mask = _make_inputs()
    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([55]),
    )

    assert int(action.item()) == 0


def test_no_observation_selection_space_path_maps_to_actions() -> None:
    oracle = DummyOracle(next_selection=7)
    method = AACOAFAMethod(
        aaco_oracle=cast("AACOOracle", cast("object", oracle)),
        dataset_name="cube_nm",
        unmasker_class_name="CubeNMUnmasker",
        unmasker_kwargs={"n_contexts": 5},
        _selection_size=51,
    )

    masked_features = torch.zeros((1, 55), dtype=torch.float32)
    feature_mask = torch.zeros((1, 55), dtype=torch.bool)
    selection_mask = torch.ones((1, 51), dtype=torch.bool)
    selection_mask[0, 7] = False

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([55]),
    )

    assert int(action.item()) == 8


def test_no_observation_selection_space_path_maps_stop() -> None:
    oracle = DummyOracle(next_selection=None)
    method = AACOAFAMethod(
        aaco_oracle=cast("AACOOracle", cast("object", oracle)),
        dataset_name="cube_nm",
        unmasker_class_name="CubeNMUnmasker",
        unmasker_kwargs={"n_contexts": 5},
        _selection_size=51,
    )

    masked_features = torch.zeros((1, 55), dtype=torch.float32)
    feature_mask = torch.zeros((1, 55), dtype=torch.bool)
    selection_mask = torch.zeros((1, 51), dtype=torch.bool)

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([55]),
    )

    assert int(action.item()) == 0
