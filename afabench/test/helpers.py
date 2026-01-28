import torch
from torch.nn import functional as F

from afabench.common.custom_types import (
    AFAAction,
    AFAActionFn,
    AFAPredictFn,
    AFASelection,
    AFAUnmaskFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_random_afa_predict_fn(n_classes: int) -> AFAPredictFn:
    """Return an AFAPredictFn that randomly chooses a class."""

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> Label:
        assert masked_features.ndim == 2
        batch_size = masked_features.shape[0]
        t = torch.randint(low=0, high=n_classes, size=(batch_size, 1))
        return F.one_hot(t, num_classes=n_classes)

    return f


def get_deterministic_afa_predict_fn(
    predictions: list[list[int]], n_classes: int
) -> AFAPredictFn:
    """Return an AFAPredictFn that chooses classes in a deterministic way."""

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> Label:
        assert masked_features.ndim == 2
        batch_size = masked_features.shape[0]
        assert batch_size == len(predictions)
        t = torch.zeros((batch_size, n_classes))
        for sample in range(batch_size):
            time_step = feature_mask[sample].sum()
            if time_step >= len(predictions[sample]):
                time_step = len(predictions[sample]) - 1
            t[sample] = F.one_hot(
                torch.tensor(predictions[sample][time_step]),
                num_classes=n_classes,
            )
        return t

    return f


def get_sequential_action_fn() -> AFAActionFn:
    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        """Return an AFAActionFn that picks actions sequentially, ending with the stop action."""
        assert selection_mask is not None
        assert selection_mask.ndim == 2
        assert masked_features.ndim == 2
        batch_size = masked_features.shape[0]
        t = torch.full((batch_size, 1), -1, dtype=torch.int)
        for sample in range(batch_size):
            choose_stop_action = True
            for selection_idx, selection in enumerate(selection_mask[sample]):
                if not selection:
                    t[sample] = selection_idx + 1
                    choose_stop_action = False
                    break
            if choose_stop_action:
                t[sample] = 0
        assert not (t == -1).any()

        return t

    return f


def get_deterministic_action_fn(actions: list[list[int]]) -> AFAActionFn:
    """Return an AFAActionFn that picks actions in a deterministic order."""

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> AFAAction:
        assert selection_mask is not None
        assert selection_mask.ndim == 2
        assert masked_features.ndim == 2
        batch_size = masked_features.shape[0]
        t = torch.full((batch_size, 1), -1, dtype=torch.int)
        for sample in range(batch_size):
            time = selection_mask[sample].sum()
            t[sample] = actions[sample][time]
        assert not (t == -1).any()

        return t

    return f


def get_direct_unmask_fn() -> AFAUnmaskFn:
    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,  # noqa: ARG001
        afa_selection: AFASelection,
        selection_mask: SelectionMask,  # noqa: ARG001
        label: Label | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> FeatureMask:
        assert masked_features.ndim == 2
        batch_size = masked_features.shape[0]
        new_feature_mask = feature_mask.clone()
        new_feature_mask[
            torch.arange(batch_size), afa_selection.squeeze(-1)
        ] = True
        return new_feature_mask

    return f
