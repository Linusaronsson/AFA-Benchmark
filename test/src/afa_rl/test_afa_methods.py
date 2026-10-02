from typing import override

import torch
from tensordict import TensorDict

from afabench.components.classifiers import UniformDummyAFAClassifier
from afabench.components.methods.rl.common.afa_methods import RLAFAMethod


class _RecordingPolicy(torch.nn.Module):
    """Records the tensordicts it is called with."""

    def __init__(self) -> None:
        super().__init__()
        self.received_allowed_action_masks: list[torch.Tensor] = []

    @override
    def forward(self, td: TensorDict) -> TensorDict:
        allowed_action_mask = td["allowed_action_mask"].bool()
        self.received_allowed_action_masks.append(allowed_action_mask.clone())
        # Deterministically pick the first allowed feature action, falling
        # back to stop only when no feature action is allowed.
        actions = torch.zeros(allowed_action_mask.shape[0], dtype=torch.long)
        for row_idx, row_mask in enumerate(allowed_action_mask):
            allowed_feature_actions = (
                row_mask[1:].nonzero(as_tuple=False).flatten()
            )
            if allowed_feature_actions.numel() > 0:
                actions[row_idx] = allowed_feature_actions[0] + 1
        td["action"] = actions
        return td


def test_act_stops_exhausted_instances_without_calling_policy() -> None:
    policy = _RecordingPolicy()
    method = RLAFAMethod(
        policy_tdmodule=policy,  # type: ignore[arg-type]
        afa_classifier=UniformDummyAFAClassifier(n_classes=2),
        _device=torch.device("cpu"),
    )

    # Instance 0 is exhausted (every feature selected), instance 1 still has
    # a feature left to select.
    selection_mask = torch.tensor(
        [[True, True], [True, False]], dtype=torch.bool
    )

    actions = method.act(
        masked_features=torch.zeros((2, 2)),
        feature_mask=torch.zeros((2, 2), dtype=torch.bool),
        selection_mask=selection_mask,
    )

    assert torch.equal(actions, torch.tensor([[0], [2]]))

    # The policy must only ever see the non-exhausted instance.
    assert len(policy.received_allowed_action_masks) == 1
    received_mask = policy.received_allowed_action_masks[0]
    assert received_mask.shape[0] == 1
    # Column 0 is the stop action, so check that some *selection* is still
    # allowed: an all-performed selection mask would make this all-False.
    assert received_mask[:, 1:].any(dim=-1).all()


def test_act_never_invokes_policy_when_all_instances_exhausted() -> None:
    policy = _RecordingPolicy()
    method = RLAFAMethod(
        policy_tdmodule=policy,  # type: ignore[arg-type]
        afa_classifier=UniformDummyAFAClassifier(n_classes=2),
        _device=torch.device("cpu"),
    )

    selection_mask = torch.ones((3, 2), dtype=torch.bool)

    actions = method.act(
        masked_features=torch.zeros((3, 2)),
        feature_mask=torch.zeros((3, 2), dtype=torch.bool),
        selection_mask=selection_mask,
    )

    assert torch.equal(actions, torch.zeros((3, 1), dtype=torch.long))
    assert len(policy.received_allowed_action_masks) == 0
