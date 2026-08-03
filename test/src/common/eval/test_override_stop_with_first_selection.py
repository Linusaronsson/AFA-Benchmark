import torch

from afabench.evaluation.eval import override_stop_with_first_selection


def test_only_overrides_stop_actions() -> None:
    afa_action = torch.tensor([[0], [2]], dtype=torch.int)
    new_afa_action = override_stop_with_first_selection(
        afa_action=afa_action,
        selection_mask=torch.tensor([[1, 0, 1], [0, 0, 0]]),
    )
    assert torch.equal(new_afa_action, torch.tensor([[2], [2]]))


def test_keeps_stop_when_no_selection_is_available() -> None:
    afa_action = torch.tensor([[0], [0]], dtype=torch.int)
    new_afa_action = override_stop_with_first_selection(
        afa_action=afa_action,
        selection_mask=torch.tensor([[1, 1, 1], [1, 0, 1]]),
    )
    assert torch.equal(new_afa_action, torch.tensor([[0], [2]]))
