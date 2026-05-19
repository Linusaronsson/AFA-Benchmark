import torch

from afabench.eval.eval import override_stop_with_first_selection


def test_only_overrides_stop_actions() -> None:
    afa_action = torch.tensor([[0], [2]], dtype=torch.int)
    new_afa_action = override_stop_with_first_selection(
        afa_action=afa_action,
        selection_mask=torch.tensor([[1, 0, 1], [0, 0, 0]]),
    )
    assert torch.equal(new_afa_action, torch.tensor([[2], [2]]))


def test_supports_multidim_selections() -> None:
    new_afa_action = override_stop_with_first_selection(
        afa_action=torch.tensor([[0], [0]]),
        selection_mask=torch.tensor(
            [
                [[1, 0, 1], [0, 1, 1]],
                [[1, 1, 1], [0, 1, 1]],
            ]
        ),
    )
    assert torch.equal(new_afa_action, torch.tensor([[2], [4]]))
