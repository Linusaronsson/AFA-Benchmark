from unittest import TestCase

import torch

from afabench.afa_rl.common.dataset_utils import get_wrapped_batch


class TestDatasets(TestCase):
    def test_get_wrapped_batch(self) -> None:
        t = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int64)
        torch.testing.assert_close(
            get_wrapped_batch(t, idx=1, numel=2),
            torch.tensor([[3, 4], [5, 6]], dtype=torch.int64),
        )
        torch.testing.assert_close(
            get_wrapped_batch(t, idx=1, numel=3),
            torch.tensor([[3, 4], [5, 6], [1, 2]], dtype=torch.int64),
        )
        torch.testing.assert_close(
            get_wrapped_batch(t, idx=1, numel=6),
            torch.tensor(
                [[3, 4], [5, 6], [1, 2], [3, 4], [5, 6], [1, 2]],
                dtype=torch.int64,
            ),
        )
