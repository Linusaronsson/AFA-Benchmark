import os

import pytest
import torch

from afabench.core.utils import set_seed


def test_set_seed_enables_deterministic_torch_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    set_seed(7)
    first = torch.rand(4)
    set_seed(7)

    assert torch.equal(first, torch.rand(4))
    assert torch.are_deterministic_algorithms_enabled()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
