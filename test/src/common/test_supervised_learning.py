from pathlib import Path

import pytest
import torch
from torch import nn

from afabench.training.supervised_learning import (
    ensure_finite_module_state,
    lightning_root,
)


def test_lightning_uses_node_local_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SNIC_TMP", "/scratch/local/job")

    assert lightning_root() == Path("/scratch/local/job")


def test_lightning_keeps_local_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SNIC_TMP", raising=False)

    assert lightning_root() == Path("extra/logs/lightning")


def test_nonfinite_model_state_is_rejected() -> None:
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="weight"):
        ensure_finite_module_state(model)
