from pathlib import Path

import pytest

from afabench.training.supervised_learning import lightning_root


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
