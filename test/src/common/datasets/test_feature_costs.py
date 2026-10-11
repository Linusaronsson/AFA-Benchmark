from pathlib import Path

import pytest
import torch

from afabench.datasets.datasets import DiabetesDataset, MiniBooNEDataset

pytestmark = pytest.mark.optional


def test_feature_costs_non_uniform_when_csv_present() -> None:
    cost_path = Path("data/diabetes/feature_costs.csv")
    if not cost_path.exists():
        pytest.skip("Missing diabetes cost file")
    dataset = DiabetesDataset(root="data/diabetes/diabetes.csv")
    costs = dataset.get_feature_acquisition_costs()
    assert costs.shape == dataset.feature_shape
    assert not torch.allclose(costs, torch.ones_like(costs))
    assert costs.min() < costs.max()


def test_feature_costs_uniform_when_csv_missing() -> None:
    cost_path = Path("data/miniboone/feature_costs.csv")
    if cost_path.exists():
        pytest.skip("Miniboone cost file present; cannot test fallback.")
    dataset = MiniBooNEDataset(root="data/miniboone/miniboone.csv")
    costs = dataset.get_feature_acquisition_costs()
    assert torch.allclose(costs, torch.ones_like(costs))
