from pathlib import Path

import pytest
import torch

from afabench.common.datasets.datasets import DiabetesDataset, MiniBooNEDataset

pytestmark = pytest.mark.optional


def test_feature_costs_non_uniform_when_csv_present() -> None:
    cost_path = Path("extra/data/misc/feature_costs/diabetes.csv")
    if not cost_path.exists():
        pytest.skip("Missing diabetes cost file")
    dataset = DiabetesDataset(root="extra/data/misc/diabetes.csv")
    costs = dataset.get_feature_acquisition_costs()
    assert costs.shape == dataset.feature_shape
    assert not torch.allclose(costs, torch.ones_like(costs))
    assert costs.min() < costs.max()


def test_feature_costs_uniform_when_csv_missing() -> None:
    cost_path = Path("extra/data/misc/feature_costs/miniboone.csv")
    if cost_path.exists():
        pytest.skip("Miniboone cost file present; cannot test fallback.")
    dataset = MiniBooNEDataset(root="extra/data/misc/miniboone.csv")
    costs = dataset.get_feature_acquisition_costs()
    assert torch.allclose(costs, torch.ones_like(costs))
