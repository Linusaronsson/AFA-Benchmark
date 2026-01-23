import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from afabench.common.registry import get_class

pytestmark = pytest.mark.optional

DATASETS_TO_TEST = [
    ("AFAContextDataset", {"n_samples": 10, "seed": 42}),
    ("CubeDataset", {"n_samples": 10, "seed": 42}),
    ("DiabetesDataset", {"root": "extra/data/misc/diabetes.csv"}),
    ("MiniBooNEDataset", {"root": "extra/data/misc/miniboone.csv"}),
    ("PhysionetDataset", {"root": "extra/data/misc/physionet.csv"}),
    # No {(Fashion)MNISTDataset, ImagenetteDataset} because of image data and large size
    ("BankMarketingDataset", {"path": "extra/data/misc/bank-marketing.csv"}),
    ("CKDDataset", {"path": "extra/data/misc/chronic_kidney_disease.csv"}),
    ("ACTG175Dataset", {"path": "extra/data/misc/actg.csv"}),
]


@pytest.mark.parametrize(("dataset_name", "kwargs"), DATASETS_TO_TEST)
def test_dataset_roundtrip(dataset_name: str, kwargs: dict[str, Any]) -> None:
    """Verify that every dataset class can save and reload itself losslessly."""
    dataset_class = get_class(dataset_name)

    # Instantiate dataset
    dataset = dataset_class(**kwargs)
    orig_features, orig_labels = dataset.get_all_data()

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "data.pt"

        # Save
        dataset.save(save_path)

        # Load
        loaded = dataset_class.load(save_path)
        loaded_features, loaded_labels = loaded.get_all_data()

    # Compare tensors
    assert torch.allclose(orig_features, loaded_features), (
        f"{dataset_name}: Features mismatch after save/load"
    )
    assert torch.allclose(orig_labels, loaded_labels), (
        f"{dataset_name}: Labels mismatch after save/load"
    )
