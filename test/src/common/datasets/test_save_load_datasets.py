import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch

from afabench.common.registry import get_class

pytestmark = pytest.mark.optional

DATASETS_TO_TEST = [
    ("CubeNMDataset", {"n_samples": 10, "seed": 42}),
    ("CubeNMARDataset", {"n_samples": 10, "seed": 42}),
    ("CubeDataset", {"n_samples": 10, "seed": 42}),
    ("DiabetesDataset", {"root": "extra/data/misc/diabetes.csv"}),
    ("MiniBooNEDataset", {"root": "extra/data/misc/miniboone.csv"}),
    ("PhysionetDataset", {"root": "extra/data/misc/physionet.csv"}),
    ("FICODataset", {"path": "extra/data/misc/fico.csv"}),
    (
        "PharyngitisDataset",
        {"path": "extra/data/misc/pharyngitis.xls"},
    ),
    # No {(Fashion)MNISTDataset, ImagenetteDataset} because of image data and large size
    ("BankMarketingDataset", {"path": "extra/data/misc/bank-marketing.csv"}),
    ("CKDDataset", {"path": "extra/data/misc/chronic_kidney_disease.csv"}),
    ("ACTG175Dataset", {"path": "extra/data/misc/actg.csv"}),
]

# Datasets that require manually placing local files in extra/data/misc.
# Other datasets with path/root arguments are expected to auto-fetch.
MANUAL_LOCAL_ONLY_DATASETS = {
    "DiabetesDataset",
    "MiniBooNEDataset",
    "PhysionetDataset",
    "PharyngitisDataset",
}


@pytest.mark.parametrize(("dataset_name", "kwargs"), DATASETS_TO_TEST)
def test_dataset_roundtrip(dataset_name: str, kwargs: dict[str, Any]) -> None:
    """Verify that every dataset class can save and reload itself losslessly."""
    dataset_class = get_class(dataset_name)

    local_data_path = kwargs.get("path") or kwargs.get("root")
    if (
        dataset_name in MANUAL_LOCAL_ONLY_DATASETS
        and local_data_path is not None
        and not Path(local_data_path).exists()
    ):
        pytest.skip(f"Missing dataset file: {local_data_path}")

    # Instantiate dataset
    dataset = dataset_class(**kwargs)
    orig_features, orig_labels = dataset.get_all_data()

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / "data.bundle"
        save_path.mkdir(parents=True, exist_ok=True)

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


def test_cube_nm_ar_feature_costs_match_feature_shape() -> None:
    dataset_class = get_class("CubeNMARDataset")
    dataset = dataset_class(n_samples=10, seed=42)

    feature_costs = dataset.get_feature_acquisition_costs()

    assert feature_costs.shape == torch.Size((dataset.feature_shape.numel(),))


def test_cube_nm_ar_rescue_feature_encodes_only_final_label_bit() -> None:
    dataset_class = get_class("CubeNMARDataset")
    dataset = dataset_class(n_samples=64, seed=0, rescue_feature_std=0.0)
    features, labels = dataset.get_all_data()
    label_idx = labels.argmax(dim=1)
    expected_rescue = (label_idx >= 4).float()

    assert torch.equal(features[:, -1], expected_rescue)


def test_cube_nm_ar_rejects_legacy_dataset_schema(tmp_path: Path) -> None:
    dataset_class = get_class("CubeNMARDataset")
    save_path = tmp_path / "data.bundle"
    save_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "features": torch.zeros((1, 26)),
            "labels": torch.zeros((1, 8)),
            "config": {
                "n_samples": 1,
                "seed": 0,
                "n_contexts": 5,
                "n_safe_contexts": 2,
                "context_feature_std": 0.0,
                "informative_feature_std": 0.05,
                "non_informative_feature_mean": 0.5,
                "non_informative_feature_std": 0.05,
                "rescue_feature_std": 0.01,
                "rescue_feature_cost": 4.0,
                "use_cheap_context_features": True,
                "n_hint_features": 5,
            },
        },
        save_path / "dataset.pt",
    )

    with pytest.raises(KeyError, match="simplified schema"):
        dataset_class.load(save_path)
