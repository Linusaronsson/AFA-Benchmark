import json
import torch
import pytest
import shutil
import tempfile

from pathlib import Path


from afabench.common.datasets import CubeDataset
from afabench.common.utils import (
    save_artifact,
    load_artifact_metadata,
    get_artifact_path,
    load_dataset,
    parse_artifact_name,
    build_artifact_name,
)


@pytest.fixture
def temp_base_dir():
    """Create temporary base directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create a simple test dataset."""
    dataset = CubeDataset(n_samples=100, seed=42)
    dataset.generate_data()
    return dataset


def test_save_and_load_artifact_metadata(temp_base_dir):
    """Test saving and loading metadata."""
    artifact_dir = temp_base_dir / "test_artifact"
    metadata = {
        "dataset_type": "cube",
        "split_idx": 1,
        "seed": 42,
        "some_param": 3.14,
    }

    # Create temp file for testing
    temp_file = temp_base_dir / "test.pt"
    torch.save({"data": "test"}, temp_file)

    save_artifact(
        artifact_dir=artifact_dir,
        files={"test.pt": temp_file},
        metadata=metadata,
    )

    # Check metadata was saved
    assert (artifact_dir / "metadata.json").exists()
    assert (artifact_dir / "test.pt").exists()

    # Load and verify metadata
    loaded_metadata = load_artifact_metadata(artifact_dir)
    assert loaded_metadata == metadata


def test_save_artifact_with_multiple_files(temp_base_dir):
    """Test saving multiple files."""
    artifact_dir = temp_base_dir / "multi_file_artifact"

    # Create temp files
    file1 = temp_base_dir / "file1.pt"
    file2 = temp_base_dir / "file2.pt"
    torch.save({"a": 1}, file1)
    torch.save({"b": 2}, file2)

    save_artifact(
        artifact_dir=artifact_dir,
        files={"train.pt": file1, "val.pt": file2},
        metadata={"type": "test"},
    )

    assert (artifact_dir / "train.pt").exists()
    assert (artifact_dir / "val.pt").exists()
    assert (artifact_dir / "metadata.json").exists()


def test_get_artifact_path(temp_base_dir):
    """Test artifact path construction."""
    # Dataset
    path = get_artifact_path("dataset", "cube_split_1", temp_base_dir)
    assert path == temp_base_dir / "data" / "cube_split_1"

    # Trained method
    path = get_artifact_path(
        "trained_method",
        "shim2018/cube_split_1_budget_3_seed_42",
        temp_base_dir,
    )
    assert path == temp_base_dir / "shim2018" / "cube_split_1_budget_3_seed_42"

    # Trained classifier
    path = get_artifact_path(
        "trained_classifier", "masked_mlp/cube_split_1", temp_base_dir
    )
    assert (
        path == temp_base_dir / "classifiers" / "masked_mlp" / "cube_split_1"
    )

    # Pretrained model
    path = get_artifact_path(
        "pretrained_model", "shim2018/cube_split_1_seed_42", temp_base_dir
    )
    assert (
        path
        == temp_base_dir / "shim2018" / "pretrained" / "cube_split_1_seed_42"
    )


def test_parse_artifact_name():
    """Test parsing artifact names."""
    method, dataset, params = parse_artifact_name(
        "train_shim2018-cube_split_1-budget_3-seed_42"
    )
    assert method == "shim2018"
    assert dataset == "cube_split_1"
    assert params == "budget_3_seed_42"

    method, dataset, params = parse_artifact_name(
        "aaco-cube_split_2-costparam_0.05-seed_10"
    )
    assert method == "aaco"
    assert dataset == "cube_split_2"
    assert params == "costparam_0.05_seed_10"


def test_build_artifact_name():
    """Test building artifact names."""
    name = build_artifact_name(
        method="shim2018",
        dataset="cube",
        split=1,
        seed=42,
        budget=3,
    )
    assert name == "train_shim2018-cube_split_1-budget_3-seed_42"

    name = build_artifact_name(
        method="aaco",
        dataset="cube",
        split=2,
        seed=10,
        cost_param=0.05,
        prefix="pretrain",
    )
    assert name == "pretrain_aaco-cube_split_2-costparam_0.05-seed_10"


def test_save_and_load_dataset(temp_base_dir, sample_dataset):
    """Test full dataset save/load cycle."""
    # Split dataset
    n = len(sample_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)

    train_dataset = CubeDataset(n_samples=100, seed=42)
    train_dataset.features = sample_dataset.features[:train_size]
    train_dataset.labels = sample_dataset.labels[:train_size]

    val_dataset = CubeDataset(n_samples=100, seed=42)
    val_dataset.features = sample_dataset.features[
        train_size : train_size + val_size
    ]
    val_dataset.labels = sample_dataset.labels[
        train_size : train_size + val_size
    ]

    test_dataset = CubeDataset(n_samples=100, seed=42)
    test_dataset.features = sample_dataset.features[train_size + val_size :]
    test_dataset.labels = sample_dataset.labels[train_size + val_size :]

    # Save to temp files
    temp_dir = temp_base_dir / "temp_dataset"
    temp_dir.mkdir()

    train_path = temp_dir / "train.pt"
    val_path = temp_dir / "val.pt"
    test_path = temp_dir / "test.pt"

    train_dataset.save(train_path)
    val_dataset.save(val_path)
    test_dataset.save(test_path)

    # Save as artifact
    metadata = {
        "dataset_type": "cube",
        "split_idx": 1,
        "seed": 42,
        "n_samples": 100,
    }

    artifact_dir = get_artifact_path("dataset", "cube_split_1", temp_base_dir)
    save_artifact(
        artifact_dir=artifact_dir,
        files={
            "train.pt": train_path,
            "val.pt": val_path,
            "test.pt": test_path,
        },
        metadata=metadata,
    )

    # Load back
    loaded_train, loaded_val, loaded_test, loaded_metadata = load_dataset(
        "cube_split_1", temp_base_dir
    )

    # Verify
    assert torch.allclose(loaded_train.features, train_dataset.features)
    assert torch.allclose(loaded_train.labels, train_dataset.labels)
    assert torch.allclose(loaded_val.features, val_dataset.features)
    assert torch.allclose(loaded_test.features, test_dataset.features)
    assert loaded_metadata == metadata


def test_load_dataset_with_wandb_style_name(temp_base_dir, sample_dataset):
    """Test backward compatibility with wandb-style names."""
    # First save a dataset
    train_path = temp_base_dir / "train_temp.pt"
    val_path = temp_base_dir / "val_temp.pt"
    test_path = temp_base_dir / "test_temp.pt"

    sample_dataset.save(train_path)
    sample_dataset.save(val_path)
    sample_dataset.save(test_path)

    artifact_dir = get_artifact_path("dataset", "cube_split_1", temp_base_dir)
    save_artifact(
        artifact_dir=artifact_dir,
        files={
            "train.pt": train_path,
            "val.pt": val_path,
            "test.pt": test_path,
        },
        metadata={"dataset_type": "cube", "seed": 42},
    )

    # Load with old wandb-style name
    from afabench.common.utils import load_dataset_artifact

    train, val, test, meta = load_dataset_artifact(
        "afa-team/afa-benchmark/cube_split_1:latest", temp_base_dir
    )

    assert train is not None
    assert meta["dataset_type"] == "cube"


def test_missing_metadata_raises_error(temp_base_dir):
    """Test that missing metadata raises error."""
    artifact_dir = temp_base_dir / "no_metadata"
    artifact_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="No metadata.json"):
        load_artifact_metadata(artifact_dir)


def test_missing_dataset_files_raises_error(temp_base_dir):
    """Test that missing dataset files raises error."""
    artifact_dir = get_artifact_path("dataset", "incomplete", temp_base_dir)
    artifact_dir.mkdir(parents=True)

    # Save metadata but no files
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump({"dataset_type": "cube"}, f)

    with pytest.raises(FileNotFoundError, match="missing"):
        load_dataset("incomplete", temp_base_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
