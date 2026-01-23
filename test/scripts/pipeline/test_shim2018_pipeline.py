import subprocess
from pathlib import Path

import pytest

DATASETS = [
    "cube_without_noise",
]


@pytest.fixture
def pipeline_artifacts() -> dict[str, Path]:
    return {}


@pytest.mark.dependency
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_generate_dataset(
    dataset_name: str,
    pipeline_artifacts: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    datasets_dir = tmp_path_factory.mktemp("datasets", numbered=True)
    save_path = datasets_dir / dataset_name

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "scripts/dataset_generation/generate_dataset.py",
            f"dataset={dataset_name}",
            "instance_indices=[0]",
            "seeds=[0]",
            f"save_path={save_path}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    pipeline_artifacts[f"dataset_{dataset_name}"] = save_path


@pytest.mark.dependency(depends=["test_generate_dataset"])
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_shim2018_pretrain(
    dataset_name: str,
    pipeline_artifacts: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    dataset_dir = pipeline_artifacts[f"dataset_{dataset_name}"]
    save_path = (
        tmp_path_factory.mktemp("pretrained_models")
        / f"shim2018_{dataset_name}.bundle"
    )

    result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "scripts/pretrain/shim2018.py",
            f"train_dataset_bundle_path={dataset_dir}/0/train.bundle",
            f"val_dataset_bundle_path={dataset_dir}/0/val.bundle",
            f"save_path={save_path}",
            "device=cpu",
            "seed=1",
            "use_wandb=false",
            "smoke_test=true",
            f"experiment@_global_={dataset_name}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    pipeline_artifacts[f"pretrained_shim2018_{dataset_name}"] = save_path


@pytest.mark.dependency(depends=["test_shim2018_pretrain"])
@pytest.mark.parametrize("dataset_name", DATASETS)
def test_shim2018_train(
    dataset_name: str,
    pipeline_artifacts: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    dataset_dir = pipeline_artifacts[f"dataset_{dataset_name}"]
    pretrained_model_path = pipeline_artifacts[
        f"pretrained_shim2018_{dataset_name}"
    ]
    method_save_path = tmp_path_factory.mktemp("method.bundle")

    # Run training using the shared pretrained model
    train_result = subprocess.run(  # noqa: S603
        [  # noqa: S607
            "uv",
            "run",
            "python",
            "scripts/train/shim2018.py",
            f"train_dataset_bundle_path={dataset_dir}/0/train.bundle",
            f"val_dataset_bundle_path={dataset_dir}/0/val.bundle",
            f"pretrained_model_bundle_path={pretrained_model_path}",
            f"save_path={method_save_path}",
            "components/initializers@initializer=cold",
            "components/unmaskers@unmasker=direct",
            "hard_budget=5",
            "soft_budget_param=0.1",
            "device=cpu",
            "seed=1",
            "use_wandb=false",
            "smoke_test=true",
            f"experiment@_global_={dataset_name}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert train_result.returncode == 0
