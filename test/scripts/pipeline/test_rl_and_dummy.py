"""
Test the complete RL and dummy methods pipeline using Snakemake.

This test suite validates the end-to-end pipeline for training and evaluating
RL-based and dummy methods on various datasets. The tests are organized as a
dependency chain:

    test_generate_dataset[dataset] → test_pretrain_model[method-dataset] →
    test_train_method[method-dataset] → test_eval_method[method-dataset]

Each test uses Snakemake to run the actual pipeline stages, ensuring that the
workflow infrastructure is tested alongside the method implementations.

Test Structure:
    - test_generate_dataset[dataset]: Generates each dataset (parametrized)
    - test_pretrain_model[method-dataset]: Pretrains each method-dataset combo (parametrized)
    - test_train_method[method-dataset]: Trains each method-dataset combo (parametrized)
    - test_eval_method[method-dataset]: Evaluates each method-dataset combo (parametrized)

Configuration:
    - Modify METHODS to add/remove methods to test
    - Modify DATASETS to add/remove datasets to test
    - Tests use smoke_test=true and device=cpu for fast execution

Usage:
    # Run all tests (will take a long time)
    pytest test/scripts/pipeline/test_rl_and_dummy.py -v

    # Run tests for a specific method
    pytest test/scripts/pipeline/test_rl_and_dummy.py -k "ol" -v

    # Run tests for a specific dataset
    pytest test/scripts/pipeline/test_rl_and_dummy.py -k "cube_without_noise" -v

    # Run complete pipeline for one method-dataset combination
    pytest test/scripts/pipeline/test_rl_and_dummy.py -k "ol-cube_without_noise" -v

Dependencies:
    - pytest-dependency plugin (installed)
    - Snakemake workflow files in extra/workflow/snakefiles/orchestration/
    - Configuration files in extra/workflow/conf/
"""

import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
from pytest_dependency import depends


@pytest.fixture(scope="session", autouse=True)
def unlock_snakemake():
    """Unlock Snakemake directory at the start of the test session."""
    subprocess.run(
        ["rm", "-rf", ".snakemake/locks"],
        check=False,
    )


class TestRLAndDummyPipeline:
    """Test the complete RL and dummy methods pipeline using Snakemake."""

    # Configuration
    SNAKEFILE: ClassVar[str] = (
        "extra/workflow/snakefiles/orchestration/rl_and_dummy.smk"
    )
    CONFIG_FILES: ClassVar[list[str]] = [
        "extra/workflow/conf/hard_budgets_single.yaml",
        "extra/workflow/conf/methods.yaml",
        "extra/workflow/conf/method_options.yaml",
        "extra/workflow/conf/soft_budget_params_single.yaml",
        "extra/workflow/conf/unmaskers.yaml",
    ]

    # Test methods and datasets
    METHODS: ClassVar[list[str]] = ["odin_model_free", "jafa", "ol"]
    DATASETS: ClassVar[list[str]] = [
        "cube_without_noise",
        "synthetic_mnist_without_noise",
    ]

    @pytest.fixture(scope="class")
    def output_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Create a temporary output directory for the pipeline."""
        return tmp_path_factory.mktemp("pipeline_output")

    @pytest.mark.dependency
    def test_generate_dataset(
        self, dataset: str, output_dir: Path, request: pytest.FixtureRequest
    ) -> None:
        """Generate a single dataset using Snakemake."""
        result = subprocess.run(
            [
                "uv",
                "run",
                "snakemake",
                "-s",
                "extra/workflow/snakefiles/orchestration/dataset_generation.smk",
                "--config",
                f"datasets=['{dataset}']",
                "dataset_instance_indices=[0]",
                "--cores",
                "1",
                "--forceall",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"Dataset generation failed for {dataset}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify dataset was created
        dataset_dir = Path(f"extra/output/datasets/{dataset}")
        assert dataset_dir.exists(), f"Dataset {dataset} directory not created"
        assert (dataset_dir / "0").exists(), (
            f"Dataset {dataset} instance 0 not created"
        )

    @pytest.mark.dependency
    def test_pretrain_model(
        self,
        method: str,
        dataset: str,
        output_dir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Pretrain a single method on a single dataset using Snakemake."""
        depends(
            request,
            [f"TestRLAndDummyPipeline::test_generate_dataset[{dataset}]"],
        )

        cmd = [
            "uv",
            "run",
            "snakemake",
            "-s",
            self.SNAKEFILE,
            "all_pretrain_model",
            "--configfile",
            *self.CONFIG_FILES,
            "--config",
            f"methods=['{method}']",
            f"datasets=['{dataset}']",
            "dataset_instance_indices=[0]",
            "smoke_test=true",
            "use_wandb=false",
            "device=cpu",
            "--cores",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"Pretraining failed for {method} on {dataset}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify pretrained model was created
        pretrained_dir = (
            Path("extra/output/pretrained_models")
            / method
            / f"dataset-{dataset}+instance_idx-0"
        )
        assert pretrained_dir.exists(), (
            f"Pretrained model directory for {method} on {dataset} not created"
        )

    @pytest.mark.dependency
    def test_train_method(
        self,
        method: str,
        dataset: str,
        output_dir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Train a single method on a single dataset using Snakemake."""
        depends(
            request,
            [
                f"TestRLAndDummyPipeline::test_pretrain_model[{method}-{dataset}]"
            ],
        )

        cmd = [
            "uv",
            "run",
            "snakemake",
            "-s",
            self.SNAKEFILE,
            "all_train_method",
            "--configfile",
            *self.CONFIG_FILES,
            "--config",
            f"methods=['{method}']",
            f"datasets=['{dataset}']",
            "dataset_instance_indices=[0]",
            "smoke_test=true",
            "use_wandb=false",
            "device=cpu",
            "--cores",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"Training failed for {method} on {dataset}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify trained method was created
        trained_dir = (
            Path("extra/output/trained_methods")
            / method
            / f"dataset-{dataset}+instance_idx-0"
        )
        assert trained_dir.exists(), (
            f"Trained method directory for {method} on {dataset} not created"
        )

    @pytest.mark.dependency
    def test_eval_method(
        self,
        method: str,
        dataset: str,
        output_dir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """Evaluate a single method on a single dataset using Snakemake."""
        depends(
            request,
            [f"TestRLAndDummyPipeline::test_train_method[{method}-{dataset}]"],
        )

        cmd = [
            "uv",
            "run",
            "snakemake",
            "-s",
            self.SNAKEFILE,
            "all_eval_method",
            "--configfile",
            *self.CONFIG_FILES,
            "--config",
            f"methods=['{method}']",
            f"datasets=['{dataset}']",
            "dataset_instance_indices=[0]",
            "smoke_test=true",
            "use_wandb=false",
            "device=cpu",
            "--cores",
            "1",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, (
            f"Evaluation failed for {method} on {dataset}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify evaluation results were created
        eval_dir = (
            Path("extra/output/eval_results")
            / method
            / f"dataset-{dataset}+instance_idx-0"
        )
        assert eval_dir.exists(), (
            f"Evaluation results directory for {method} on {dataset} not created"
        )


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests with methods and datasets from the class."""
    if (
        metafunc.cls
        and hasattr(metafunc.cls, "METHODS")
        and hasattr(metafunc.cls, "DATASETS")
    ):
        # Parametrize dataset generation tests over datasets only
        if (
            "dataset" in metafunc.fixturenames
            and "method" not in metafunc.fixturenames
        ):
            metafunc.parametrize("dataset", metafunc.cls.DATASETS)
        # Parametrize method tests over both method and dataset (cartesian product)
        elif (
            "method" in metafunc.fixturenames
            and "dataset" in metafunc.fixturenames
        ):
            metafunc.parametrize(
                "method,dataset",
                [
                    (m, d)
                    for m in metafunc.cls.METHODS
                    for d in metafunc.cls.DATASETS
                ],
            )
