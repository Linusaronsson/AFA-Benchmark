import subprocess
from pathlib import Path
from typing import ClassVar, cast

import pytest
from pytest_dependency import depends


def check_output_exists(path: Path) -> bool:
    """Check if a required output path exists."""
    return path.exists()


@pytest.mark.pipeline
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
        "extra/workflow/conf/pretrain_mapping.yaml",
        "extra/workflow/conf/soft_budget_params_single.yaml",
        "extra/workflow/conf/unmaskers.yaml",
    ]

    # Test methods and datasets
    METHODS: ClassVar[list[str]] = ["odin_model_free", "jafa", "ol"]
    DATASETS: ClassVar[list[str]] = [
        "cube_without_noise",
        "afa_context_without_noise",
        "synthetic_mnist_without_noise",
    ]

    @pytest.fixture(scope="class")
    def output_dir(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        """Create a temporary output directory for the pipeline."""
        return tmp_path_factory.mktemp("pipeline_output")

    @pytest.mark.dependency(scope="session")
    def test_generate_dataset(
        self,
        dataset: str,
        cores: int,
    ) -> None:
        """Generate a single dataset using Snakemake."""
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "snakemake",
                "-s",
                "extra/workflow/snakefiles/orchestration/dataset_generation.smk",
                "--config",
                f"datasets=['{dataset}']",
                "dataset_instance_indices=[0]",
                "--cores",
                str(cores),
                "--forceall",
                "--rerun-incomplete",
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

    @pytest.mark.dependency(scope="session")
    def test_pretrain_model(
        self,
        method: str,
        dataset: str,
        device_str: str,
        cores: int,
        smoke_test: bool,
        force_rerun: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Pretrain a single method on a single dataset using Snakemake."""
        # Check if dependency outputs exist (unless force_rerun is set)
        dataset_dir = Path(f"extra/output/datasets/{dataset}/0")
        if force_rerun or not check_output_exists(dataset_dir):
            # Enforce dependency if force_rerun or output doesn't exist
            depends(
                request,
                [f"TestRLAndDummyPipeline::test_generate_dataset[{dataset}]"],
                scope="session",
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
            f"smoke_test={'true' if smoke_test else 'false'}",
            "use_wandb=false",
            f"device={device_str}",
            "--cores",
            str(cores),
            "--rerun-incomplete",
        ]

        result = subprocess.run(  # noqa: S603
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

    @pytest.mark.dependency(scope="session")
    def test_train_method(
        self,
        method: str,
        dataset: str,
        device_str: str,
        cores: int,
        smoke_test: bool,
        force_rerun: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Train a single method on a single dataset using Snakemake."""
        # Check if dependency outputs exist (unless force_rerun is set)
        pretrained_dir = (
            Path("extra/output/pretrained_models")
            / method
            / f"dataset-{dataset}+instance_idx-0"
        )
        if force_rerun or not check_output_exists(pretrained_dir):
            # Enforce dependency if force_rerun or output doesn't exist
            depends(
                request,
                [
                    f"TestRLAndDummyPipeline::test_pretrain_model[{method}-{dataset}]"
                ],
                scope="session",
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
            f"smoke_test={'true' if smoke_test else 'false'}",
            "use_wandb=false",
            f"device={device_str}",
            "--cores",
            str(cores),
            "--rerun-incomplete",
        ]

        result = subprocess.run(  # noqa: S603
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

    @pytest.mark.dependency(scope="session")
    def test_eval_method(
        self,
        method: str,
        dataset: str,
        device_str: str,
        cores: int,
        smoke_test: bool,
        force_rerun: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Evaluate a single method on a single dataset using Snakemake."""
        # Check if dependency outputs exist (unless force_rerun is set)
        trained_dir = (
            Path("extra/output/trained_methods")
            / method
            / f"dataset-{dataset}+instance_idx-0"
        )
        if force_rerun or not check_output_exists(trained_dir):
            # Enforce dependency if force_rerun or output doesn't exist
            depends(
                request,
                [
                    f"TestRLAndDummyPipeline::test_train_method[{method}-{dataset}]"
                ],
                scope="session",
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
            f"smoke_test={'true' if smoke_test else 'false'}",
            "use_wandb=false",
            f"device={device_str}",
            "--cores",
            str(cores),
            "--rerun-incomplete",
        ]

        result = subprocess.run(  # noqa: S603
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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize tests with methods and datasets from the class."""
    if (
        metafunc.cls
        and hasattr(metafunc.cls, "METHODS")
        and hasattr(metafunc.cls, "DATASETS")
    ):
        # Get methods and datasets from command-line options or use defaults
        methods_option = metafunc.config.getoption("--methods")
        datasets_option = metafunc.config.getoption("--datasets")

        methods: list[str]
        datasets: list[str]

        if methods_option and isinstance(methods_option, list):
            methods = methods_option
        else:
            methods = cast("list[str]", metafunc.cls.METHODS)

        if datasets_option and isinstance(datasets_option, list):
            datasets = datasets_option
        else:
            datasets = cast("list[str]", metafunc.cls.DATASETS)

        # Parametrize dataset generation tests over datasets only
        if (
            "dataset" in metafunc.fixturenames
            and "method" not in metafunc.fixturenames
        ):
            metafunc.parametrize("dataset", datasets)
        # Parametrize method tests over both method and dataset (cartesian product)
        elif (
            "method" in metafunc.fixturenames
            and "dataset" in metafunc.fixturenames
        ):
            metafunc.parametrize(
                "method,dataset",
                [(m, d) for m in methods for d in datasets],
            )
