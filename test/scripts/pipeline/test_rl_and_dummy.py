import subprocess
from pathlib import Path
from typing import ClassVar, cast

import pytest
import yaml
from pytest_dependency import depends


def check_output_exists(path: Path) -> bool:
    """Check if a required output path exists."""
    return path.exists()


def load_method_to_pretrained_model_mapping() -> dict[str, str | None]:
    """
    Load the mapping from methods to pretrained model names.

    Returns a dictionary mapping method names to their pretrained_model_name,
    or None if the method doesn't use a pretrained model.
    """
    method_options_path = Path("extra/workflow/conf/method_options.yaml")
    with method_options_path.open() as f:
        data = yaml.safe_load(f)

    return {
        method: opts.get("pretrained_model_name")
        for method, opts in data["method_options"].items()
    }


def load_methods_from_config() -> list[str]:
    """
    Load the list of methods from the methods config file.

    Returns a list of method names to test.
    """
    methods_path = Path("extra/workflow/conf/methods.yaml")
    with methods_path.open() as f:
        data = yaml.safe_load(f)
    return data["methods"]


def load_datasets_from_config() -> list[str]:
    """
    Load the list of datasets from the datasets config file.

    Returns a list of dataset names to test.
    """
    datasets_path = Path("extra/workflow/conf/datasets_sanity_check.yaml")
    with datasets_path.open() as f:
        data = yaml.safe_load(f)
    return data["datasets"]


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
        "extra/workflow/conf/datasets_sanity_check.yaml",
    ]

    # Test methods and datasets - loaded from config files
    METHODS: ClassVar[list[str]] = load_methods_from_config()
    DATASETS: ClassVar[list[str]] = load_datasets_from_config()

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
        pretrain_name: str,
        dataset: str,
        device_str: str,
        cores: int,
        smoke_test: bool,
        force_rerun: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Pretrain a single pretrain config on a single dataset using Snakemake."""
        # Check if dependency outputs exist (unless force_rerun is set)
        dataset_dir = Path(f"extra/output/datasets/{dataset}/0")
        if not force_rerun and not check_output_exists(dataset_dir):
            # Enforce dependency if force_rerun is not set and output doesn't exist
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
            f"Pretraining failed for {pretrain_name} on {dataset}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify pretrained model was created
        pretrained_dir = (
            Path("extra/output/pretrained_models")
            / pretrain_name
            / f"dataset-{dataset}+instance_idx-0"
        )
        assert pretrained_dir.exists(), (
            f"Pretrained model directory for {pretrain_name} on {dataset} not created"
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
        # Load method to pretrained model mapping
        method_to_pretrain = load_method_to_pretrained_model_mapping()
        pretrained_model_name = method_to_pretrain.get(method)

        # Check if dependency outputs exist (unless force_rerun is set)
        # Only check pretrain dependency if the method uses a pretrained model
        if pretrained_model_name:
            pretrained_dir = (
                Path("extra/output/pretrained_models")
                / pretrained_model_name
                / f"dataset-{dataset}+instance_idx-0"
            )
            if not force_rerun and not check_output_exists(pretrained_dir):
                # Enforce dependency if force_rerun is not set and output doesn't exist
                depends(
                    request,
                    [
                        f"TestRLAndDummyPipeline::test_pretrain_model[{pretrained_model_name}-{dataset}]"
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
        if not force_rerun and not check_output_exists(trained_dir):
            # Enforce dependency if force_rerun is not set and output doesn't exist
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
    """Dynamically parametrize tests with methods, datasets, and pretrain names."""
    if not metafunc.cls:
        return

    # Check if this is our test class with the required class variables
    has_methods = hasattr(metafunc.cls, "METHODS")
    has_datasets = hasattr(metafunc.cls, "DATASETS")

    if not (has_methods and has_datasets):
        return

    # Get command-line options
    methods_option = metafunc.config.getoption("--methods")
    datasets_option = metafunc.config.getoption("--datasets")

    # Determine which values to use (CLI options override class defaults)
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

    # Calculate which pretrain configs are needed based on selected methods
    method_to_pretrain = load_method_to_pretrained_model_mapping()
    pretrain_names = list(
        {
            method_to_pretrain[method]
            for method in methods
            if method in method_to_pretrain
            and method_to_pretrain[method] is not None
        }
    )

    # Parametrize based on which fixtures the test function uses

    # Dataset generation tests: only need dataset
    if (
        "dataset" in metafunc.fixturenames
        and "method" not in metafunc.fixturenames
        and "pretrain_name" not in metafunc.fixturenames
    ):
        metafunc.parametrize("dataset", datasets)

    # Pretrain tests: need pretrain_name and dataset
    elif (
        "pretrain_name" in metafunc.fixturenames
        and "dataset" in metafunc.fixturenames
    ):
        metafunc.parametrize(
            "pretrain_name,dataset",
            [(p, d) for p in pretrain_names for d in datasets],
        )

    # Method tests (train/eval): need method and dataset
    elif (
        "method" in metafunc.fixturenames
        and "dataset" in metafunc.fixturenames
    ):
        metafunc.parametrize(
            "method,dataset",
            [(m, d) for m in methods for d in datasets],
        )
