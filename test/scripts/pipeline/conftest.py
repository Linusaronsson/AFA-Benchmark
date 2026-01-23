"""Pytest configuration for pipeline tests."""

import subprocess

import pytest
import torch


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command-line options for test configuration."""
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Device to use for testing (cpu, cuda, etc.). Default: cpu",
    )
    parser.addoption(
        "--cores",
        action="store",
        type=int,
        default=1,
        help="Number of cores to use for Snakemake. Default: 1",
    )
    parser.addoption(
        "--no-smoke-test",
        action="store_true",
        default=False,
        help="Run full tests instead of smoke tests. Default: False (runs smoke tests)",
    )
    parser.addoption(
        "--force-rerun",
        action="store_true",
        default=False,
        help="Force rerun of all tests, ignoring existing outputs. Default: False",
    )
    parser.addoption(
        "--methods",
        action="store",
        nargs="*",
        default=None,
        help="Space-separated list of methods to test (e.g., --methods jafa ol odin_model_free). Default: all methods defined in test class",
    )
    parser.addoption(
        "--datasets",
        action="store",
        nargs="*",
        default=None,
        help="Space-separated list of datasets to test (e.g., --datasets synthetic_mnist_without_noise cube_without_noise). Default: all datasets defined in test class",
    )


@pytest.fixture(scope="session")
def device_str(request: pytest.FixtureRequest) -> str:
    """Get the device string from command-line options."""
    result = request.config.getoption("--device")
    assert isinstance(result, str)
    return result


@pytest.fixture(scope="session")
def device(device_str: str) -> torch.device:
    """Get the torch.device object from command-line options."""
    return torch.device(device_str)


@pytest.fixture(scope="session")
def cores(request: pytest.FixtureRequest) -> int:
    """Get the number of cores from command-line options."""
    result = request.config.getoption("--cores")
    assert isinstance(result, int)
    return result


@pytest.fixture(scope="session")
def smoke_test(request: pytest.FixtureRequest) -> bool:
    """Get the smoke test flag from command-line options."""
    result = request.config.getoption("--no-smoke-test")
    assert isinstance(result, bool)
    return not result


@pytest.fixture(scope="session")
def force_rerun(request: pytest.FixtureRequest) -> bool:
    """Get the force rerun flag from command-line options."""
    result = request.config.getoption("--force-rerun")
    assert isinstance(result, bool)
    return result


@pytest.fixture(scope="session")
def methods(request: pytest.FixtureRequest) -> list[str] | None:
    """Get the methods list from command-line options."""
    methods_arg = request.config.getoption("--methods")
    if methods_arg is not None:
        assert isinstance(methods_arg, list)
        return methods_arg
    return None


@pytest.fixture(scope="session")
def datasets(request: pytest.FixtureRequest) -> list[str] | None:
    """Get the datasets list from command-line options."""
    datasets_arg = request.config.getoption("--datasets")
    if datasets_arg is not None:
        assert isinstance(datasets_arg, list)
        return datasets_arg
    return None


@pytest.fixture(scope="session", autouse=True)
def unlock_snakemake() -> None:
    """Unlock Snakemake directory at the start of the test session."""
    subprocess.run(
        ["rm", "-rf", ".snakemake/locks"],  # noqa: S607
        check=False,
    )
