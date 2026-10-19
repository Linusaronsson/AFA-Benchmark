from pathlib import Path

import pytest

from afabench.core.checkout import (
    AFABENCH_CHECKOUT,
    find_checkout,
    require_checkout,
)

REPO_ROOT = Path(__file__).parents[3]


def _package(root: Path, project_name: str) -> Path:
    package = root / "src" / "afabench"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n'
    )
    return package


def test_this_checkout_is_found() -> None:
    assert REPO_ROOT.resolve() == AFABENCH_CHECKOUT


def test_a_checkout_is_found_from_its_package(tmp_path: Path) -> None:
    package = _package(tmp_path, "afabench")

    assert find_checkout(package) == tmp_path


def test_an_installed_package_has_no_checkout(tmp_path: Path) -> None:
    # A wheel installed into a venv inside the user's own repository
    package = tmp_path / ".venv/lib/python3.12/site-packages/afabench"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "theirs"\n')

    assert find_checkout(package) is None


def test_another_projects_src_layout_is_no_checkout(tmp_path: Path) -> None:
    package = _package(tmp_path, "theirs")

    assert find_checkout(package) is None


def test_require_checkout_names_what_needs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("afabench.core.checkout.AFABENCH_CHECKOUT", None)

    with pytest.raises(RuntimeError, match="dataset keys"):
        require_checkout("dataset keys")
