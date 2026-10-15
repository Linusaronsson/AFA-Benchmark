"""The venvs a site profile names, and the check the launcher makes of them."""

from pathlib import Path

import pytest
import yaml

from workflow.venvs.submit_builds import VENV_NAME, lock_record, venv_problem

REPO_ROOT = Path(__file__).parents[2]
SITES = REPO_ROOT / "workflow/profiles/site"
LOCK = "# the locked dependencies\n"


def sites_naming_venvs() -> list[Path]:
    return [
        site.parent
        for site in sorted(SITES.rglob("site.yaml"))
        if any(
            "venv" in allocation
            for allocation in yaml.safe_load(site.read_text())[
                "execution_site"
            ].values()
        )
    ]


@pytest.mark.parametrize(
    "site", sites_naming_venvs(), ids=lambda path: path.name
)
def test_jobs_start_the_venvs_the_site_names(site: Path) -> None:
    # Each job's Snakemake starts from the venv workflow/venvs/bin/python
    # finds by the node's architecture, which must be the venv its
    # allocation names.
    profile = yaml.safe_load((site / "config.yaml").read_text())
    allocations = yaml.safe_load((site / "site.yaml").read_text())[
        "execution_site"
    ]

    assert "software-deployment" not in profile["shared-fs-usage"]
    assert profile["precommand"].startswith(
        "export PATH=$PWD/workflow/venvs/bin:"
    )
    for allocation in allocations.values():
        assert VENV_NAME.fullmatch(allocation["venv"])


def built_venv(checkout: Path, venv: str, lock: str) -> None:
    (checkout / "uv.lock").write_text(LOCK)
    lock_record(checkout / venv).parent.mkdir(parents=True)
    lock_record(checkout / venv).write_text(lock)


def test_a_venv_built_from_the_lock_runs_jobs(tmp_path: Path) -> None:
    built_venv(tmp_path, ".venv-x86_64", LOCK)

    assert venv_problem(tmp_path, ".venv-x86_64") is None


def test_a_venv_built_from_another_lock_is_refused(tmp_path: Path) -> None:
    built_venv(tmp_path, ".venv-x86_64", "# an older lock\n")

    assert "another uv.lock" in str(venv_problem(tmp_path, ".venv-x86_64"))


def test_a_venv_whose_build_did_not_finish_is_refused(tmp_path: Path) -> None:
    # build.sbatch records the lock only once the sync succeeded.
    built_venv(tmp_path, ".venv-x86_64", LOCK)
    lock_record(tmp_path / ".venv-x86_64").unlink()

    assert "No venv built" in str(venv_problem(tmp_path, ".venv-x86_64"))
