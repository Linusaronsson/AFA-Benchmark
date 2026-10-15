"""Running jobs in the venv of their node's architecture (docs/adr/0015)."""

import platform
import re
import sys
from pathlib import Path

import pytest

from test.workflow.submission_harness import SITE, WorkflowHarness


def shell_commands(output: str) -> list[str]:
    return re.findall(r"Shell command: \n\s*(.*)", output)


@pytest.mark.parametrize(
    "site",
    [
        "arrhenius",
        "vera",
        "alvis",
        "examples/mixed-gres",
        "examples/mixed-gpus",
    ],
)
def test_every_site_runs_scripts_with_the_python_on_path(
    tmp_path: Path, site: str
) -> None:
    workflow = WorkflowHarness(tmp_path)

    result = workflow.run(
        "--dry-run",
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site" / site),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = shell_commands(result.stdout)
    assert commands
    assert all(command.startswith("python ") for command in commands)


# Each job starts a nested Snakemake in the fake sbatch, so this takes ~10 s.
def test_arrhenius_jobs_run_in_the_venv_of_their_nodes_architecture(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}
    # The venv build.sbatch would build for this node, logging each start
    venv = tmp_path / f".venv-{platform.machine()}"
    log = tmp_path / "venv.log"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").write_text(
        f'#!/bin/sh\necho "$@" >> {log}\nexec {sys.executable} "$@"\n'
    )
    (venv / "bin/python").chmod(0o755)

    result = workflow.submit_first_wave(
        2,
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/arrhenius"),
        target="all_train_methods",
    )

    submissions = workflow.submissions()
    assert len(submissions) == 2, result.stdout
    for args in submissions:
        wrap = next(arg for arg in args if arg.startswith("--wrap="))
        # Each job's Snakemake starts from PATH, not the login node's Python.
        assert f"export PATH={tmp_path}/workflow/venvs/bin:" in wrap
        assert " && python -m snakemake " in wrap
    starts = log.read_text().splitlines()
    assert sum(start.startswith("-m snakemake ") for start in starts) == 2
    # The job record wrapper, which runs the script, starts there too.
    assert sum("-m afabench.core.job_record" in start for start in starts) == 2
    devices = sorted(args["device"] for _, args in workflow.script_arguments())
    assert devices == ["cpu", "cuda"]


def test_an_unknown_allocation_key_fails_before_submission(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution_site"] = {
        **SITE,
        "cpu": {**SITE["cpu"], "image": "containers/afabench-x86_64.sif"},
    }

    result = workflow.run("--executor", "slurm")

    assert result.returncode != 0
    assert "Unknown execution_site.cpu keys: ['image']" in (
        result.stdout + result.stderr
    )
    assert workflow.submissions() == []
