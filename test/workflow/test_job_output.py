"""
The terminal shows the resolved configuration, job counts and progress.

It leaves out the messages of every job, in a dry run and a real run, unless
`list_jobs=true`. Errors still reach it, and a real run's log file keeps
everything.
"""

import importlib.util
import io
import logging
from pathlib import Path
from types import ModuleType

import pytest

from test.workflow.submission_harness import REPO_ROOT, WorkflowHarness

# Every job's rule block, shell command, SLURM submission and completion;
# none of them is part of a job error.
JOB_MESSAGES = [
    "    reason: ",
    "Shell command:",
    "submitted with SLURM jobid",
    "Finished jobid",
]


def job_messages(output: str) -> list[str]:
    return [message for message in JOB_MESSAGES if message in output]


@pytest.mark.parametrize("list_jobs", [False, None])
def test_a_dry_run_prints_the_configuration_and_job_counts_only(
    tmp_path: Path, list_jobs: bool | None
) -> None:
    workflow = WorkflowHarness(tmp_path)
    if list_jobs is None:
        del workflow.config["list_jobs"]
    else:
        workflow.config["list_jobs"] = list_jobs

    result = workflow.run("--dry-run", target="all")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Resolved workflow configuration:" in output
    assert "Job stats:" in output
    assert "\nrule " not in output
    assert "Shell command:" not in output


def test_list_jobs_prints_every_job(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = True

    result = workflow.run("--dry-run", target="all")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "\nrule train_method:" in output
    assert "Shell command:" in output


def test_a_run_prints_progress_and_logs_every_job(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = False

    result = workflow.run()

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Job stats:" in output
    assert "steps (100%) done" in output
    assert job_messages(output) == []
    [log] = (tmp_path / ".snakemake/log").glob("*.snakemake.log")
    assert "    reason: " in log.read_text()


# A real SLURM run waits minutes on status checks, so its submission
# message is logged here, as the SLURM executor plugin words it.
def test_slurm_submissions_reach_the_log_file_only(tmp_path: Path) -> None:
    job_output = _load_job_output_module()
    logger = logging.getLogger("test_slurm_submissions")
    terminal = io.StringIO()
    logger.addHandler(logging.StreamHandler(terminal))
    logger.addHandler(logging.FileHandler(tmp_path / "run.log"))
    logger.setLevel(logging.INFO)

    job_output.hide_job_messages(logger, list_jobs=False)
    logger.info(
        "Job 6 has been submitted with SLURM jobid 123 (log: slurm-123.log)."
    )
    logger.info("1 of 5 steps (20%) done")

    assert terminal.getvalue() == "1 of 5 steps (20%) done\n"
    assert (
        "submitted with SLURM jobid 123" in (tmp_path / "run.log").read_text()
    )


def test_a_failed_job_still_prints_its_error(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = False
    (tmp_path / "scripts/train_method/alpha.py").write_text(
        "import sys\nsys.exit(3)\n"
    )

    result = workflow.run()

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Error in rule train_method:" in output
    assert job_messages(output) == []


def _load_job_output_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "workflow_job_output", REPO_ROOT / "workflow/src/job_output.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
