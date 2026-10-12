"""
The terminal shows the resolved configuration, job counts and progress.

It leaves out the messages of every job, in a dry run and a real run, unless
`list_jobs=true`, but for one line naming each job when it is submitted or
starts, and one when it finishes, both with the job's W&B run link when it
logs to W&B. Errors still reach it, and a real run's log file keeps
everything.
"""

import importlib.util
import io
import logging
import re
from pathlib import Path
from types import ModuleType

import pytest

from test.workflow.submission_harness import REPO_ROOT, SITE, WorkflowHarness

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
    assert output.count("Job stats:") == 1
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


WANDB_ENVIRONMENT = {"WANDB_ENTITY": "lab", "WANDB_PROJECT": "afabench"}
RUN_URL = "https://wandb.ai/lab/afabench/runs/"


def test_a_run_prints_progress_and_logs_every_job(tmp_path: Path) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = False

    result = workflow.run()

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Job stats:" in output
    assert "steps (100%) done" in output
    assert "Select jobs to execute" not in output
    assert job_messages(output) == []
    assert "Started train_method (" in output
    assert "Finished train_method (" in output
    assert "method=alpha" in output
    assert RUN_URL not in output
    [log] = (tmp_path / ".snakemake/log").glob("*.snakemake.log")
    assert "    reason: " in log.read_text()


def test_a_local_job_links_the_wandb_run_its_script_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in WANDB_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = False
    workflow.config["use_wandb"] = True

    result = workflow.run()

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    runs = workflow.wandb_runs()
    assert len(runs) == len(workflow.script_arguments())
    for run_id, name in runs:
        assert re.fullmatch(
            r"(train|eval)_method-(alpha|beta)-cube-[a-z0-9]{8}", run_id
        )
        assert f"Started {name}: {RUN_URL}{run_id}\n" in output
        assert f"Finished {name}: {RUN_URL}{run_id}\n" in output


def test_wandb_logging_needs_the_entity_and_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "afabench")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    workflow = WorkflowHarness(tmp_path)
    workflow.config["use_wandb"] = True

    result = workflow.run("--dry-run")

    assert result.returncode != 0
    assert "set WANDB_ENTITY" in result.stdout + result.stderr


# Each job starts a nested Snakemake in the fake sbatch, so this takes ~10 s.
@pytest.mark.optional
def test_a_slurm_job_links_the_wandb_run_of_its_slurm_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in WANDB_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    workflow = WorkflowHarness(tmp_path)
    workflow.config["list_jobs"] = False
    workflow.config["use_wandb"] = True
    workflow.config["execution_site"] = SITE

    result = workflow.submit_first_wave(
        2, "--executor", "slurm", target="all_train_methods"
    )

    runs = workflow.wandb_runs()
    assert len(runs) == 2, result.stdout
    for run_id, name in runs:
        slurm_job_id = run_id.rsplit("-", 1)[1]
        assert run_id.startswith("train_method-")
        assert (
            f"Submitted {name} as SLURM job {slurm_job_id}: {RUN_URL}{run_id}\n"
            in result.stdout
        )
    assert "has been submitted with SLURM jobid" not in result.stdout


# A real SLURM run waits minutes on status checks, so its job messages are
# logged here, as Snakemake and the SLURM executor plugin word them.
def test_slurm_submissions_name_the_job_and_link_its_wandb_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_output = _load_job_output_module(monkeypatch)
    wandb_runs = _load_wandb_runs_module()
    runs = wandb_runs.WandbRuns(
        wandb_runs.WandbProject(entity="lab", project="afabench"),
        slurm_job_id=None,
    )
    runs.param("train_method", lambda wildcards: wildcards["method"])
    logger, terminal = _logger(tmp_path, "test_slurm_submissions")

    job_output.hide_job_messages(logger, list_jobs=False, wandb_runs=runs)
    for job_id, rule in [(6, "train_method"), (7, "transform_eval_data")]:
        _job_info(logger, job_id, rule, local=False)
    logger.info(
        "Job 6 has been submitted with SLURM jobid 123 (log: slurm-123.log)."
    )
    logger.info(
        "Job 7 has been submitted with SLURM jobid 124 (log: slurm-124.log)."
    )
    logger.info(
        "Done", extra={"event": _log_event().JOB_FINISHED, "job_id": 6}
    )

    # Without the timestamp
    lines = [
        line.split("] ", 1)[1] for line in terminal.getvalue().splitlines()
    ]
    assert lines == [
        "Submitted train_method (method=dime, dataset=cube) as SLURM job 123: "
        f"{RUN_URL}train_method-dime-cube-123",
        "Submitted transform_eval_data (method=dime, dataset=cube) as SLURM job 124",
        "Finished train_method (method=dime, dataset=cube): "
        f"{RUN_URL}train_method-dime-cube-123",
    ]
    assert (
        "submitted with SLURM jobid 123" in (tmp_path / "run.log").read_text()
    )


def test_a_job_in_its_slurm_job_starts_the_run_of_its_slurm_job() -> None:
    wandb_runs = _load_wandb_runs_module()
    runs = wandb_runs.WandbRuns(
        wandb_runs.WandbProject(entity="lab", project="afabench"),
        slurm_job_id="123",
    )
    param = runs.param("eval_method", lambda wildcards: wildcards["method"])

    command = param({"method": "dime", "dataset": "cube/x"}, {})

    assert command == (
        "WANDB_RUN_ID=eval_method-dime-cube_x-123 "
        "WANDB_NAME='eval_method (method=dime, dataset=cube/x)' "
    )


def test_a_job_without_wandb_logging_gets_no_run() -> None:
    wandb_runs = _load_wandb_runs_module()
    runs = wandb_runs.WandbRuns(None, slurm_job_id="123")
    param = runs.param("eval_method", lambda wildcards: wildcards["method"])

    assert param({"method": "dime", "dataset": "cube"}, {}) == ""


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


def _logger(tmp_path: Path, name: str) -> tuple[logging.Logger, io.StringIO]:
    """Return a logger with a terminal and a log file, and its terminal."""
    logger = logging.getLogger(name)
    terminal = io.StringIO()
    logger.addHandler(logging.StreamHandler(terminal))
    logger.addHandler(logging.FileHandler(tmp_path / "run.log"))
    logger.setLevel(logging.INFO)
    return logger, terminal


def _job_info(
    logger: logging.Logger, job_id: int, rule: str, *, local: bool
) -> None:
    logger.info(
        f" Rule: {rule}, Jobid: {job_id}",
        extra={
            "event": _log_event().JOB_INFO,
            "jobid": job_id,
            "rule_name": rule,
            "local": local,
            "wildcards": {"method": "dime", "dataset": "cube"},
        },
    )


def _log_event() -> type:
    from snakemake.logging import LogEvent  # noqa: PLC0415

    return LogEvent


def _load_job_output_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # The module imports its neighbours, as the Snakefile puts them on the path.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "workflow/src"))
    return _load_module("job_output")


def _load_wandb_runs_module() -> ModuleType:
    return _load_module("wandb_runs")


def _load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"workflow_{name}", REPO_ROOT / f"workflow/src/{name}.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
