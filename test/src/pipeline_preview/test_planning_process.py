"""The preview plans in a child process it can cancel (#158)."""

import os
import time
from pathlib import Path

from afabench.pipeline_plan.planning import PipelinePlan
from afabench.pipeline_preview.planning_process import PlanningProcess

ARGUMENTS = ["--profile", "workflow/profiles/pipeline/all", "all"]


def test_the_childs_plan_and_diagnostics_are_returned(
    planning_interpreter: list[str], plan: PipelinePlan
) -> None:
    process = PlanningProcess(ARGUMENTS, interpreter=planning_interpreter)
    process.start()

    result = process.wait()

    assert result.plan == plan
    assert "Building DAG of jobs..." in result.diagnostics


def test_a_failed_planning_has_no_plan_and_keeps_the_diagnostics(
    failing_interpreter: list[str],
) -> None:
    process = PlanningProcess(ARGUMENTS, interpreter=failing_interpreter)
    process.start()

    result = process.wait()

    assert result.plan is None
    assert "MissingInputException: cube/0/train.bundle" in result.diagnostics
    assert "exited with 1" in result.diagnostics


def test_cancelling_ends_the_child(
    hanging_interpreter: tuple[list[str], Path],
) -> None:
    interpreter, pid_file = hanging_interpreter
    process = PlanningProcess(ARGUMENTS, interpreter=interpreter)
    process.start()
    while not pid_file.exists():
        time.sleep(0.05)

    process.cancel()

    assert process.wait().plan is None
    assert not alive(int(pid_file.read_text()))


def alive(pid: int) -> bool:
    """Whether the process exists; a reaped child does not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
