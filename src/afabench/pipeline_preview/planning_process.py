"""
Plan an invocation in a child process, keeping Snakemake out of the app.

The child is `python -m afabench.pipeline_plan`, so Snakemake's global
state and everything it prints stay in the child; its output is kept as
the planning diagnostics. The child can be cancelled when the user quits
before it finishes, and is not left behind.
"""

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import final

from afabench.pipeline_plan.plan_json import PlanJsonError, load_plan
from afabench.pipeline_plan.planning import PipelinePlan

PLANNER_MODULE = "afabench.pipeline_plan"
# How long a terminated child gets to exit before it is killed.
TERMINATE_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, kw_only=True)
class PlanningResult:
    """What planning in the child ended with."""

    # None when planning failed or was cancelled.
    plan: PipelinePlan | None
    # Everything the child printed, with how it ended.
    diagnostics: str


@final
class PlanningProcess:
    """A child process planning `snakemake <snakemake_arguments>`."""

    def __init__(
        self,
        snakemake_arguments: Sequence[str],
        *,
        interpreter: Sequence[str] = (sys.executable,),
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._snakemake_arguments: list[str] = list(snakemake_arguments)
        self._interpreter: list[str] = list(interpreter)
        self._cwd: Path | None = cwd
        self._env: dict[str, str] | None = None if env is None else dict(env)
        self._child: subprocess.Popen[str] | None = None
        self._scratch: Path | None = None
        self._cancelled: bool = False

    @property
    def plan_path(self) -> Path:
        assert self._scratch is not None, "not started"
        return self._scratch / "plan.json"

    def start(self) -> None:
        """Start the child; `wait` returns what it planned."""
        self._scratch = Path(tempfile.mkdtemp(prefix="afabench-preview-"))
        self._child = subprocess.Popen(
            [
                *self._interpreter,
                "-m",
                PLANNER_MODULE,
                "--output",
                str(self.plan_path),
                "--",
                *self._snakemake_arguments,
            ],
            cwd=self._cwd,
            env=self._env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait(self) -> PlanningResult:
        """Block until the child ends; return its plan or diagnostics."""
        assert self._child is not None, "not started"
        output, _ = self._child.communicate()
        try:
            if self._cancelled:
                ending = "Planning was cancelled."
            elif self._child.returncode != 0:
                ending = f"The planner exited with {self._child.returncode}."
            else:
                return self._planned(output)
            return PlanningResult(plan=None, diagnostics=output + ending)
        finally:
            self._remove_scratch()

    def _planned(self, output: str) -> PlanningResult:
        try:
            plan = load_plan(self.plan_path.read_text())
        except (OSError, PlanJsonError) as error:
            return PlanningResult(
                plan=None,
                diagnostics=f"{output}The planner wrote no plan: {error}",
            )
        return PlanningResult(plan=plan, diagnostics=output)

    def cancel(self) -> None:
        """End the child if it still runs; `wait` then returns no plan."""
        if self._child is None or self._child.poll() is not None:
            return
        self._cancelled = True
        self._child.terminate()
        try:
            self._child.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._child.kill()

    def _remove_scratch(self) -> None:
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
