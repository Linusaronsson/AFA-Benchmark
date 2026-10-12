"""
Keep the terminal to the resolved configuration, job counts and progress.

Snakemake prints several messages for every job, thousands of lines for the
benchmark: its rule block, shell command, submission and completion in a
real run, its rule block and shell command in a dry run. `quiet: [rules]` in
a profile would hide job errors too, so the per-job messages are filtered
here instead, from the terminal only: a real run's log file under
.snakemake/log keeps them, and errors and progress still reach the terminal.
`--config list_jobs=true` keeps them in the terminal too.

Each job's start stays as one line naming its rule and wildcards: its
submission, with its SLURM job ID, or, for a local job, its start. So does
its completion, which Snakemake's own message gives only as a job id. A job
whose script starts a W&B run has its run's link at the end of both lines
(workflow/src/wandb_runs.py). The scheduler's "Select jobs to execute..."
before each selection goes.

A dry run logs its job stats before the job listing and again after it;
without the listing in between, the repeat is filtered too.
"""

import copy
import logging
import re

from snakemake.logging import LogEvent, timestamp
from wandb_runs import WandbRuns, job_name

# A job's rule block (rule, wildcards, resources, reason), shell command,
# start and completion.
JOB_EVENTS = frozenset(
    {
        LogEvent.JOB_INFO,
        LogEvent.GROUP_INFO,
        LogEvent.SHELLCMD,
        LogEvent.JOB_STARTED,
    }
)
# The SLURM executor plugin's report of each submission
SLURM_SUBMISSION = re.compile(
    r"^Job (?P<job_id>\d+) has been submitted with SLURM jobid (?P<slurm_job_id>\S+) "
)
# The scheduler's plain message before each job selection
HIDDEN_MESSAGES = ("Select jobs to execute...",)


class JobMessageFilter(logging.Filter):
    """Drop per-job messages and repeated job stats; name started and finished jobs."""

    def __init__(self, wandb_runs: WandbRuns | None = None) -> None:
        super().__init__()
        self.wandb_runs: WandbRuns | None = wandb_runs
        self.shown_stats: set[str] = set()
        # Each started job's rule and wildcards, by job id
        self.jobs: dict[int, tuple[str, dict[str, str]]] = {}
        # Each started job's W&B run link, by job id
        self.links: dict[int, str] = {}

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        event = getattr(record, "event", None)
        if event == LogEvent.JOB_INFO:
            self.jobs[record.jobid] = (
                record.rule_name,
                dict(record.wildcards),
            )
            # A dry run marks no job local; it starts none.
            if record.local:
                return self._started(record, record.jobid, slurm_job_id=None)
        if event in JOB_EVENTS:
            return False
        if event == LogEvent.JOB_FINISHED:
            return self._finished(record)
        if event == LogEvent.RUN_INFO:
            stats = record.getMessage()
            if stats in self.shown_stats:
                return False
            self.shown_stats.add(stats)
        message = record.getMessage()
        submission = SLURM_SUBMISSION.match(message)
        if submission is not None:
            return self._started(
                record,
                int(submission["job_id"]),
                slurm_job_id=submission["slurm_job_id"],
            )
        return not any(hidden in message for hidden in HIDDEN_MESSAGES)

    def _started(
        self, record: logging.LogRecord, job_id: int, slurm_job_id: str | None
    ) -> bool | logging.LogRecord:
        job = self.jobs.get(job_id)
        if job is None:
            return False
        rule, wildcards = job
        link = (
            None
            if self.wandb_runs is None
            else self.wandb_runs.link(rule, wildcards, slurm_job_id)
        )
        if link is not None:
            self.links[job_id] = link
        name = job_name(rule, wildcards)
        if slurm_job_id is None:
            return _plain(record, f"Started {name}{_link(link)}")
        return _plain(
            record,
            f"Submitted {name} as SLURM job {slurm_job_id}{_link(link)}",
        )

    def _finished(self, record: logging.LogRecord) -> logging.LogRecord:
        job = self.jobs.pop(record.job_id, None)
        if job is None:
            return record
        link = self.links.pop(record.job_id, None)
        return _plain(record, f"Finished {job_name(*job)}{_link(link)}")


def _link(link: str | None) -> str:
    return "" if link is None else f": {link}"


def _plain(record: logging.LogRecord, message: str) -> logging.LogRecord:
    """
    Return a copy of `record` holding `message` after the time.

    A copy, so the log file's handler keeps Snakemake's message. As a plain
    message, it is formatted without a line of its own for the timestamp.
    """
    plain = copy.copy(record)
    plain.msg = f"{timestamp()} {message}"
    plain.args = ()
    plain.event = None
    return plain


def hide_job_messages(
    logger: logging.Logger,
    *,
    list_jobs: bool,
    wandb_runs: WandbRuns | None = None,
) -> None:
    """Filter per-job messages from `logger`'s terminal output."""
    if list_jobs:
        return
    for handler in logger.handlers:
        # Snakemake attaches a real run's log file before parsing.
        if isinstance(handler, logging.FileHandler):
            continue
        if not any(
            isinstance(existing, JobMessageFilter)
            for existing in handler.filters
        ):
            handler.addFilter(JobMessageFilter(wandb_runs))
