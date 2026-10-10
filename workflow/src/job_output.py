"""
Keep the terminal to the resolved configuration, job counts and progress.

Snakemake prints several messages for every job, thousands of lines for the
benchmark: its rule block, shell command, submission and completion in a
real run, its rule block and shell command in a dry run. `quiet: [rules]` in
a profile would hide job errors too, so the per-job messages are filtered
here instead, from the terminal only: a real run's log file under
.snakemake/log keeps them, and errors and progress still reach the terminal.
`--config list_jobs=true` keeps them in the terminal too.

Each job's completion stays, as one line naming its rule and wildcards, which
Snakemake's own completion message, a job id, leaves out. The scheduler's
"Select jobs to execute..." before each selection goes.

A dry run logs its job stats before the job listing and again after it;
without the listing in between, the repeat is filtered too.
"""

import copy
import logging

from snakemake.logging import LogEvent, timestamp

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
# Plain messages: the SLURM executor plugin's report of each submission, and
# the scheduler's before each job selection.
HIDDEN_MESSAGES = (
    "has been submitted with SLURM jobid",
    "Select jobs to execute...",
)


class JobMessageFilter(logging.Filter):
    """Drop per-job messages and repeated job stats; name finished jobs."""

    def __init__(self) -> None:
        super().__init__()
        self.shown_stats: set[str] = set()
        # Each started job's rule and wildcards, by job id
        self.job_names: dict[int, str] = {}

    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        event = getattr(record, "event", None)
        if event == LogEvent.JOB_INFO:
            self.job_names[record.jobid] = _job_name(record)
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
        return not any(hidden in message for hidden in HIDDEN_MESSAGES)

    def _finished(self, record: logging.LogRecord) -> logging.LogRecord:
        name = self.job_names.pop(record.job_id, None)
        if name is None:
            return record
        # A copy, so the log file's handler keeps Snakemake's message. As a
        # plain message, it is formatted without a line of its own for the
        # timestamp.
        finished = copy.copy(record)
        finished.msg = f"{timestamp()} Finished {name}"
        finished.args = ()
        finished.event = None
        return finished


def _job_name(record: logging.LogRecord) -> str:
    wildcards = ", ".join(
        f"{name}={value}" for name, value in record.wildcards.items()
    )
    return (
        f"{record.rule_name} ({wildcards})" if wildcards else record.rule_name
    )


def hide_job_messages(logger: logging.Logger, *, list_jobs: bool) -> None:
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
            handler.addFilter(JobMessageFilter())
