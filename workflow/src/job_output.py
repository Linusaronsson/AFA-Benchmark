"""
Keep the terminal to the resolved configuration, job counts and progress.

Snakemake prints several messages for every job, thousands of lines for the
benchmark: its rule block, shell command, submission and completion in a
real run, its rule block and shell command in a dry run. `quiet: [rules]` in
a profile would hide job errors too, so the per-job messages are filtered
here instead, from the terminal only: a real run's log file under
.snakemake/log keeps them, and errors and progress still reach the terminal.
`--config list_jobs=true` keeps them in the terminal too.

A dry run logs its job stats before the job listing and again after it;
without the listing in between, the repeat is filtered too.
"""

import logging

from snakemake.logging import LogEvent

# A job's rule block (rule, wildcards, resources, reason), shell command,
# start and completion.
JOB_EVENTS = frozenset(
    {
        LogEvent.JOB_INFO,
        LogEvent.GROUP_INFO,
        LogEvent.SHELLCMD,
        LogEvent.JOB_STARTED,
        LogEvent.JOB_FINISHED,
    }
)
# The SLURM executor plugin reports each submission as a plain message.
SLURM_SUBMISSION = "has been submitted with SLURM jobid"


class JobMessageFilter(logging.Filter):
    """Drop per-job messages and job stats already shown."""

    def __init__(self) -> None:
        super().__init__()
        self.shown_stats: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        event = getattr(record, "event", None)
        if event in JOB_EVENTS:
            return False
        if event == LogEvent.RUN_INFO:
            stats = record.getMessage()
            if stats in self.shown_stats:
                return False
            self.shown_stats.add(stats)
        return SLURM_SUBMISSION not in record.getMessage()


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
