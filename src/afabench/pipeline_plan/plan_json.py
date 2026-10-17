"""
Encode a pipeline plan as JSON, to pass it between processes.

The interactive preview plans in a child process (`python -m
afabench.pipeline_plan`) that writes its plan this way, so Snakemake's
global state and diagnostics stay out of the terminal application.
"""

import json
from dataclasses import asdict, fields
from typing import Any

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    PipelinePlan,
    PlannedJob,
    StatusReason,
)


class PlanJsonError(ValueError):
    """The JSON does not encode a pipeline plan."""


def dump_plan(plan: PipelinePlan) -> str:
    """Return the JSON encoding of `plan`."""
    return json.dumps(
        {
            "jobs": [
                {
                    **asdict(job),
                    "dependencies": sorted(job.dependencies),
                }
                for job in plan.jobs
            ]
        },
        indent=2,
    )


def load_plan(text: str) -> PipelinePlan:
    """Return the pipeline plan `dump_plan` encoded as `text`."""
    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        message = f"Not JSON: {error}"
        raise PlanJsonError(message) from error
    if not isinstance(document, dict) or "jobs" not in document:
        message = "A pipeline plan is a JSON object with jobs"
        raise PlanJsonError(message)
    try:
        return PipelinePlan(
            jobs=[_planned_job(encoded) for encoded in document["jobs"]]
        )
    except (KeyError, TypeError) as error:
        message = f"A job is missing or mistyped: {error}"
        raise PlanJsonError(message) from error


# The fields encoded as they are; the rest hold tuples, sets or objects.
PLAIN_FIELDS = [
    field.name
    for field in fields(PlannedJob)
    if field.name
    not in ("identity", "reasons", "inputs", "outputs", "dependencies")
]


def _planned_job(encoded: dict[str, Any]) -> PlannedJob:
    identity = encoded["identity"]
    return PlannedJob(
        **{name: encoded[name] for name in PLAIN_FIELDS},
        identity=None if identity is None else JobIdentity(**identity),
        reasons=tuple(
            StatusReason(
                kind=reason["kind"],
                files=tuple(reason["files"]),
                detail=reason["detail"],
            )
            for reason in encoded["reasons"]
        ),
        inputs=tuple(encoded["inputs"]),
        outputs=tuple(encoded["outputs"]),
        dependencies=frozenset(encoded["dependencies"]),
    )
