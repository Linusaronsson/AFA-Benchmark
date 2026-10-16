"""
The settings every orchestration Snakefile shares, as the globals the rules
read.

`load_config` (src/afabench/core/workflow_settings.py) validates the merged
config and rejects a config mistake before any rule is defined; see
docs/reference/pipeline_configuration.md. Rules ask `WORKFLOW_SETTINGS` what
a run covers and which scripts, classifier, pretrained model, unmasker and
eval batch size a method uses, and address every output with
`OUTPUT_LAYOUT`. Include this file before the rules.
"""

import os
import sys

# The workflow's own modules, such as execution.py, live in workflow/src.
# workflow.basedir is the directory of the orchestration Snakefile.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(workflow.basedir)), "src"))
# The orchestration environment does not install the project, so afabench is
# imported from the checkout's src/.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(workflow.basedir))), "src")
)

from afabench.core.output_layout import OutputLayout
from afabench.core.workflow_settings import load_config
from execution import ExecutionPolicy
from job_output import hide_job_messages
from job_records import JobRecordCommands
from wandb_runs import WandbProject, WandbRuns

WORKFLOW_SETTINGS = load_config(config)
WANDB_RUNS = WandbRuns(
    WandbProject.from_environment(os.environ) if WORKFLOW_SETTINGS.use_wandb else None,
    group=WORKFLOW_SETTINGS.wandb_group,
    # A job's own Snakemake on a cluster names its attempt by the SLURM job.
    slurm_job_id=None if workflow.is_main_process else os.environ.get("SLURM_JOB_ID"),
)
# Jobs re-parse this file on their node; only the invocation prints.
if workflow.is_main_process:
    logger.info(WORKFLOW_SETTINGS.summary(wandb_group_url=WANDB_RUNS.group_url()))
    hide_job_messages(
        logger, list_jobs=WORKFLOW_SETTINGS.list_jobs, wandb_runs=WANDB_RUNS
    )
EXECUTION = ExecutionPolicy(
    config,
    method_classifiers=WORKFLOW_SETTINGS.method_classifier_script_names,
    default_resources=(
        workflow.resource_settings.default_resources.parsed
        if workflow.resource_settings.default_resources
        else {}
    ),
    submits_to_cluster=lambda: workflow.is_main_process and workflow.non_local_exec,
)

OUTPUT_LAYOUT = OutputLayout(
    root=WORKFLOW_SETTINGS.output_root,
    initializer=WORKFLOW_SETTINGS.initializer,
    eval_split=WORKFLOW_SETTINGS.eval_dataset_split,
)
JOB_RECORDS = JobRecordCommands(
    EXECUTION,
    output_root=WORKFLOW_SETTINGS.output_root,
    smoke_test=WORKFLOW_SETTINGS.smoke_test,
)
# The scalars the rules read; everything else they ask of
# WORKFLOW_SETTINGS and OUTPUT_LAYOUT.
INITIALIZER = WORKFLOW_SETTINGS.initializer
EVAL_DATASET_SPLIT = WORKFLOW_SETTINGS.eval_dataset_split
USE_WANDB = WORKFLOW_SETTINGS.use_wandb
SMOKE_TEST = WORKFLOW_SETTINGS.smoke_test

# Import release readers only for explicit reuse; ordinary DAG construction
# stays independent of torch and Parquet.
RELEASE_REUSE = None
if WORKFLOW_SETTINGS.reuse_release:
    if WORKFLOW_SETTINGS.stages.dataset_generation:
        raise ValueError(
            "reuse_release=true requires stages={dataset_generation: false}; "
            "run all_generate_datasets first without reuse_release."
        )
    from pathlib import Path
    from afabench.release.reuse import ReleaseReuse
    RELEASE_REUSE = ReleaseReuse(Path(WORKFLOW_SETTINGS.output_root))
    RELEASE_REUSE.validate_datasets(
        OUTPUT_LAYOUT, WORKFLOW_SETTINGS.datasets,
        WORKFLOW_SETTINGS.dataset_realization_indices,
    )


def release_dataset_input(consumer, dataset, role):
    from pathlib import Path
    from snakemake.io import apply_wildcards

    def resolve(wildcards):
        path = apply_wildcards(dataset, wildcards)
        target = apply_wildcards(consumer, wildcards)
        if RELEASE_REUSE is not None and RELEASE_REUSE.compatible(
            Path(target), Path(path), role
        ):
            return ancient(path)
        return path

    return resolve
