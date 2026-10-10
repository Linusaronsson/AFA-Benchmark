"""
The settings every orchestration Snakefile shares, as the globals the rules
read.

`load_config` (afabench/core/workflow_settings.py) validates the merged
config and rejects a config mistake before any rule is defined; see
docs/reference/pipeline_configuration.md. Rules ask `WORKFLOW_SETTINGS` which
run and classifier a method uses, and address them with `OUTPUT_LAYOUT`.
Include this file before the rules.
"""

import os
import sys

# The workflow's own modules, such as execution.py, live in workflow/src.
# workflow.basedir is the directory of the orchestration Snakefile.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(workflow.basedir)), "src"))

from afabench.core.output_layout import OutputLayout
from afabench.core.workflow_settings import load_config
from execution import ExecutionPolicy
from images import ImageCommands
from job_records import JobRecordCommands

WORKFLOW_SETTINGS = load_config(config)
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

DATASET_REALIZATION_INDICES = WORKFLOW_SETTINGS.dataset_realization_indices
INITIALIZER = WORKFLOW_SETTINGS.initializer
EVAL_DATASET_SPLIT = WORKFLOW_SETTINGS.eval_dataset_split
OUTPUT_ROOT = WORKFLOW_SETTINGS.output_root
OUTPUT_LAYOUT = OutputLayout(
    root=OUTPUT_ROOT, initializer=INITIALIZER, eval_split=EVAL_DATASET_SPLIT
)
# For the merged results and plots, which the layout does not address.
INITIALIZER_TAG = OUTPUT_LAYOUT.initializer_tag
USE_WANDB = WORKFLOW_SETTINGS.use_wandb
SMOKE_TEST = WORKFLOW_SETTINGS.smoke_test
JOB_RECORDS = JobRecordCommands(
    EXECUTION, output_root=OUTPUT_ROOT, smoke_test=SMOKE_TEST
)
IMAGES = ImageCommands(EXECUTION, output_root=OUTPUT_ROOT)
PRETRAIN_NAMES = WORKFLOW_SETTINGS.pretrain_names
PRETRAIN_SCRIPT_NAMES = WORKFLOW_SETTINGS.pretrain_script_names
PRETRAIN_PARAMS = WORKFLOW_SETTINGS.pretrain_params
METHODS = WORKFLOW_SETTINGS.methods
REFERENCE_METHODS = WORKFLOW_SETTINGS.reference_methods
METHOD_TRAIN_SCRIPT_NAMES = WORKFLOW_SETTINGS.method_train_script_names
METHOD_CLASSIFIER_SCRIPT_NAMES = WORKFLOW_SETTINGS.method_classifier_script_names
METHOD_CLASSIFIER_SCRIPT_PARAMS = WORKFLOW_SETTINGS.method_classifier_script_params
METHOD_TO_PRETRAINED_MODEL = WORKFLOW_SETTINGS.method_to_pretrained_model
METHOD_SPECIFIC_PARAMS = WORKFLOW_SETTINGS.method_specific_params
DATASETS = WORKFLOW_SETTINGS.datasets
UNMASKERS = WORKFLOW_SETTINGS.unmaskers
BUDGET_PARAMS = WORKFLOW_SETTINGS.budget_params
CLASSIFIER_NAMES = WORKFLOW_SETTINGS.classifier_names
METHOD_SETS = WORKFLOW_SETTINGS.method_sets
EVAL_BATCH_SIZES = WORKFLOW_SETTINGS.eval_batch_sizes
DATASETS_USED_PER_PRETRAIN_NAME = WORKFLOW_SETTINGS.datasets_used_per_pretrain_name
HEATMAP_METHOD_SET = "heatmap_comparison"
