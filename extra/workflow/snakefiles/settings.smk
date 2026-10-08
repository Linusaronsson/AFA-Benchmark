"""
The settings every orchestration Snakefile shares, as the globals the rules
read.

`load_config` (afabench/core/workflow_settings.py) validates the merged
config and rejects a config mistake before any rule is defined; see
docs/reference/pipeline_configuration.md. Include this file before the
rules.
"""

import os
import sys

# The workflow's own modules, such as execution.py, live in extra/workflow/src.
# workflow.basedir is the directory of the orchestration Snakefile.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(workflow.basedir)), "src"))

from afabench.core.output_layout import OutputLayout
from afabench.core.workflow_settings import load_config
from execution import ExecutionPolicy

_settings = load_config(config)
EXECUTION = ExecutionPolicy(
    config,
    method_classifiers=_settings.method_classifier_script_names,
    default_resources=(
        workflow.resource_settings.default_resources.parsed
        if workflow.resource_settings.default_resources
        else {}
    ),
    submits_to_cluster=lambda: workflow.is_main_process and workflow.non_local_exec,
)

DATASET_REALIZATION_INDICES = _settings.dataset_realization_indices
INITIALIZER = _settings.initializer
INITIALIZER_TAG = f"initializer-{INITIALIZER}"
EVAL_DATASET_SPLIT = _settings.eval_dataset_split
OUTPUT_LAYOUT = OutputLayout(
    root="extra/output", initializer=INITIALIZER, eval_split=EVAL_DATASET_SPLIT
)
USE_WANDB = _settings.use_wandb
SMOKE_TEST = _settings.smoke_test
PRETRAIN_NAMES = _settings.pretrain_names
PRETRAIN_SCRIPT_NAMES = _settings.pretrain_script_names
PRETRAIN_PARAMS = _settings.pretrain_params
METHODS = _settings.methods
REFERENCE_METHODS = _settings.reference_methods
COMPARED_METHODS_WITH_PRETRAINING_STAGE = _settings.compared_methods_with_pretraining_stage
METHOD_TRAIN_SCRIPT_NAMES = _settings.method_train_script_names
METHOD_CLASSIFIER_SCRIPT_NAMES = _settings.method_classifier_script_names
METHOD_CLASSIFIER_SCRIPT_PARAMS = _settings.method_classifier_script_params
METHOD_TO_PRETRAINED_MODEL = _settings.method_to_pretrained_model
METHOD_SPECIFIC_PARAMS = _settings.method_specific_params
DATASETS = _settings.datasets
UNMASKERS = _settings.unmaskers
BUDGET_PARAMS = _settings.budget_params
CLASSIFIER_NAMES = _settings.classifier_names
METHOD_SETS = _settings.method_sets
EVAL_BATCH_SIZES = _settings.eval_batch_sizes
DATASETS_USED_PER_PRETRAIN_NAME = _settings.datasets_used_per_pretrain_name
HEATMAP_METHOD_SET = "heatmap_comparison"
