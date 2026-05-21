"""
No-train orchestration pipeline.

This variant skips dataset generation, classifier training, method training, and
evaluation rules and only runs transformation/aggregation/plotting on existing
outputs.

Runtime filters (--config):
    methods, method_sets, datasets, dataset_instance_indices, initializer,
    eval_dataset_split.

Output namespacing:
    - Initializer-dependent artifacts are expected under
      `initializer-<initializer>` paths.
"""

import os
import sys
import time
from datetime import datetime

snakefile_dir = workflow.basedir
workflow_dir = os.path.dirname(os.path.dirname(snakefile_dir))
src_dir = os.path.join(workflow_dir, "src")
sys.path.insert(0, src_dir)

from config import load_config

_config = load_config(config)

NO_PRETRAIN_STR = _config["NO_PRETRAIN_STR"]
DATASET_INSTANCE_INDICES = _config["DATASET_INSTANCE_INDICES"]
INITIALIZER = _config["INITIALIZER"]
INITIALIZER_TAG = f"initializer-{INITIALIZER}"
EVAL_DATASET_SPLIT = _config["EVAL_DATASET_SPLIT"]
DEVICE = _config["DEVICE"]
USE_WANDB = _config["USE_WANDB"]
SMOKE_TEST = _config["SMOKE_TEST"]
PRETRAIN_NAMES = _config["PRETRAIN_NAMES"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHODS = _config["METHODS"]
METHODS_WITH_PRETRAINING_STAGE = _config["METHODS_WITH_PRETRAINING_STAGE"]
METHODS_WITHOUT_PRETRAINING_STAGE = _config["METHODS_WITHOUT_PRETRAINING_STAGE"]
METHOD_TRAIN_SCRIPT_NAMES = _config["METHOD_TRAIN_SCRIPT_NAMES"]
METHOD_CLASSIFIER_SCRIPT_NAMES = _config["METHOD_CLASSIFIER_SCRIPT_NAMES"]
METHOD_CLASSIFIER_SCRIPT_PARAMS = _config["METHOD_CLASSIFIER_SCRIPT_PARAMS"]
METHOD_TO_PRETRAINED_MODEL = _config["METHOD_TO_PRETRAINED_MODEL"]
METHOD_SPECIFIC_PARAMS = _config["METHOD_SPECIFIC_PARAMS"]
DATASETS = _config["DATASETS"]
UNMASKERS = _config["UNMASKERS"]
BUDGET_PARAMS = _config["BUDGET_PARAMS"]
CLASSIFIER_NAMES = _config["CLASSIFIER_NAMES"]
METHOD_SETS = _config["METHOD_SETS"]
EVAL_BATCH_SIZES = _config["EVAL_BATCH_SIZES"]
HARD_BUDGET_IGNORED_DATASETS = _config["HARD_BUDGET_IGNORED_DATASETS"]
SOFT_BUDGET_IGNORED_DATASETS = _config["SOFT_BUDGET_IGNORED_DATASETS"]
DATASETS_USED_PER_METHOD = _config["DATASETS_USED_PER_METHOD"]
HEATMAP_METHOD_SET = "heatmap_comparison"

# NOTE: exclude training rules!
# include: "../rules/training.smk"
# include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
