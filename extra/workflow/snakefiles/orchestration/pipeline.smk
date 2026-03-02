"""
Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    datasets (list[str], required): Subset of datasets to run
    dataset_instance_indices (list[int], default=[0,1]): Subset of random seeds
    device (str, default='cpu'): Device for training
    use_wandb (bool, default=False): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Legacy initializer for both train/eval
    train_initializer (str, optional): Initializer for train/pretrain stages
    eval_initializer (str, optional): Initializer used during evaluation
    eval_dataset_split (str, default='val'): Dataset split for evaluation

Output namespacing:
    - Training artifacts are namespaced by
      `initializer-<train_initializer>`.
    - Evaluation and plots are namespaced by
      `train_initializer-<train_initializer>+eval_initializer-<eval_initializer>`.

Config files (--configfile):
    Fixed definitions:
        method_options.yaml,
        pretrain_mapping.yaml,
        methods.yaml
        classifier_names.yaml
    Runtime params:
        eval_hard_budgets.yaml,
        soft_budget_params_*.yaml,
        unmaskers.yaml

    Note: method_options.yaml can include eval_to_train_hard_budget_mapping to
    specify different budgets for training vs evaluation per method/dataset.

    Note: method_options.yaml can include eval_batch_size to specify different
    batch sizes for evaluation per method and dataset. Format:
        eval_batch_size:
          default: <batch_size>
          <dataset_name>: <batch_size>

    Note: method_options.yaml can include hard_budget_ignored_datasets to skip
    hard budget training/evaluation for specific datasets per method. Format:
        hard_budget_ignored_datasets: [dataset1, dataset2, ...]
    When set, hard budget combinations are excluded for those datasets.

    Note: method_options.yaml can include soft_budget_ignored_datasets to skip
    soft budget training/evaluation for specific datasets per method. Format:
     soft_budget_ignored_datasets: [dataset1, dataset2, ...]
     When set, soft budget combinations are excluded for those datasets.

     Note: Pretraining is skipped for datasets ignored by BOTH hard and soft
     budgets (i.e., a dataset is pretrained only if at least one budget type
     is used for that dataset/method combination).
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
TRAIN_INITIALIZER = _config["TRAIN_INITIALIZER"]
EVAL_INITIALIZER = _config["EVAL_INITIALIZER"]
TRAIN_INITIALIZER_TAG = f"initializer-{TRAIN_INITIALIZER}"
INITIALIZER_TAG = (
    f"train_initializer-{TRAIN_INITIALIZER}+"
    f"eval_initializer-{EVAL_INITIALIZER}"
)
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
METHOD_TO_CLASSIFIER_BUNDLE_METHOD = _config[
    "METHOD_TO_CLASSIFIER_BUNDLE_METHOD"
]
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

include: "../rules/dataset_generation.smk"
include: "../rules/training.smk"
include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
