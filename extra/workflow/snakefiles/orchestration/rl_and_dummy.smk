import os
import sys
import time
from datetime import datetime

# Add src directory to Python path for imports
# Use workflow.basedir which gives us the directory containing the Snakefile
# The Snakefile is at: extra/workflow/snakefiles/orchestration/RL_and_dummy.smk
# We need to import from: extra/workflow/src/
snakefile_dir = workflow.basedir
workflow_dir = os.path.dirname(os.path.dirname(snakefile_dir))
src_dir = os.path.join(workflow_dir, "src")
sys.path.insert(0, src_dir)

from config import load_config

_config = load_config(config)

NO_PRETRAIN_STR = _config["NO_PRETRAIN_STR"]
DATASET_INSTANCE_INDICES = _config["DATASET_INSTANCE_INDICES"]
INITIALIZER = _config["INITIALIZER"]
EVAL_DATASET_SPLIT = _config["EVAL_DATASET_SPLIT"]
DEVICE = _config["DEVICE"]
USE_WANDB = _config["USE_WANDB"]
SMOKE_TEST = _config["SMOKE_TEST"]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHODS = _config["METHODS"]
METHODS_WITH_PRETRAINING_STAGE = _config["METHODS_WITH_PRETRAINING_STAGE"]
METHODS_WITHOUT_PRETRAINING_STAGE = _config["METHODS_WITHOUT_PRETRAINING_STAGE"]
METHOD_TRAIN_SCRIPT_NAMES = _config["METHOD_TRAIN_SCRIPT_NAMES"]
METHOD_PRETRAIN_SCRIPT_NAMES = _config["METHOD_PRETRAIN_SCRIPT_NAMES"]
METHOD_SPECIFIC_PARAMS = _config["METHOD_SPECIFIC_PARAMS"]
DATASETS = _config["DATASETS"]
UNMASKERS = _config["UNMASKERS"]
HARD_BUDGETS = _config["HARD_BUDGETS"]
SOFT_BUDGET_PARAMS = _config["SOFT_BUDGET_PARAMS"]
HARD_BUDGET_AND_SOFT_BUDGET_PARAMS = _config["HARD_BUDGET_AND_SOFT_BUDGET_PARAMS"]

include: "../rules/training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"

rule all:
    input:
        "extra/output/plot_results/eval_perf/rl_and_dummy",
        "extra/output/plot_results/time/rl_and_dummy",


rule all_pretrain_model:
    input:
        [
            (
                f"extra/output/pretrained_models/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
        ]


rule all_train_method:
    input:
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (hard_budget, soft_budget_param) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/trained_methods/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}.bundle"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (hard_budget, soft_budget_param) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ]
