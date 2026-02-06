"""
AACO workflow using common pipeline rules.

Notes:
    - AACO+NN treats AACO as its pretrained model. Ensure
      method_options.yaml sets aaco_nn.pretrained_model_name: aaco
      and pretrain_mapping.yaml includes an aaco entry.
    - Outputs are nested under eval_split-{eval_dataset_split}/
      to match the shared evaluation/plotting rules.

Runtime filters (--config):
    methods (list[str], required): Subset of methods from method_options.yaml
    datasets (list[str], required): Subset of datasets to run
    dataset_instance_indices (list[int], default=[0,1]): Subset of seeds
    device (str, default='cpu'): Device for training
    use_wandb (bool, default=False): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Initialization strategy
    eval_dataset_split (str, default='test'): Dataset split for evaluation

Config files (--configfile):
    method_options.yaml
    pretrain_mapping.yaml
    methods.yaml
    classifier_names.yaml
    eval_hard_budgets.yaml
    soft_budget_params_*.yaml
    unmaskers.yaml

Example:
    snakemake -s extra/workflow/snakefiles/orchestration/AACO.smk \
        --configfile \
            extra/workflow/conf/eval_hard_budgets_fast.yaml \
            extra/workflow/conf/methods.yaml \
            extra/workflow/conf/method_options.yaml \
            extra/workflow/conf/pretrain_mapping.yaml \
            extra/workflow/conf/soft_budget_params_fast.yaml \
            extra/workflow/conf/unmaskers.yaml \
            extra/workflow/conf/classifier_names.yaml \
        --config methods='["aaco","aaco_nn"]' \
        --cores 4
"""

import os
import sys

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
PRETRAIN_NAMES = _config["PRETRAIN_NAMES"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHODS = _config["METHODS"]
METHODS_WITH_PRETRAINING_STAGE = _config["METHODS_WITH_PRETRAINING_STAGE"]
METHODS_WITHOUT_PRETRAINING_STAGE = _config[
    "METHODS_WITHOUT_PRETRAINING_STAGE"
]
METHOD_TRAIN_SCRIPT_NAMES = _config["METHOD_TRAIN_SCRIPT_NAMES"]
METHOD_TO_PRETRAINED_MODEL = _config["METHOD_TO_PRETRAINED_MODEL"]
METHOD_SPECIFIC_PARAMS = _config["METHOD_SPECIFIC_PARAMS"]
DATASETS = _config["DATASETS"]
UNMASKERS = _config["UNMASKERS"]
BUDGET_PARAMS = _config["BUDGET_PARAMS"]
CLASSIFIER_NAMES = _config["CLASSIFIER_NAMES"]
METHOD_SETS = _config["METHOD_SETS"]
EVAL_BATCH_SIZES = _config["EVAL_BATCH_SIZES"]
DATASETS_USED_PER_METHOD = _config["DATASETS_USED_PER_METHOD"]

include: "../rules/training.smk"
include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"


rule all:
    input:
        [
            (
                f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
                f"eval_perf/method_set-{method_set}+classifier_type-builtin"
            )
            for method_set in METHOD_SETS
        ]
        + [
            (
                f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
                f"eval_perf/method_set-{method_set}+classifier_type-external"
            )
            for method_set in METHOD_SETS
        ]
        + [
            (
                f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
                f"eval_actions/method_set-{method_set}+classifier_type-builtin"
            )
            for method_set in METHOD_SETS
        ]
        + [
            (
                f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
                f"eval_actions/method_set-{method_set}+classifier_type-external"
            )
            for method_set in METHOD_SETS
        ]
        + [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/time/"
        ]


rule all_pretrain_model:
    input:
        [
            (
                f"extra/output/pretrained_models/{pretrain_name}/"
                f"dataset-{dataset}+"
                f"instance_idx-{dataset_instance_idx}/"
                f"pretrain_seed-{dataset_instance_idx}/"
                "model.bundle"
            )
            for pretrain_name in PRETRAIN_NAMES
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS_USED_PER_METHOD[method]
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
                f"train_hard_budget-{train_hard_budget}+"
                f"train_soft_budget_param-{train_soft_budget_param}/"
                "method.bundle"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                _eval_hard_budget,
                train_soft_budget_param,
                _eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
        + [
            (
                f"extra/output/trained_methods/{method}/"
                f"dataset-{dataset}+"
                f"instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                f"train_seed-{dataset_instance_idx}+"
                f"train_hard_budget-{train_hard_budget}+"
                f"train_soft_budget_param-{train_soft_budget_param}/"
                "method.bundle"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                _eval_hard_budget,
                train_soft_budget_param,
                _eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]


rule all_eval_method:
    input:
        [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{method}/"
                f"dataset-{dataset}+"
                f"instance_idx-{dataset_instance_idx}/"
                f"pretrain_seed-{dataset_instance_idx}/"
                f"train_seed-{dataset_instance_idx}+"
                f"train_hard_budget-{train_hard_budget}+"
                f"train_soft_budget_param-{train_soft_budget_param}/"
                f"eval_seed-{dataset_instance_idx}+"
                f"eval_hard_budget-{eval_hard_budget}+"
                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                "eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
                eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
        + [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{method}/"
                f"dataset-{dataset}+"
                f"instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                f"train_seed-{dataset_instance_idx}+"
                f"train_hard_budget-{train_hard_budget}+"
                f"train_soft_budget_param-{train_soft_budget_param}/"
                f"eval_seed-{dataset_instance_idx}+"
                f"eval_hard_budget-{eval_hard_budget}+"
                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                "eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
                eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
