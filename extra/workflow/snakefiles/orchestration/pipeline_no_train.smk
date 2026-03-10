"""
No-train orchestration pipeline.

This variant skips dataset generation, classifier training, method training, and
evaluation rules and only runs transformation/aggregation/plotting on existing
outputs.

Runtime filters (--config):
    methods, method_sets, datasets, dataset_instance_indices, initializer,
    train_initializer, eval_initializer, eval_dataset_split.

Output namespacing:
    - Training artifacts are expected under
      `initializer-<train_initializer>` paths.
    - Evaluation/plot artifacts are expected under
      `train_initializer-<train_initializer>+eval_initializer-<eval_initializer>`.
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
STOP_SHIELD_DELTAS = _config["STOP_SHIELD_DELTAS"]
CUBE_NM_AR_BUDGET_MODE = _config["CUBE_NM_AR_BUDGET_MODE"]
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
# include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"

rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-builtin" for method_set in METHOD_SETS
        ] +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{method_set}+classifier_type-external" for method_set in METHOD_SETS
        ] +
        # The next two sets of plots should be identical, but include them both just in case
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/method_set-{HEATMAP_METHOD_SET}+classifier_type-external"
        ] +
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"
        ] +
        (
            [
                "extra/output/plot_results/cube_nm_ar/"
                f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
                f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/stop_shield-none"
            ]
            if "cube_nm_ar" in DATASETS
            else []
        ) +
        [
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
            f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/stop_shield-{delta}"
            for delta in STOP_SHIELD_DELTAS
            if "cube_nm_ar" in DATASETS
        ]

rule all_train_classifier:
    input:
        [
            (
                f"extra/output/trained_classifiers/{TRAIN_INITIALIZER_TAG}/"
                    f"dataset-{dataset}.bundle"
            )
            for dataset in DATASETS
        ]


rule all_pretrain_model:
    input:
        [
            (
                f"extra/output/pretrained_models/{TRAIN_INITIALIZER_TAG}/{pretrain_name}/"
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
                f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{method}/"
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
            for (train_hard_budget, _eval_hard_budget, train_soft_budget_param, _eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/trained_methods/{TRAIN_INITIALIZER_TAG}/{method}/"
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
            for (train_hard_budget, _eval_hard_budget, train_soft_budget_param, _eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]

rule all_eval_method:
    input:
        [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]


rule all_eval_method_shielded:
    input:
        [
            (
                f"extra/output/eval_results_shielded/delta-{stop_shield_delta}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for stop_shield_delta in STOP_SHIELD_DELTAS
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results_shielded/delta-{stop_shield_delta}/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.csv"
            )
            for stop_shield_delta in STOP_SHIELD_DELTAS
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (train_hard_budget, eval_hard_budget, train_soft_budget_param, eval_soft_budget_param) in BUDGET_PARAMS[method][dataset]
        ]
