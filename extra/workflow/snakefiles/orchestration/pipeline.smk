"""
Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    datasets (list[str], required): Subset of datasets to run
    dataset_instance_indices (list[int], default=[0,1]): Subset of random seeds
    device (str, default='cpu'): Fallback device when no compute-platform
        override is configured for a method/pretraining family
    default_compute_platform (str, default='cpu'): Default compute platform
        label used by workflow rules
    compute_platform_devices (dict[str, str], default={"cpu":"cpu",
        "gpu":"cuda"}): Maps logical compute platforms to script devices
    classifier_compute_platform (str, default=default_compute_platform):
        Compute platform label for shared classifier training
    slurm_extra_by_compute_platform (dict[str, str], optional): Optional
        per-platform SLURM extras such as GPU or CPU-node constraints
    use_wandb (bool, default=False): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Legacy initializer for both train/eval
    train_initializer (str, optional): Initializer for train/pretrain stages
    eval_initializer (str, optional): Initializer used during evaluation
    eval_dataset_split (str, default='val'): Dataset split for evaluation
    stop_shield_deltas (list[float], default=[]): Extra stop-shielded eval
        passes to run in parallel with standard evaluation.
    dual_lambdas (list[float], default=[]): Extra dualized-stop eval passes
        to run in parallel with standard evaluation.

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
STOP_SHIELD_DELTAS = _config["STOP_SHIELD_DELTAS"]
DUAL_LAMBDAS = _config["DUAL_LAMBDAS"]
CUBE_NM_AR_BUDGET_MODE = _config["CUBE_NM_AR_BUDGET_MODE"]
DEVICE = _config["DEVICE"]
DEFAULT_COMPUTE_PLATFORM = _config["DEFAULT_COMPUTE_PLATFORM"]
COMPUTE_PLATFORM_DEVICES = _config["COMPUTE_PLATFORM_DEVICES"]
CLASSIFIER_COMPUTE_PLATFORM = _config["CLASSIFIER_COMPUTE_PLATFORM"]
SLURM_EXTRA_BY_COMPUTE_PLATFORM = _config[
    "SLURM_EXTRA_BY_COMPUTE_PLATFORM"
]
USE_WANDB = _config["USE_WANDB"]
SMOKE_TEST = _config["SMOKE_TEST"]
PRETRAIN_NAMES = _config["PRETRAIN_NAMES"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
PRETRAIN_COMPUTE_PLATFORMS = _config["PRETRAIN_COMPUTE_PLATFORMS"]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHODS = _config["METHODS"]
METHODS_WITH_PRETRAINING_STAGE = _config["METHODS_WITH_PRETRAINING_STAGE"]
METHODS_WITHOUT_PRETRAINING_STAGE = _config["METHODS_WITHOUT_PRETRAINING_STAGE"]
METHOD_TRAIN_SCRIPT_NAMES = _config["METHOD_TRAIN_SCRIPT_NAMES"]
METHOD_COMPUTE_PLATFORMS = _config["METHOD_COMPUTE_PLATFORMS"]
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


def _device_for_compute_platform(compute_platform: str) -> str:
    return COMPUTE_PLATFORM_DEVICES[compute_platform]


def _slurm_extra_for_compute_platform(compute_platform: str) -> str:
    return SLURM_EXTRA_BY_COMPUTE_PLATFORM.get(compute_platform, "")


def _classifier_device() -> str:
    return _device_for_compute_platform(CLASSIFIER_COMPUTE_PLATFORM)


def _classifier_slurm_extra() -> str:
    return _slurm_extra_for_compute_platform(CLASSIFIER_COMPUTE_PLATFORM)


def _pretrain_compute_platform(pretrained_model_name: str) -> str:
    return PRETRAIN_COMPUTE_PLATFORMS.get(
        pretrained_model_name,
        DEFAULT_COMPUTE_PLATFORM,
    )


def _pretrain_device(pretrained_model_name: str) -> str:
    return _device_for_compute_platform(
        _pretrain_compute_platform(pretrained_model_name)
    )


def _pretrain_slurm_extra(pretrained_model_name: str) -> str:
    return _slurm_extra_for_compute_platform(
        _pretrain_compute_platform(pretrained_model_name)
    )


def _method_compute_platform(method: str) -> str:
    return METHOD_COMPUTE_PLATFORMS.get(method, DEFAULT_COMPUTE_PLATFORM)


def _method_device(method: str) -> str:
    return _device_for_compute_platform(_method_compute_platform(method))


def _method_slurm_extra(method: str) -> str:
    return _slurm_extra_for_compute_platform(_method_compute_platform(method))

include: "../rules/dataset_generation.smk"
include: "../rules/training.smk"
include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
