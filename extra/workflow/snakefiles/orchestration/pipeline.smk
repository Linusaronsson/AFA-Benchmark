"""
Full orchestration pipeline: datasets, classifiers, pretraining, training,
evaluation and plots.

Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    datasets (list[str], required): Subset of datasets to run. Every dataset
        key needs a file extra/conf/dataset_key/<key>.yaml.
    dataset_instance_indices (list[int], default=[0,1,2,3,4]): Subset of random seeds
    device (str, default='cpu'): Deprecated invocation-wide device for
        computational jobs, with a warning. Cannot be combined with execution.
    execution (mapping, default={}): CPU/cuda defaults for classifier,
        pretraining, training and evaluation. methods.<name> overrides training,
        evaluation and method-specific classifier choices; pretrained_models
        overrides pretraining by named model. External classifiers use only
        the classifier default. Overrides take precedence over stage defaults.
        Shipped declarations: extra/workflow/conf/execution/{kdd26,all}.yaml.
    execution_site_file (str, optional): Profile-owned YAML allocation map.
        Alternatively provide execution_site in a configuration file. A CLI
        --config replaces the workflow profile's config, so repeat
        execution_site_file=<site>/site.yaml whenever passing --config.
    use_wandb (bool, default=True): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Initialization strategy, a file in
        extra/conf/initializer/
    eval_dataset_split (str, default='test'): Dataset split for evaluation

Training contract:
    The pretrain_model and train_method rules (rules/training.smk) pass every
    pretraining and training script the plain `key=value` training contract
    rendered by extra/workflow/src/training_contract.py: dataset, classifier
    and pretrained-model bundle paths, save_path, initializer, unmasker,
    dataset_key, hard_budget, soft_budget_param, device, seed, use_wandb and
    smoke_test (pretraining receives no pretrained model and no budgets).
    Methods add their own arguments through method_specific_params and
    pretrain_params. See docs/adr/0001-training-contract-as-library.md.

Execution configuration and required files:
    Methods retain their independent scripts and native bundle/result paths.
    Site profiles own CPU/GPU partition, account and GPU request syntax;
    CPU counts, memory and runtime remain separate resource settings.
    Invalid selected-method and prerequisite execution fails before submission,
    including conflicting device arguments in classifier/pretraining params.
    See docs/tutorials/mixed_execution.md and prerequisite_execution.md for
    execution YAML, site.yaml, migration and captured-submission tests.

Usage:
    Full benchmark, one invocation from an authorized SLURM submit host of a
    single cluster, with the repository, environment and outputs on a shared
    filesystem (config/kdd26 bundles execution/kdd26.yaml):
        snakemake --profile extra/workflow/profiles/config/kdd26 \
            --workflow-profile extra/workflow/profiles/<site> -n -p all
    Inspect the planned resources and device arguments, then remove -n -p.
    Full method set: list the all.yaml config files with --configfile and add
    extra/workflow/conf/execution/all.yaml. Local CPU smoke test without SLURM
    or GPUs (config/all has no execution file, so every job runs on CPU):
        snakemake --profile extra/workflow/profiles/config/all all --jobs 8 \
            --config "datasets=[cube]" "dataset_instance_indices=[0]" \
            smoke_test=true use_wandb=false
    See docs/tutorials/reproduce_full_results.md and slurm_integration.md.

CPU-only processing:
    Dataset generation (full pipeline only), transformations, aggregation and
    visualization always resolve to CPU, including with legacy device=cuda.
    These fixed activities have no execution defaults/overrides. The profile's
    execution_site.cpu allocation maps their partition/account and clears GPU
    requests; CPU counts, memory and runtime remain independently configurable.
    Conflicting rule allocation overrides fail before any submission. Heavy
    processing is submitted normally, not designated as login-node/local work.
    See docs/tutorials/cpu_processing_execution.md for site requirements and
    final-target command-boundary verification.

Output namespacing:
    - All initializer-dependent artifacts are stored under
      `initializer-<initializer>` to allow side-by-side comparisons
      (for example `cold` vs `missingness`) without overwriting.

Config files (--configfile):
    Fixed definitions:
        method_options.yaml,
        pretrain_mapping.yaml,
        methods.yaml
        classifier_names.yaml
    Runtime params:
        eval_hard_budgets.yaml,
        soft_budget_params_*.yaml,
        unmaskers.yaml (values are files in extra/conf/unmasker/)

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

     Note: Pretraining is skipped per pretrained model for datasets ignored by
     BOTH hard and soft budgets across all methods that consume that model.
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
from execution import ExecutionPolicy

_config = load_config(config)
EXECUTION = ExecutionPolicy(config)
DEFAULT_RESOURCES = workflow.resource_settings.default_resources.parsed if workflow.resource_settings.default_resources else {}

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
DATASETS_USED_PER_PRETRAIN_NAME = _config["DATASETS_USED_PER_PRETRAIN_NAME"]
HEATMAP_METHOD_SET = "heatmap_comparison"

include: "../rules/dataset_generation.smk"
include: "../rules/training.smk"
include: "../rules/classifier_training.smk"
include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
