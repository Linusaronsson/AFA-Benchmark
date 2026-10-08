"""
No-eval orchestration pipeline.

This variant skips dataset generation, classifier training, method training, and
evaluation rules. It only runs transformation, aggregation, and visualization rules
for existing evaluation outputs.

Runtime filters (--config, select subsets to run):
    methods (list[str], required): Subset of methods from method_options.yaml
    reference_methods (list[str], default=[]): Methods whose plotting-ready
        tables were restored from a benchmark release; aggregated with
        `methods` but never produced. See pipeline.smk.
    datasets (list[str], required): Subset of datasets to run
    dataset_realization_indices (list[int], default=[0,1,2,3,4]): Subset of
        random seeds
    device (str, default='cpu'): Deprecated global option, ignored by CPU-only
        processing; cannot be combined with execution.
    execution (mapping, default={}): Per-stage CPU/cuda policy for computational jobs;
        processing stages are fixed CPU-only and cannot be overridden.
    execution_site_file (str, required for SLURM submission): Profile-owned
        YAML allocation map; submitting without one fails before any job.
        Alternatively provide execution_site in a configuration file. A CLI
        --config replaces the workflow profile's config, so repeat
        execution_site_file=<site>/site.yaml whenever passing --config.
    use_wandb (bool, default=True): Enable W&B logging
    smoke_test (bool, default=False): Run smoke tests
    initializer (str, default='cold'): Initialization strategy
    eval_dataset_split (str, default='test'): Dataset split for existing
        evaluation outputs

CPU-only processing:
    Dataset generation (full pipeline only), transformations, aggregation and
    visualization always resolve to CPU, including with legacy device=cuda.
    These fixed stages have no execution defaults/overrides. The profile's
    execution_site.cpu allocation maps their partition/account and clears GPU
    requests; CPU counts, memory and runtime remain independently configurable.
    Conflicting rule allocation overrides fail before any submission. Heavy
    processing is submitted normally, not designated as login-node/local work.
    See docs/how-to/cpu_processing_execution.md for site requirements and
    final-target command-boundary verification.

Required files and usage:
    Existing evaluation parquet files and timing files at native output paths,
    scientific YAML configuration, and, for SLURM submission, the
    profile-owned site.yaml allocation map. No trained scripts are dispatched.
    snakemake -s extra/workflow/snakefiles/orchestration/pipeline_no_eval.smk \
        --workflow-profile extra/workflow/profiles/mixed-gres \
        --configfile <scientific.yaml> -n -p all
    Remove -n to submit processing jobs from an authorized shared-filesystem
    SLURM controller. Omit the cluster profile for ordinary local CPU use.

Output namespacing:
    - Every rule addresses dataset, classifier, pretrained-model and method
      bundles, evaluation tables and time files under extra/output through
      OUTPUT_LAYOUT (afabench/core/output_layout.py), the native layout that
      release manifests and restored benchmark releases also use.
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

from afabench.core.output_layout import OutputLayout
from config import load_config
from execution import ExecutionPolicy

_config = load_config(config)
EXECUTION = ExecutionPolicy(
    config,
    method_classifiers=_config["METHOD_CLASSIFIER_SCRIPT_NAMES"],
    default_resources=(
        workflow.resource_settings.default_resources.parsed
        if workflow.resource_settings.default_resources
        else {}
    ),
    submits_to_cluster=lambda: workflow.is_main_process and workflow.non_local_exec,
)

DATASET_REALIZATION_INDICES = _config["DATASET_REALIZATION_INDICES"]
INITIALIZER = _config["INITIALIZER"]
INITIALIZER_TAG = f"initializer-{INITIALIZER}"
EVAL_DATASET_SPLIT = _config["EVAL_DATASET_SPLIT"]
OUTPUT_LAYOUT = OutputLayout(
    root="extra/output", initializer=INITIALIZER, eval_split=EVAL_DATASET_SPLIT
)
USE_WANDB = _config["USE_WANDB"]
SMOKE_TEST = _config["SMOKE_TEST"]
PRETRAIN_NAMES = _config["PRETRAIN_NAMES"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHODS = _config["METHODS"]
REFERENCE_METHODS = _config["REFERENCE_METHODS"]
COMPARED_METHODS_WITH_PRETRAINING_STAGE = _config["COMPARED_METHODS_WITH_PRETRAINING_STAGE"]
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

# NOTE: exclude training rules and eval rules!
# include: "../rules/training.smk"
# include: "../rules/classifier_training.smk"
# include: "../rules/evaluation.smk"
include: "../rules/transformations.smk"
include: "../rules/aggregation.smk"
include: "../rules/visualization.smk"
include: "../rules/helpers.smk"
