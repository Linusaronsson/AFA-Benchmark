"""
Snakemake workflow for AACO and AACO+NN methods.

PREREQUISITE: Run mlp.smk first to train MLP classifiers!
    snakemake -s extra/workflow/snakefiles/mlp.smk \
        --configfile extra/workflow/conf/mlp.yaml -j 4

This workflow handles:
1. Training AACO methods with various cost parameters
2. Training AACO+NN (behavioral cloning from AACO)
3. Evaluation of both methods

Usage:
    snakemake -s extra/workflow/snakefiles/AACO.smk \
        --configfile extra/workflow/conf/aaco_full.yaml -j 4
"""
import os
from datetime import datetime

# Configuration
DATASET_PATH_PREFIX = config.get("dataset_path_prefix", "extra/data")
DATASET_INSTANCE_INDICES = config.get("dataset_instance_indices", (0, 1))
INITIALIZER = config.get("initializer", "aaco_default")
EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")
DEVICE = config.get("device", "cpu")
DEVICE_AACO = config.get("device_aaco", DEVICE)
DEVICE_AACO_NN = config.get("device_aaco_nn", DEVICE)
# Per-dataset device overrides (for datasets that need GPU)
DEVICE_AACO_BY_DATASET_RAW = config.get("device_aaco_by_dataset", {})
DEVICE_AACO_NN_BY_DATASET_RAW = config.get("device_aaco_nn_by_dataset", {})
EVAL_DEVICE_BY_METHOD = config.get(
    "eval_device_by_method",
    {"aaco": DEVICE_AACO, "aaco_nn": DEVICE_AACO_NN},
)
USE_WANDB = config.get("use_wandb", False)
SMOKE_TEST = config.get("smoke_test", False)

# Datasets to process
DATASETS = config.get("datasets", None)
if DATASETS is None:
    raise ValueError("Expected datasets to be provided.")

# Unmaskers per dataset
UNMASKERS_RAW = config.get("unmaskers", None)
if UNMASKERS_RAW is None:
    raise ValueError("Expected unmaskers to be provided.")
UNMASKERS = UNMASKERS_RAW | {
    dataset: UNMASKERS_RAW["default"]
    for dataset in DATASETS
    if dataset not in UNMASKERS_RAW
}

# Hard budgets for evaluation and hard-budget training
HARD_BUDGETS_RAW = config.get("hard_budgets", None)
if HARD_BUDGETS_RAW is None:
    raise ValueError("Expected hard_budgets to be provided.")
HARD_BUDGETS = HARD_BUDGETS_RAW | {
    dataset: HARD_BUDGETS_RAW["default"]
    for dataset in DATASETS
    if dataset not in HARD_BUDGETS_RAW
}

# Cost parameters (soft budget) for AACO training
COST_PARAMS_RAW = config.get("cost_params", None)
if COST_PARAMS_RAW is None:
    raise ValueError("Expected cost_params to be provided.")
COST_PARAMS = COST_PARAMS_RAW | {
    dataset: COST_PARAMS_RAW["default"]
    for dataset in DATASETS
    if dataset not in COST_PARAMS_RAW
}

# Whether to also train AACO+NN
TRAIN_AACO_NN = config.get("train_aaco_nn", False)
# Whether to evaluate soft budgets (hard_budget=null with cost_param metadata)
SOFT_BUDGET_EVAL = config.get("soft_budget_eval", False)

# Hard budget settings for training
TRAIN_HARD_BUDGETS = {
    dataset: (
        (["null"] if SOFT_BUDGET_EVAL else []) + HARD_BUDGETS[dataset]
    )
    for dataset in DATASETS
}

# Methods to include in final output
AACO_METHODS = ["aaco"]
if TRAIN_AACO_NN:
    AACO_METHODS.append("aaco_nn")


def _normalize_dataset_name(dataset: str) -> str:
    name = dataset.replace("_without_noise", "")
    return name.replace("_", "")


def _eval_device(method: str) -> str:
    return EVAL_DEVICE_BY_METHOD.get(method, DEVICE)


def _device_aaco(dataset: str) -> str:
    """Get device for AACO training, with per-dataset override."""
    return DEVICE_AACO_BY_DATASET_RAW.get(dataset, DEVICE_AACO)


def _device_aaco_nn(dataset: str) -> str:
    """Get device for AACO+NN training, with per-dataset override."""
    return DEVICE_AACO_NN_BY_DATASET_RAW.get(dataset, DEVICE_AACO_NN)


def _slurm_extra_aaco(dataset: str) -> str:
    """Get SLURM extra args for AACO training based on device."""
    device = _device_aaco(dataset)
    if device == "cuda":
        return "--gres=gpu:T4:1"
    return "--constraint=NOGPU"


def _slurm_extra_aaco_nn(dataset: str) -> str:
    """Get SLURM extra args for AACO+NN training based on device."""
    device = _device_aaco_nn(dataset)
    if device == "cuda":
        return "--gres=gpu:T4:1"
    return "--constraint=NOGPU"


# ============================================================================
# MAIN TARGETS
# ============================================================================

rule all:
    input:
        "extra/output/plot_results/AACO",


rule train_aaco_all:
    input:
        [
            (
                f"extra/output/trained_methods/aaco/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"train_seed-{dataset_instance_idx}+"
                        f"train_hard_budget-{train_hard_budget}+"
                        f"cost_param-{cost_param}.bundle"
            )
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for cost_param in COST_PARAMS[dataset]
            for train_hard_budget in TRAIN_HARD_BUDGETS[dataset]
        ]


rule train_aaco_nn_all:
    input:
        [
            (
                f"extra/output/trained_methods/aaco_nn/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"train_seed-{dataset_instance_idx}+"
                        f"train_hard_budget-{train_hard_budget}+"
                        f"cost_param-{cost_param}.bundle"
            )
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for cost_param in COST_PARAMS[dataset]
            for train_hard_budget in TRAIN_HARD_BUDGETS[dataset]
        ] if TRAIN_AACO_NN else []


# ============================================================================
# TRAINING RULES
# ============================================================================
# NOTE: MLP classifiers are assumed to be already trained via mlp.smk
# The classifier path is specified as an input dependency to fail early if missing.

rule train_aaco:
    """Train AACO method with a given cost parameter."""
    input:
        train=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.dataset_instance_idx}/train.bundle",
        classifier=(
            "extra/output/classifiers/masked_mlp_classifier/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "seed-{train_seed}.bundle"
        ),
    output:
        directory(
            "extra/output/trained_methods/aaco/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "cost_param-{cost_param}.bundle"
        ),
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        device=lambda wildcards: _device_aaco(wildcards.dataset),
    resources:
        slurm_extra=lambda wildcards: _slurm_extra_aaco(wildcards.dataset),
    shell:
        """
        python scripts/train/aaco.py \
            dataset_artifact_name={input.train} \
            classifier_bundle_path={input.classifier} \
            save_path={output} \
            cost_param={wildcards.cost_param} \
            hard_budget={wildcards.train_hard_budget} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} \
            seed={wildcards.train_seed} \
            smoke_test={SMOKE_TEST}
        """


rule train_aaco_nn:
    """Train AACO+NN (behavioral cloning from AACO)."""
    input:
        train=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.dataset_instance_idx}/train.bundle",
        aaco_method=(
            "extra/output/trained_methods/aaco/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "cost_param-{cost_param}.bundle"
        ),
        classifier=(
            "extra/output/classifiers/masked_mlp_classifier/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "seed-{train_seed}.bundle"
        ),
    output:
        directory(
            "extra/output/trained_methods/aaco_nn/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "cost_param-{cost_param}.bundle"
        ),
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        device=lambda wildcards: _device_aaco_nn(wildcards.dataset),
    resources:
        slurm_extra=lambda wildcards: _slurm_extra_aaco_nn(wildcards.dataset),
    shell:
        """
        python scripts/train/aaco_nn.py \
            aaco_bundle_path={input.aaco_method} \
            dataset_artifact_name={input.train} \
            classifier_bundle_path={input.classifier} \
            save_path={output} \
            hard_budget={wildcards.train_hard_budget} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} \
            seed={wildcards.train_seed} \
            smoke_test={SMOKE_TEST}
        """


# ============================================================================
# EVALUATION RULES
# ============================================================================

rule eval_aaco_method:
    """Evaluate an AACO-based method."""
    input:
        dataset=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.dataset_instance_idx}/{EVAL_DATASET_SPLIT}.bundle",
        method=(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{eval_hard_budget}+"
                    "cost_param-{cost_param}.bundle"
        ),
    output:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        initializer_dataset_name=lambda wildcards: _normalize_dataset_name(
            wildcards.dataset
        ),
        eval_device=lambda wildcards: _eval_device(wildcards.method),
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input.method} \
            components/initializers@initializer={INITIALIZER} \
            initializer.kwargs.dataset_name={params.initializer_dataset_name} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input.dataset} \
            save_path={output} \
            classifier_bundle_path=null \
            seed={wildcards.eval_seed} \
            device={params.eval_device} \
            hard_budget={wildcards.eval_hard_budget} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        """

rule eval_aaco_method_soft:
    """Evaluate an AACO-based method in soft budget mode."""
    input:
        dataset=lambda wc: f"{DATASET_PATH_PREFIX}/{wc.dataset}/{wc.dataset_instance_idx}/{EVAL_DATASET_SPLIT}.bundle",
        method=(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-null+"
                    "cost_param-{cost_param}.bundle"
        ),
    output:
        "extra/output/eval_results_soft/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        initializer_dataset_name=lambda wildcards: _normalize_dataset_name(
            wildcards.dataset
        ),
        eval_device=lambda wildcards: _eval_device(wildcards.method),
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input.method} \
            components/initializers@initializer={INITIALIZER} \
            initializer.kwargs.dataset_name={params.initializer_dataset_name} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input.dataset} \
            save_path={output} \
            classifier_bundle_path=null \
            seed={wildcards.eval_seed} \
            device={params.eval_device} \
            hard_budget=null \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        """

# ============================================================================
# POST-PROCESSING RULES
# ============================================================================

rule count_selections:
    """Convert prev_selections_performed list to selections_performed count."""
    input:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py count_selections {input} {output}
        """

rule count_selections_soft:
    """Convert prev_selections_performed list to selections_performed count."""
    input:
        "extra/output/eval_results_soft/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results_soft2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py count_selections {input} {output}
        """

rule add_metadata_to_eval_data:
    """Add training and evaluation metadata columns."""
    input:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col afa_method={wildcards.method} \
            --col dataset={wildcards.dataset} \
            --col train_seed={wildcards.train_seed} \
            --col cost_param={wildcards.cost_param} \
            --col hard_budget={wildcards.eval_hard_budget} \
            --col soft_budget_param=""
        """

rule add_metadata_to_eval_data_soft:
    """Add training and evaluation metadata columns."""
    input:
        "extra/output/eval_results_soft2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results_soft3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col afa_method={wildcards.method} \
            --col dataset={wildcards.dataset} \
            --col train_seed={wildcards.train_seed} \
            --col cost_param={wildcards.cost_param} \
            --col hard_budget="" \
            --col soft_budget_param={wildcards.cost_param}
        """

rule pivot_long_classifier:
    """Pivot classifier predictions to tidy format."""
    input:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-{eval_hard_budget}/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py pivot_long_classifier {input} {output}
        """

rule pivot_long_classifier_soft:
    """Pivot classifier predictions to tidy format."""
    input:
        "extra/output/eval_results_soft3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    output:
        "extra/output/eval_results_soft4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "train_seed-{train_seed}+"
                "cost_param-{cost_param}/"
                    "eval_seed-{eval_seed}+"
                    "eval_hard_budget-null/"
                        "eval_data.csv",
    shell:
        """
        python scripts/misc/transform_eval_data.py pivot_long_classifier {input} {output}
        """

# ============================================================================
# MERGE AND PLOT
# ============================================================================

rule merge_eval:
    """Merge all evaluation results into a single CSV."""
    input:
        (
            [
                (
                    f"extra/output/eval_results4/{method}/"
                        f"dataset-{dataset}+"
                        f"instance_idx-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"cost_param-{cost_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{hard_budget}/"
                                    f"eval_data.csv"
                )
                for method in AACO_METHODS
                for dataset in DATASETS
                for dataset_instance_idx in DATASET_INSTANCE_INDICES
                for cost_param in COST_PARAMS[dataset]
                for hard_budget in HARD_BUDGETS[dataset]
            ]
            + (
                [
                    (
                        f"extra/output/eval_results_soft4/{method}/"
                            f"dataset-{dataset}+"
                            f"instance_idx-{dataset_instance_idx}/"
                                f"train_seed-{dataset_instance_idx}+"
                                f"cost_param-{cost_param}/"
                                    f"eval_seed-{dataset_instance_idx}+"
                                    f"eval_hard_budget-null/"
                                        f"eval_data.csv"
                    )
                    for method in AACO_METHODS
                    for dataset in DATASETS
                    for dataset_instance_idx in DATASET_INSTANCE_INDICES
                    for cost_param in COST_PARAMS[dataset]
                ]
                if SOFT_BUDGET_EVAL
                else []
            )
        )
    output:
        "extra/output/merged_eval_results/AACO.csv",
    shell:
        """
            csvstack {input} > {output}
        """


rule plot:
    """Generate plots from merged evaluation results."""
    input:
        "extra/output/merged_eval_results/AACO.csv",
    output:
        directory("extra/output/plot_results/AACO"),
    shell:
        """
        python scripts/plotting/plot_eval.py {input} {output}
        """
