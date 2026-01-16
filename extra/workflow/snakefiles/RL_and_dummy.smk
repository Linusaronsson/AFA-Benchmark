import os
import time
from datetime import datetime

NO_PRETRAIN_STR = "NO_PRETRAIN"

DATASET_INSTANCE_INDICES = config.get("dataset_instance_indices", (0, 1))
INITIALIZER = config.get("initializer", "cold")
EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")  # whether to use "val" or "test" during evaluation
DEVICE = config.get("device", "cpu")
USE_WANDB = config.get("use_wandb", False)
SMOKE_TEST = config.get("smoke_test", False)

METHOD_OPTIONS = config.get("method_options", None) # mapping method->options
if METHOD_OPTIONS is None:
    raise ValueError("Expected method_options to be provided.")
METHODS = config.get("methods", []) # a list, allowing is to run experiments on a subset of methods if desired
if METHODS is None:
    raise ValueError("Expected methods to be provided.")
METHODS_WITH_PRETRAINING_STAGE = [
    method
    for method, options in METHOD_OPTIONS.items()
    if options["has_pretraining_stage"] and method in METHODS
]
METHODS_WITHOUT_PRETRAINING_STAGE = [
    method
    for method, options in METHOD_OPTIONS.items()
    if not options["has_pretraining_stage"] and method in METHODS
]
DATASETS = config.get("datasets", None) # a list, allowing us to run experiments on a subset of datasets if desired
if DATASETS is None:
    raise ValueError("Expected datasets to be provided.")
UNMASKERS_RAW = config.get("unmaskers", None) # mapping dataset->unmasker
if UNMASKERS_RAW is None:
    raise ValueError("Expected unmaskers to be provided.")
# Fill in missing datasets with the default value
UNMASKERS = UNMASKERS_RAW | {dataset: UNMASKERS_RAW["default"] for dataset in DATASETS if dataset not in UNMASKERS_RAW}
HARD_BUDGETS_RAW = config.get("hard_budgets", None) # mapping dataset->list of hard budgets
if HARD_BUDGETS_RAW is None:
    raise ValueError("Expected hard_budgets to be provided.")
# Fill in missing datasets with the default value
HARD_BUDGETS = HARD_BUDGETS_RAW | {dataset: HARD_BUDGETS_RAW["default"] for dataset in DATASETS if dataset not in HARD_BUDGETS_RAW}
SOFT_BUDGET_PARAMS_RAW = config.get("soft_budget_params", None) # mapping method->dataset->list of soft budget params
if SOFT_BUDGET_PARAMS_RAW is None:
    raise ValueError("Expected soft_budget_params to be provided.")
# Fill in missing datasets with the default value
SOFT_BUDGET_PARAMS = {
    method: (
        method_soft_budget_params
        | {dataset: method_soft_budget_params["default"] for dataset in DATASETS if dataset not in method_soft_budget_params}
    )
    for method, method_soft_budget_params in SOFT_BUDGET_PARAMS_RAW.items()
}

# RL and dummy methods are either trained with a hard budget or a soft budget, but not both.
# Note how we use "null" instead of None, since it will be passed to hydra
HARD_BUDGET_AND_SOFT_BUDGET_PARAMS = {
    method: {
        dataset: [(hard_budget, "null") for hard_budget in HARD_BUDGETS[dataset]] + [("null", soft_budget_param) for soft_budget_param in SOFT_BUDGET_PARAMS[method][dataset]]
        for dataset in DATASETS
    }
    for method in METHODS
}

rule all:
    input:
        "extra/output/plot_results/eval_perf/RL_and_dummy",
        "extra/output/plot_results/time/RL_and_dummy",

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



rule pretrain_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/pretrained_models/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "model.bundle"
        ),

        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "pretrain_time.txt"

    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/pretrain/{wildcards.method}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            device={DEVICE} \
            seed={wildcards.pretrain_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """

rule train_method_with_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",

        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "model.bundle"
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    "pretrain_seed-{pretrain_seed}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{wildcards.method}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            pretrained_model_bundle_path={input[2]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """

rule train_method_without_pretrained_model:
    input:
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/train.bundle",
        "extra/output/datasets/{dataset}/{dataset_instance_idx}/val.bundle",
    output:
        directory(
            "extra/output/trained_methods/{method}/"
                "dataset-{dataset}+"
                "instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        "train_seed-{train_seed}+"
                        "train_hard_budget-{train_hard_budget}+"
                        "train_soft_budget_param-{train_soft_budget_param}/"
                            "method.bundle"
        ),
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt"
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/train/{wildcards.method}.py \
            train_dataset_bundle_path={input[0]} \
            val_dataset_bundle_path={input[1]} \
            save_path={output[0]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_hard_budget} \
            soft_budget_param={wildcards.train_soft_budget_param} \
            device={DEVICE} \
            seed={wildcards.train_seed} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST} \
            experiment@_global_={wildcards.dataset}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """

# rule train_classifier:
#     input:
#         "extra/data/datasets/{dataset}/{dataset_instance_idx}",
#     output:
#         directory(
#             "extra/trained_classifiers/masked_mlp_classifier/dataset-{dataset}+instance_idx-{dataset_instance_idx}"
#         ),
#     shell:
#         """
#         python scripts/train/masked_mlp_classifier.py \
#             dataset_artifact_path={input} \
#         """

rule eval_method:
    input:
        f"extra/output/datasets/{{dataset}}/{{dataset_instance_idx}}/{EVAL_DATASET_SPLIT}.bundle",

        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "method.bundle",
        # "extra/trained_classifiers/masked_mlp_classifier/dataset-{dataset}+instance-{dataset_instance}",
    output:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
        "extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_time.txt",
    params:
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
    resources:
        shell_exec="bash"
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output[0]} \
            classifier_bundle_path=null \
            seed={wildcards.eval_seed} \
            device={DEVICE} \
            hard_budget={wildcards.eval_hard_budget} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """

# --- RULES TO PRODUCE EVAL PLOT ---

# Add eval_soft_budget_param column. Not used by dummy methods, but kept for consistency.
rule add_eval_metadata_to_eval_data:
    input:
        "extra/output/eval_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col eval_soft_budget_param=""
        """

# Convert the `prev_selections_performed` (list[int]) and `selection_performed` columns into `selections_performed` (int)
rule count_selections:
    input:
        "extra/output/eval_results2/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data.py count_selections {input} {output}
        """

# Add some metadata columns from training
rule add_train_metadata_to_eval_data:
    input:
        "extra/output/eval_results3/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data.py add_metadata {input} {output} \
            --col afa_method={wildcards.method} \
            --col dataset={wildcards.dataset} \
            --col train_seed={wildcards.train_seed} \
            --col train_hard_budget={wildcards.train_hard_budget} \
            --col train_soft_budget_param={wildcards.train_soft_budget_param}
        """

# Make sure that train_hard_budget and eval_hard_budget are the same
# Rename this to hard_budget
# Make sure that only one of train_soft_budget_param and eval_soft_budget_param is set, if any
# Rename this to soft_budget_param
rule validate_hard_budget_and_soft_budget_param:
    input:
        "extra/output/eval_results4/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results5/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data.py validate_budgets {input} {output}
        """

# Instead of two separate classifier columns, the plotting script expects tidy data
rule pivot_long_classifier:
    input:
        "extra/output/eval_results5/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    output:
        "extra/output/eval_results6/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "{pretrain_folder}"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_data.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/misc/transform_eval_data.py pivot_long_classifier {input} {output}
        """

rule merge_eval_perf:
    input:
        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{hard_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                hard_budget,
                soft_budget_param,
            ) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results6/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{hard_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                hard_budget,
                soft_budget_param,
            ) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ]
    resources:
        shell_exec="bash"
    output:
        "extra/output/merged_results/RL_and_dummy_eval_perf.csv",
    shell:
        """
            csvstack {input} > {output}
        """


rule plot_eval:
    input:
        "extra/output/merged_results/RL_and_dummy_eval_perf.csv",
    output:
        directory("extra/output/plot_results/eval_perf/RL_and_dummy"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval.py -i {input} {output}
        """

# --- RULES TO PRODUCE TIME PLOT ---


# Combines txt files containing separate pretrain, train and eval times into a single dataframe.
rule time_df_with_pretrain:
    input:
        "extra/output/pretrained_models/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "pretrain_time.txt",
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt",
        "extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_time.txt"
    output:
        "extra/output/combined_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "combined_time.csv"
    resources:
        shell_exec="nu"
    shell:
        """
        {{afa_method: {wildcards.method}, dataset: {wildcards.dataset}, time_pretrain: (open {input[0]}), time_train: (open {input[1]}), time_eval: (open {input[2]})}} | save {output}
        """

# Combines csv files containing separate pretrain, train and eval times into a single dataframe. Pretrain time is set to null.
rule time_df_without_pretrain:
    input:
        "extra/output/trained_methods/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt",
        "extra/output/eval_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "eval_time.txt"
    output:
        "extra/output/combined_time_results/{method}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}/"
                            "combined_time.csv"
    resources:
        shell_exec="nu"
    shell:
        """
        {{afa_method: {wildcards.method}, dataset: {wildcards.dataset}, time_pretrain: null, time_train: (open {input[0]}), time_eval: (open {input[1]})}} | save {output}
        """


rule merge_time:
    # Inputs are expected to have columns afa_method, dataset, pretrain_time, train_time, eval_time
    input:
        [
            (
                f"extra/output/combined_time_results/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{hard_budget}/"
                                    f"combined_time.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                hard_budget,
                soft_budget_param,
            ) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/combined_time_results/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{hard_budget}+"
                            f"train_soft_budget_param-{soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{hard_budget}/"
                                    f"combined_time.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                hard_budget,
                soft_budget_param,
            ) in HARD_BUDGET_AND_SOFT_BUDGET_PARAMS[method][dataset]
        ]
    output:
        "extra/output/merged_results/RL_and_dummy_time.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        csvstack {input} > {output}
        """

rule plot_time:
    input:
        "extra/output/merged_results/RL_and_dummy_time.csv",
    output:
        directory("extra/output/plot_results/time/RL_and_dummy"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_total_time.py -i {input} {output}
        """
