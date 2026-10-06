"""
Data aggregation and result combination rules.

Combines individual results into unified datasets:
- Merging evaluation performance data
- Combining time measurements with and without pretraining
- Merging time measurements across all runs
"""


rule merge_eval_perf:
    """Merge evaluation performance results from all methods within a method set."""
    input: lambda wc:
        [
            (
                f"extra/output/eval_results_transformed/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.parquet"
            )
            for method in METHOD_SETS[wc.method_set] if method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
                eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results_transformed/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"eval_data.parquet"
            )
            for method in METHOD_SETS[wc.method_set] if method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
                eval_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
    params:
        execution_device=lambda wc, resources: EXECUTION.checked_device("aggregation", "merge_eval_perf", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "aggregation", "merge_eval_perf")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "aggregation", "merge_eval_perf")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "aggregation", "merge_eval_perf"),
        gres=lambda wc: EXECUTION.resource("gres", "aggregation", "merge_eval_perf"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "aggregation", "merge_eval_perf"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "aggregation", "merge_eval_perf"),
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+all.parquet",
    shell:
        """
            python scripts/misc/merge_dataframes.py {input} --output {output}
        """

rule split_by_classifier_type:
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+all.parquet"
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+classifier_type-builtin.parquet",
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+classifier_type-external.parquet"
    params:
        execution_device=lambda wc, resources: EXECUTION.checked_device("aggregation", "split_by_classifier_type", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "aggregation", "split_by_classifier_type")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "aggregation", "split_by_classifier_type")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "aggregation", "split_by_classifier_type"),
        gres=lambda wc: EXECUTION.resource("gres", "aggregation", "split_by_classifier_type"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "aggregation", "split_by_classifier_type"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "aggregation", "split_by_classifier_type"),
    shell:
        """
            python scripts/misc/split_eval_perf_by_classifier.py \
                --input_path {input} \
                --output_builtin {output[0]} \
                --output_external {output[1]}
        """


rule time_df_with_pretrain:
    """Combine pretrain, train, and eval time measurements into a single dataframe."""
    input:
        lambda wildcards: (
            f"extra/output/pretrained_models/{INITIALIZER_TAG}/{METHOD_TO_PRETRAINED_MODEL[wildcards.method]}/"
                f"dataset-{wildcards.dataset}+"
                f"instance_idx-{wildcards.dataset_instance_idx}/"
                    f"pretrain_seed-{wildcards.pretrain_seed}/"
                        "pretrain_time.txt"
        ),
        f"extra/output/trained_methods/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt",
        f"extra/output/eval_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_time.txt"
    output:
        f"extra/output/combined_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                "pretrain_seed-{pretrain_seed}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "combined_time.parquet"
    params:
        execution_device=lambda wc, resources: EXECUTION.checked_device("aggregation", "time_df_with_pretrain", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "aggregation", "time_df_with_pretrain")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "aggregation", "time_df_with_pretrain")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "aggregation", "time_df_with_pretrain"),
        gres=lambda wc: EXECUTION.resource("gres", "aggregation", "time_df_with_pretrain"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "aggregation", "time_df_with_pretrain"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "aggregation", "time_df_with_pretrain"),
    shell:
        """
        python scripts/misc/merge_time_results.py \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --time_pretrain_path {input[0]} \
            --time_train_path {input[1]} \
            --time_eval_path {input[2]}
        """


rule time_df_without_pretrain:
    """Combine train and eval time measurements, with pretrain time set to null."""
    input:
        f"extra/output/trained_methods/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "train_time.txt",
        f"extra/output/eval_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "eval_time.txt"
    output:
        f"extra/output/combined_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{{method}}/"
            "dataset-{dataset}+"
            "instance_idx-{dataset_instance_idx}/"
                f"{NO_PRETRAIN_STR}/"
                    "train_seed-{train_seed}+"
                    "train_hard_budget-{train_hard_budget}+"
                    "train_soft_budget_param-{train_soft_budget_param}/"
                        "eval_seed-{eval_seed}+"
                        "eval_hard_budget-{eval_hard_budget}+"
                        "eval_soft_budget_param-{eval_soft_budget_param}/"
                            "combined_time.parquet"
    params:
        execution_device=lambda wc, resources: EXECUTION.checked_device("aggregation", "time_df_without_pretrain", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "aggregation", "time_df_without_pretrain")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "aggregation", "time_df_without_pretrain")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "aggregation", "time_df_without_pretrain"),
        gres=lambda wc: EXECUTION.resource("gres", "aggregation", "time_df_without_pretrain"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "aggregation", "time_df_without_pretrain"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "aggregation", "time_df_without_pretrain"),
    shell:
        """
        python scripts/misc/merge_time_results.py \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --time_train_path {input[0]} \
            --time_eval_path {input[1]}
        """


rule merge_time:
    """Merge time measurements from all method-dataset combinations."""
    input:
        [
            (
                f"extra/output/combined_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"combined_time.parquet"
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
        ] +
        [
            (
                f"extra/output/combined_time_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}+"
                                f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                    f"combined_time.parquet"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
                eval_soft_budget_param
            ) in BUDGET_PARAMS[method][dataset]
        ]
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/all.parquet",
    params:
        execution_device=lambda wc, resources: EXECUTION.checked_device("aggregation", "merge_time", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "aggregation", "merge_time")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "aggregation", "merge_time")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "aggregation", "merge_time"),
        gres=lambda wc: EXECUTION.resource("gres", "aggregation", "merge_time"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "aggregation", "merge_time"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "aggregation", "merge_time"),
    shell:
        """
        python scripts/misc/merge_dataframes.py {input} --output {output}
        """
