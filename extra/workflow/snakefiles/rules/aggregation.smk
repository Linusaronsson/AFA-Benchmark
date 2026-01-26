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
                f"extra/output/eval_results_transformed/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHOD_SETS[wc.method_set] if method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/eval_results_transformed/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}/"
                                    f"eval_data.csv"
            )
            for method in METHOD_SETS[wc.method_set] if method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
    resources:
        shell_exec="bash"
    output:
        "extra/output/merged_results/eval_perf/method_set-{method_set}.csv",
    shell:
        """
            csvstack {input} > {output}
        """

rule split_by_classifier_type:
    input:
        "extra/output/merged_results/eval_perf/method_set-{method_set}.csv"
    output:
        "extra/output/merged_results/eval_perf/method_set-{method_set}+classifier_type-builtin.csv",
        "extra/output/merged_results/eval_perf/method_set-{method_set}+classifier_type-external.csv"
    resources:
        shell_exec="nu"
    shell:
        """
            let df = open {input}
            $df | where {{$in.classifier == builtin}} | reject classifier | save {output[0]}
            $df | where {{$in.classifier == external}} | reject classifier | save {output[1]}
        """


rule time_df_with_pretrain:
    """Combine pretrain, train, and eval time measurements into a single dataframe."""
    input:
        lambda wildcards: (
            f"extra/output/pretrained_models/{METHOD_TO_PRETRAINED_MODEL[wildcards.method]}/"
                f"dataset-{wildcards.dataset}+"
                f"instance_idx-{wildcards.dataset_instance_idx}/"
                    f"pretrain_seed-{wildcards.pretrain_seed}/"
                        "pretrain_time.txt"
        ),
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


rule time_df_without_pretrain:
    """Combine train and eval time measurements, with pretrain time set to null."""
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
    """Merge time measurements from all method-dataset combinations."""
    input:
        [
            (
                f"extra/output/combined_time_results/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"{NO_PRETRAIN_STR}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}/"
                                    f"combined_time.csv"
            )
            for method in METHODS_WITHOUT_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ] +
        [
            (
                f"extra/output/combined_time_results/{method}/"
                    f"dataset-{dataset}+"
                    f"instance_idx-{dataset_instance_idx}/"
                        f"pretrain_seed-{dataset_instance_idx}/"
                            f"train_seed-{dataset_instance_idx}+"
                            f"train_hard_budget-{train_hard_budget}+"
                            f"train_soft_budget_param-{train_soft_budget_param}/"
                                f"eval_seed-{dataset_instance_idx}+"
                                f"eval_hard_budget-{eval_hard_budget}/"
                                    f"combined_time.csv"
            )
            for method in METHODS_WITH_PRETRAINING_STAGE
            for dataset in DATASETS
            for dataset_instance_idx in DATASET_INSTANCE_INDICES
            for (
                train_hard_budget,
                eval_hard_budget,
                train_soft_budget_param,
            ) in BUDGET_PARAMS[method][dataset]
        ]
    output:
        "extra/output/merged_results/all.csv",
    resources:
        shell_exec="bash"
    shell:
        """
        csvstack {input} > {output}
        """
