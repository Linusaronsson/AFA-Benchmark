"""
Data aggregation and result combination rules.

Combines individual results into unified datasets:
- Merging evaluation performance data
- Combining time measurements with and without pretraining
- Merging time measurements across all runs
- Collecting every job record into the job duration table
"""

from afabench.core.output_layout import (
    EvaluationRun,
    TrainingRun,
    pretrain_seed_folder,
)


rule merge_eval_perf:
    """Merge evaluation performance results from all methods within a method set.

    Reference methods' tables are restored files that no rule produces, so a
    missing one fails the plan instead of scheduling its production.
    """
    input: lambda wc:
        [
            OUTPUT_LAYOUT.transformed_evaluation_table(
                WORKFLOW_SETTINGS.evaluation_run(
                    method=method,
                    dataset=dataset,
                    dataset_realization_index=dataset_realization_index,
                    budget_combination=budget_combination,
                )
            )
            for method in METHOD_SETS[wc.method_set]
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for budget_combination in BUDGET_PARAMS[method][dataset]
        ]
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "merge_eval_perf", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "merge_eval_perf"),
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
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "split_by_classifier_type", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "split_by_classifier_type"),
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
        lambda wildcards: OUTPUT_LAYOUT.pretrain_time(
            pretrained_model_name=METHOD_TO_PRETRAINED_MODEL[wildcards.method],
            dataset=wildcards.dataset,
            dataset_realization_index=wildcards.dataset_realization_index,
            pretrain_seed=wildcards.pretrain_seed,
        ),
        OUTPUT_LAYOUT.train_time(
            TrainingRun.wildcards(pretrain_folder=pretrain_seed_folder("{pretrain_seed}"))
        ),
        OUTPUT_LAYOUT.eval_time(
            EvaluationRun.wildcards(pretrain_folder=pretrain_seed_folder("{pretrain_seed}"))
        ),
    output:
        OUTPUT_LAYOUT.combined_time(
            EvaluationRun.wildcards(pretrain_folder=pretrain_seed_folder("{pretrain_seed}"))
        ),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "time_df_with_pretrain", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "time_df_with_pretrain"),
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
        OUTPUT_LAYOUT.train_time(
            TrainingRun.wildcards(pretrain_folder=pretrain_seed_folder(None))
        ),
        OUTPUT_LAYOUT.eval_time(
            EvaluationRun.wildcards(pretrain_folder=pretrain_seed_folder(None))
        ),
    output:
        OUTPUT_LAYOUT.combined_time(
            EvaluationRun.wildcards(pretrain_folder=pretrain_seed_folder(None))
        ),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "time_df_without_pretrain", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "time_df_without_pretrain"),
    shell:
        """
        python scripts/misc/merge_time_results.py \
            --output_path {output} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --time_train_path {input[0]} \
            --time_eval_path {input[1]}
        """


rule collect_job_records:
    """Collect every job record under the output root into the job duration table.

    The selected methods' transformed evaluation tables are inputs only so
    that the table is written after every job of the run. The script reads
    every job record under extra/output, including failed and timed-out
    attempts in failed_job_records/ and records of jobs outside this
    invocation. Records are not inputs: a missing one, for example of a
    downloaded shared prerequisite, must not rerun its job.
    """
    input:
        [
            OUTPUT_LAYOUT.transformed_evaluation_table(
                WORKFLOW_SETTINGS.evaluation_run(
                    method=method,
                    dataset=dataset,
                    dataset_realization_index=dataset_realization_index,
                    budget_combination=budget_combination,
                )
            )
            for method in METHODS
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for budget_combination in BUDGET_PARAMS[method][dataset]
        ]
    output:
        "extra/output/merged_results/job_duration_table.parquet",
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "collect_job_records", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "collect_job_records"),
    shell:
        """
        python scripts/misc/collect_job_records.py \
            --output-root extra/output \
            --output {output}
        """


rule merge_time:
    """Merge time measurements from all method-dataset combinations."""
    input:
        [
            OUTPUT_LAYOUT.combined_time(
                WORKFLOW_SETTINGS.evaluation_run(
                    method=method,
                    dataset=dataset,
                    dataset_realization_index=dataset_realization_index,
                    budget_combination=budget_combination,
                )
            )
            for method in METHODS
            for dataset in DATASETS
            for dataset_realization_index in DATASET_REALIZATION_INDICES
            for budget_combination in BUDGET_PARAMS[method][dataset]
        ]
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/all.parquet",
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "merge_time", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "merge_time"),
    shell:
        """
        python scripts/misc/merge_dataframes.py {input} --output {output}
        """
