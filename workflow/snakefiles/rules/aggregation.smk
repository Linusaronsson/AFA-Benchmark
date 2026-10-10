"""
Data aggregation and result combination rules.

Combines individual results into unified datasets:
- Merging evaluation performance data
- Collecting every job record into the job duration table
"""


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
        image=IMAGES.param("aggregation", lambda wc: "merge_eval_perf"),
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "merge_eval_perf", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "merge_eval_perf"),
    output:
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+all.parquet",
    shell:
        """
            {params.image}python scripts/misc/merge_dataframes.py {input} --output {output}
        """

rule split_by_classifier_type:
    input:
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+all.parquet"
    output:
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+classifier_type-builtin.parquet",
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/method_set-{{method_set}}+classifier_type-external.parquet"
    params:
        image=IMAGES.param("aggregation", lambda wc: "split_by_classifier_type"),
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "split_by_classifier_type", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "split_by_classifier_type"),
    shell:
        """
            {params.image}python scripts/misc/split_eval_perf_by_classifier.py \
                --input_path {input} \
                --output_builtin {output[0]} \
                --output_external {output[1]}
        """


rule collect_job_records:
    """Collect every job record under the output root into the job duration table.

    The selected methods' transformed evaluation tables are inputs only so
    that the table is written after every job of the run. The script reads
    every job record under the output root, including failed and timed-out
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
        f"{OUTPUT_ROOT}/merged_results/job_duration_table.parquet",
    params:
        image=IMAGES.param("aggregation", lambda wc: "collect_job_records"),
        output_root=OUTPUT_ROOT,
        allocation_check=lambda wc, resources: EXECUTION.checked_device("aggregation", "collect_job_records", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("aggregation", lambda wc: "collect_job_records"),
    shell:
        """
        {params.image}python scripts/misc/collect_job_records.py \
            --output-root {params.output_root} \
            --output {output}
        """

