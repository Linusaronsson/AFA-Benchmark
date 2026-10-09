"""
Data transformation rules for evaluation results.

Handles transformations on evaluation results. Raw evaluation tables carry
their own identity columns and provenance record (ADR 0002); transformation
only derives plotting columns and pivots classifier columns to tidy data
format.
"""

import re

from afabench.core.output_layout import EvaluationRun

# Reference methods' plotting-ready tables are restored from a benchmark
# release. Not matching them here leaves those tables as plain input files,
# so aggregating them never schedules their evaluation or training.
TRANSFORMED_METHOD_PATTERN = (
    "(?!(?:" + "|".join(map(re.escape, REFERENCE_METHODS)) + ")/).+"
    if REFERENCE_METHODS
    else ".+"
)


rule transform_eval_data:
    """Transform raw evaluation data to final format for plotting.

    Identity comes from the raw table's columns, and its provenance record
    is copied into the output. The wildcards passed as arguments are checks:
    one that disagrees with a non-null column fails the job, and a missing
    or null column (a table written before ADR 0002) is filled from its
    wildcard. The script derives n_selections_performed and pivots the
    classifier columns to tidy data format.
    """
    input:
        OUTPUT_LAYOUT.raw_evaluation_table(EvaluationRun.wildcards()),
    output:
        eval_table=OUTPUT_LAYOUT.transformed_evaluation_table(EvaluationRun.wildcards()),
        job_record=OUTPUT_LAYOUT.transformation_job_record(EvaluationRun.wildcards()),
    wildcard_constraints:
        method=TRANSFORMED_METHOD_PATTERN,
    params:
        job_record=JOB_RECORDS.param(
            "transformation",
            lambda wc: "transform_eval_data",
            name=lambda wc: wc.method,
        ),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("transformation", lambda wc: "transform_eval_data"),
    shell:
        """
        {params.job_record} \
        python scripts/misc/transform_eval_data_pipeline.py \
            --input_path {input} \
            --output_path {output.eval_table} \
            --method {wildcards.method} \
            --dataset {wildcards.dataset} \
            --initializer {INITIALIZER} \
            --train_seed {wildcards.train_seed} \
            --train_hard_budget {wildcards.train_hard_budget} \
            --train_soft_budget_param {wildcards.train_soft_budget_param} \
            --eval_soft_budget_param {wildcards.eval_soft_budget_param}
        """
