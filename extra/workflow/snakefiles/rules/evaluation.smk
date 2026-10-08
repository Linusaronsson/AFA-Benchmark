"""
Evaluation rules.

Handles evaluation of trained methods on test/validation datasets.
"""

from afabench.core.output_layout import EvaluationRun, TrainingRun
from job_records import job_record_command


rule eval_method:
    input:
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split=EVAL_DATASET_SPLIT,
        ),
        OUTPUT_LAYOUT.method_bundle(TrainingRun.wildcards()),
        lambda wildcards: WORKFLOW_SETTINGS.classifier_bundle(
            OUTPUT_LAYOUT,
            method=wildcards.method,
            dataset=wildcards.dataset,
            dataset_realization_index=wildcards.dataset_realization_index,
        ),
    output:
        eval_table=OUTPUT_LAYOUT.raw_evaluation_table(EvaluationRun.wildcards()),
        eval_time=OUTPUT_LAYOUT.eval_time(EvaluationRun.wildcards()),
        job_record=OUTPUT_LAYOUT.evaluation_job_record(EvaluationRun.wildcards()),
    params:
        device=lambda wildcards, resources: EXECUTION.checked_device("evaluation", wildcards.method, resources),
        job_record=lambda wc, output, resources, threads: job_record_command(
            output.job_record,
            stage="evaluation",
            wildcards=wc,
            device=EXECUTION.checked_device("evaluation", wc.method, resources),
            resources=resources,
            threads=threads,
            smoke_test=SMOKE_TEST,
            time_file=output.eval_time,
            name=wc.method,
            eval_batch_size=EVAL_BATCH_SIZES[wc.method][wc.dataset],
        ),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        eval_batch_size=lambda wildcards: EVAL_BATCH_SIZES[wildcards.method][wildcards.dataset],
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("evaluation", lambda wc: wc.method),
    shell:
        """
        {params.job_record} \
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            initializer={INITIALIZER} \
            unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output.eval_table} \
            classifier_bundle_path={input[2]} \
            seed={wildcards.eval_seed} \
            device={params.device} \
            hard_budget={wildcards.eval_hard_budget} \
            soft_budget_param={wildcards.eval_soft_budget_param} \
            batch_size={params.eval_batch_size} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        """
