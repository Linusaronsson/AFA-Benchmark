"""
Evaluation rules.

Handles evaluation of trained methods on test/validation datasets.
"""

from afabench.core.output_layout import EvaluationRun, TrainingRun


rule eval_method:
    input:
        release_dataset_input(
            OUTPUT_LAYOUT.raw_evaluation_table(EvaluationRun.wildcards()),
            OUTPUT_LAYOUT.dataset_bundle(
                dataset="{dataset}",
                dataset_realization_index="{dataset_realization_index}",
                split=EVAL_DATASET_SPLIT,
            ),
            "eval_dataset",
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
        job_record=OUTPUT_LAYOUT.evaluation_job_record(EvaluationRun.wildcards()),
    params:
        wandb=WANDB_RUNS.param("eval_method", lambda wc: wc["method"]),
        device=lambda wildcards, resources: EXECUTION.checked_device("evaluation", wildcards.method, resources),
        job_record=JOB_RECORDS.param(
            "evaluation",
            lambda wc: wc.method,
            name=lambda wc: wc.method,
            eval_batch_size=lambda wc: WORKFLOW_SETTINGS.eval_batch_size(wc.method, wc.dataset),
        ),
        unmasker=lambda wildcards: WORKFLOW_SETTINGS.unmasker(wildcards.dataset),
        eval_batch_size=lambda wildcards: WORKFLOW_SETTINGS.eval_batch_size(wildcards.method, wildcards.dataset),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("evaluation", lambda wc: wc.method),
        **EXECUTION.walltime.resources("eval_method"),
    shell:
        """
        {params.wandb}{params.job_record} \
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
