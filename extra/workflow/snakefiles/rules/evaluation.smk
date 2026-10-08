"""
Evaluation rules.

Handles evaluation of trained methods on test/validation datasets.
"""

from afabench.core.output_layout import EvaluationRun, TrainingRun


rule eval_method:
    input:
        OUTPUT_LAYOUT.dataset_bundle(
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            split=EVAL_DATASET_SPLIT,
        ),
        OUTPUT_LAYOUT.method_bundle(TrainingRun.wildcards()),
        # The method's built-in classifier if it has one, else the external one.
        lambda wildcards: OUTPUT_LAYOUT.classifier_bundle(
            dataset=wildcards.dataset,
            dataset_realization_index=wildcards.dataset_realization_index,
            method=(
                wildcards.method
                if wildcards.method in METHOD_CLASSIFIER_SCRIPT_NAMES
                else None
            ),
        ),
    output:
        OUTPUT_LAYOUT.raw_evaluation_table(EvaluationRun.wildcards()),
        OUTPUT_LAYOUT.eval_time(EvaluationRun.wildcards()),
    params:
        device=lambda wildcards, resources: EXECUTION.checked_device("evaluation", wildcards.method, resources),
        unmasker=lambda wildcards: UNMASKERS[wildcards.dataset],
        eval_batch_size=lambda wildcards: EVAL_BATCH_SIZES[wildcards.method][wildcards.dataset],
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("evaluation", lambda wc: wc.method),
    shell:
        """
        START_TIME=$(date +%s.%N)
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input[1]} \
            initializer={INITIALIZER} \
            unmasker={params.unmasker} \
            dataset_bundle_path={input[0]} \
            save_path={output[0]} \
            classifier_bundle_path={input[2]} \
            seed={wildcards.eval_seed} \
            device={params.device} \
            hard_budget={wildcards.eval_hard_budget} \
            soft_budget_param={wildcards.eval_soft_budget_param} \
            batch_size={params.eval_batch_size} \
            use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST}
        END_TIME=$(date +%s.%N)
        ELAPSED=$(echo "$END_TIME $START_TIME" | awk '{{printf "%.6f", $1 - $2}}')
        echo $ELAPSED > '{output[1]}'
        """
