"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""


def _cube_nm_ar_methods() -> list[str]:
    if "cube_nm_ar" not in DATASETS:
        return []
    return [
        method
        for method in METHODS
        if "cube_nm_ar" in DATASETS_USED_PER_METHOD[method]
    ]


def _cube_nm_ar_budget_tuples(method: str) -> list[tuple]:
    return [
        budget_tuple
        for budget_tuple in BUDGET_PARAMS[method]["cube_nm_ar"]
        if (
            CUBE_NM_AR_BUDGET_MODE == "hard"
            and budget_tuple[1] != "null"
        )
        or (
            CUBE_NM_AR_BUDGET_MODE == "soft"
            and budget_tuple[3] != "null"
        )
    ]


def _cube_nm_ar_eval_inputs(
    *,
    stop_shield_delta: str | None = None,
    dual_lambda: str | None = None,
) -> list[str]:
    methods = _cube_nm_ar_methods()
    if not methods:
        return []

    if stop_shield_delta is not None and dual_lambda is not None:
        raise ValueError(
            "Expected at most one of stop_shield_delta and dual_lambda."
        )

    if stop_shield_delta is None and dual_lambda is None:
        base_root = (
            f"extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{INITIALIZER_TAG}"
        )
    elif stop_shield_delta is not None:
        base_root = (
            "extra/output/eval_results_shielded/"
            f"delta-{stop_shield_delta}/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}"
        )
    else:
        base_root = (
            "extra/output/eval_results_dualized/"
            f"lambda-{dual_lambda}/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}"
        )

    return [
        (
            f"{base_root}/{method}/"
                "dataset-cube_nm_ar+"
                f"instance_idx-{dataset_instance_idx}/"
                    f"pretrain_seed-{dataset_instance_idx}/"
                        f"train_seed-{dataset_instance_idx}+"
                        f"train_hard_budget-{train_hard_budget}+"
                        f"train_soft_budget_param-{train_soft_budget_param}/"
                            f"eval_seed-{dataset_instance_idx}+"
                            f"eval_hard_budget-{eval_hard_budget}+"
                            f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                "eval_data.csv"
        )
        for method in methods
        if method in METHODS_WITH_PRETRAINING_STAGE
        for dataset_instance_idx in DATASET_INSTANCE_INDICES
        for (
            train_hard_budget,
            eval_hard_budget,
            train_soft_budget_param,
            eval_soft_budget_param,
        ) in _cube_nm_ar_budget_tuples(method)
    ] + [
        (
            f"{base_root}/{method}/"
                "dataset-cube_nm_ar+"
                f"instance_idx-{dataset_instance_idx}/"
                    f"{NO_PRETRAIN_STR}/"
                        f"train_seed-{dataset_instance_idx}+"
                        f"train_hard_budget-{train_hard_budget}+"
                        f"train_soft_budget_param-{train_soft_budget_param}/"
                            f"eval_seed-{dataset_instance_idx}+"
                            f"eval_hard_budget-{eval_hard_budget}+"
                            f"eval_soft_budget_param-{eval_soft_budget_param}/"
                                "eval_data.csv"
        )
        for method in methods
        if method in METHODS_WITHOUT_PRETRAINING_STAGE
        for dataset_instance_idx in DATASET_INSTANCE_INDICES
        for (
            train_hard_budget,
            eval_hard_budget,
            train_soft_budget_param,
            eval_soft_budget_param,
        ) in _cube_nm_ar_budget_tuples(method)
    ]


rule plot_eval_perf:
    """Generate evaluation performance plots."""
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_perf.py {input} {output} --formats pdf svg
        """

# This rule probably does not need to use both types of classifiers, since the actions are the same (they come from the same original evaluation dataframe).
rule plot_eval_actions:
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/{{method_set}}+classifier_type-{{classifier_type}}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_actions.py {input} {output} --formats pdf svg
        """


rule plot_time:
    """Generate timing analysis plots."""
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/all.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_total_time.py -i {input} {output} --formats pdf svg
        """


rule plot_cube_nm_ar_unshielded:
    input:
        lambda wildcards: _cube_nm_ar_eval_inputs()
    output:
        directory(
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
            f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/stop_shield-none"
        ),
    params:
        methods=" ".join(_cube_nm_ar_methods()),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_cube_nm_ar_results.py \
            extra/output/eval_results/eval_split-{EVAL_DATASET_SPLIT} \
            {output} \
            --train-initializers {TRAIN_INITIALIZER} \
            --eval-initializers {EVAL_INITIALIZER} \
            --budget-mode {CUBE_NM_AR_BUDGET_MODE} \
            --methods {params.methods}
        """


rule plot_cube_nm_ar_shielded:
    input:
        lambda wildcards: _cube_nm_ar_eval_inputs(
            stop_shield_delta=wildcards.stop_shield_delta
        )
    output:
        directory(
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
            f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/"
            "stop_shield-{stop_shield_delta}"
        ),
    params:
        methods=" ".join(_cube_nm_ar_methods()),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_cube_nm_ar_results.py \
            extra/output/eval_results_shielded/delta-{wildcards.stop_shield_delta}/eval_split-{EVAL_DATASET_SPLIT} \
            {output} \
            --train-initializers {TRAIN_INITIALIZER} \
            --eval-initializers {EVAL_INITIALIZER} \
            --budget-mode {CUBE_NM_AR_BUDGET_MODE} \
            --methods {params.methods}
        """


rule plot_cube_nm_ar_dualized:
    input:
        lambda wildcards: _cube_nm_ar_eval_inputs(
            dual_lambda=wildcards.dual_lambda
        )
    output:
        directory(
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/"
            f"budget_mode-{CUBE_NM_AR_BUDGET_MODE}/"
            "dual_lambda-{dual_lambda}"
        ),
    params:
        methods=" ".join(_cube_nm_ar_methods()),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_cube_nm_ar_results.py \
            extra/output/eval_results_dualized/lambda-{wildcards.dual_lambda}/eval_split-{EVAL_DATASET_SPLIT} \
            {output} \
            --train-initializers {TRAIN_INITIALIZER} \
            --eval-initializers {EVAL_INITIALIZER} \
            --budget-mode {CUBE_NM_AR_BUDGET_MODE} \
            --methods {params.methods}
        """
