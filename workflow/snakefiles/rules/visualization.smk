"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""

import shlex


def _plot_time_selection() -> str:
    """Hydra overrides selecting this invocation's jobs from the job duration table."""
    pretrained_models = ",".join(
        f"{method}:{pretrained_model}"
        for method, pretrained_model in METHOD_TO_PRETRAINED_MODEL.items()
    )
    return shlex.join(
        [
            f"methods=[{','.join(METHODS)}]",
            # Hydra rejects new keys in a dict without the force-add prefix.
            f"++pretrained_models={{{pretrained_models}}}",
            f"initializer_tag={INITIALIZER_TAG}",
            f"eval_dataset_split={EVAL_DATASET_SPLIT}",
        ]
    )


rule plot_eval_perf:
    """Generate evaluation performance plots."""
    input:
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}"),
    params:
        image=IMAGES.param("visualization", lambda wc: "plot_eval_perf"),
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_perf", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_perf"),
    shell:
        """
        {params.image}python scripts/plotting/plot_eval_perf.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """

# This rule probably does not need to use both types of classifiers, since the actions are the same (they come from the same original evaluation dataframe).
rule plot_eval_actions:
    input:
        f"{OUTPUT_ROOT}/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/{{method_set}}+classifier_type-{{classifier_type}}"),
    params:
        image=IMAGES.param("visualization", lambda wc: "plot_eval_actions"),
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_actions", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_actions"),
    shell:
        """
        {params.image}python scripts/plotting/plot_eval_actions.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """


rule plot_time:
    """Plot the job durations of the selected methods from the job duration table.

    The table holds every job record under the output root; the script keeps
    the completed jobs of `methods` under this initializer and evaluation
    split.
    """
    input:
        f"{OUTPUT_ROOT}/merged_results/job_duration_table.parquet",
    output:
        directory(f"{OUTPUT_ROOT}/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"),
    params:
        image=IMAGES.param("visualization", lambda wc: "plot_time"),
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_time", resources),
        selection=_plot_time_selection(),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_time"),
    shell:
        """
        {params.image}python scripts/plotting/plot_total_time.py \
            input={input} output_folder={output} {params.selection} \
            formats='[pdf,svg]'
        """
