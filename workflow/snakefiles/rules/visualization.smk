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
        for method, pretrained_model in WORKFLOW_SETTINGS.method_to_pretrained_model.items()
    )
    return shlex.join(
        [
            f"methods=[{','.join(WORKFLOW_SETTINGS.methods)}]",
            # Hydra rejects new keys in a dict without the force-add prefix.
            f"++pretrained_models={{{pretrained_models}}}",
            f"initializer_tag={OUTPUT_LAYOUT.initializer_tag}",
            f"eval_dataset_split={EVAL_DATASET_SPLIT}",
        ]
    )


rule plot_eval_perf:
    """Generate evaluation performance plots."""
    input:
        OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type="{classifier_type}"),
    output:
        directory(OUTPUT_LAYOUT.eval_perf_plots(method_set="{method_set}", classifier_type="{classifier_type}")),
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
        OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type="{classifier_type}"),
    output:
        directory(OUTPUT_LAYOUT.eval_actions_plots(method_set="{method_set}", classifier_type="{classifier_type}")),
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
        OUTPUT_LAYOUT.job_duration_table(),
    output:
        directory(OUTPUT_LAYOUT.time_plots()),
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
