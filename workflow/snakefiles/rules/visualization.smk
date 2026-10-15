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


def _dataset_sets_override() -> str:
    """The Hydra override passing the run's dataset sets to plot_eval_perf."""
    dataset_sets = ",".join(
        f"{name}:[{','.join(datasets)}]"
        for name, datasets in WORKFLOW_SETTINGS.dataset_sets.items()
    )
    # Hydra rejects new keys in a dict without the force-add prefix.
    return shlex.quote(f"++dataset_sets={{{dataset_sets}}}")


def _plotting_metric_overrides() -> str:
    """Merge headline metric choices into the script's display defaults."""
    plotting = WORKFLOW_SETTINGS.plotting
    overrides = []
    if plotting.dataset_metrics:
        metrics = ",".join(
            f"{dataset}:{metric}"
            for dataset, metric in plotting.dataset_metrics.items()
        )
        overrides.append(f"++plotting.dataset_metrics={{{metrics}}}")
    if plotting.default_metric is not None:
        overrides.append(f"plotting.default_metric={plotting.default_metric}")
    return shlex.join(overrides)


rule plot_eval_perf:
    """Generate evaluation performance plots, one folder per dataset set."""
    input:
        OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type="{classifier_type}"),
    output:
        directory(OUTPUT_LAYOUT.eval_perf_plots(method_set="{method_set}", classifier_type="{classifier_type}")),
    wildcard_constraints:
        classifier_type="builtin" if WORKFLOW_SETTINGS.per_dataset_plots else "builtin|external",
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_perf", resources),
        dataset_sets=_dataset_sets_override(),
        metrics=_plotting_metric_overrides(),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_perf"),
        **EXECUTION.walltime.resources("plot_eval_perf"),
    shell:
        """
        python scripts/plotting/plot_eval_perf.py \
            input={input} output_folder={output} formats='[pdf,svg]' \
            {params.dataset_sets} {params.metrics}
        """

if WORKFLOW_SETTINGS.per_dataset_plots:
    rule plot_eval_perf_per_dataset:
        """The external usual job also owns the additional method-set figures."""
        input:
            selected=OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type="external"),
            unsplit=OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type=None),
        output:
            usual=directory(OUTPUT_LAYOUT.eval_perf_plots(method_set="{method_set}", classifier_type="external")),
            per_dataset=directory(OUTPUT_LAYOUT.per_dataset_eval_perf_plots(method_set="{method_set}")),
        params:
            allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_perf", resources),
            dataset_sets=_dataset_sets_override(),
            metrics=_plotting_metric_overrides(),
        resources:
            shell_exec="bash",
            **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_perf"),
            **EXECUTION.walltime.resources("plot_eval_perf"),
        shell:
            """
            python scripts/plotting/plot_eval_perf.py \
                input={input.selected} output_folder={output.usual} formats='[pdf,svg]' \
                {params.dataset_sets} {params.metrics} per_dataset_plots=true \
                per_dataset_input={input.unsplit} \
                per_dataset_output_folder={output.per_dataset}
            """

# This rule probably does not need to use both types of classifiers, since the actions are the same (they come from the same original evaluation dataframe).
rule plot_eval_actions:
    input:
        OUTPUT_LAYOUT.merged_table(method_set="{method_set}", classifier_type="{classifier_type}"),
    output:
        directory(OUTPUT_LAYOUT.eval_actions_plots(method_set="{method_set}", classifier_type="{classifier_type}")),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_actions", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_actions"),
        **EXECUTION.walltime.resources("plot_eval_actions"),
    shell:
        """
        python scripts/plotting/plot_eval_actions.py \
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
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_time", resources),
        selection=_plot_time_selection(),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_time"),
        **EXECUTION.walltime.resources("plot_time"),
    shell:
        """
        python scripts/plotting/plot_total_time.py \
            input={input} output_folder={output} {params.selection} \
            formats='[pdf,svg]'
        """
