"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""


rule plot_eval_perf:
    """Generate evaluation performance plots."""
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}"),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_perf", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_perf"),
    shell:
        """
        python scripts/plotting/plot_eval_perf.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """

# This rule probably does not need to use both types of classifiers, since the actions are the same (they come from the same original evaluation dataframe).
rule plot_eval_actions:
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_perf/{{method_set}}+classifier_type-{{classifier_type}}.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/eval_actions/{{method_set}}+classifier_type-{{classifier_type}}"),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_actions", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_eval_actions"),
    shell:
        """
        python scripts/plotting/plot_eval_actions.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """


rule plot_time:
    """Generate timing analysis plots."""
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/all.parquet",
    output:
        directory(f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{INITIALIZER_TAG}/time/"),
    params:
        allocation_check=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_time", resources),
    resources:
        shell_exec="bash",
        **EXECUTION.allocation_resources("visualization", lambda wc: "plot_time"),
    shell:
        """
        python scripts/plotting/plot_total_time.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """
