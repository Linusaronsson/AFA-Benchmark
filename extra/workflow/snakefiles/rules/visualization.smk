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
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_perf.py {input} {output}
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
        python scripts/plotting/plot_eval_actions.py {input} {output}
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
        python scripts/plotting/plot_total_time.py -i {input} {output}
        """
