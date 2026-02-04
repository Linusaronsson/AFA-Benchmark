"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""


rule plot_eval_perf:
    """Generate evaluation performance plots."""
    input:
        "extra/output/merged_results/eval_perf/{method_set}+classifier_type-{classifier_type}.parquet",
    output:
        directory("extra/output/plot_results/eval_perf/{method_set}+classifier_type-{classifier_type}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_perf.py {input} {output}
        """

# This rule probably does not need to use both types of classifiers, since the actions are the same (they come from the same original evaluation dataframe).
rule plot_eval_actions:
    input:
        "extra/output/merged_results/eval_perf/{method_set}+classifier_type-{classifier_type}.parquet",
    output:
        directory("extra/output/plot_results/eval_actions/{method_set}+classifier_type-{classifier_type}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_actions.py {input} {output}
        """


rule plot_time:
    """Generate timing analysis plots."""
    input:
        "extra/output/merged_results/time/all.parquet",
    output:
        directory("extra/output/plot_results/time/"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_total_time.py -i {input} {output}
        """
