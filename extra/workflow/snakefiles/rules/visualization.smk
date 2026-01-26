"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""


rule plot_eval:
    """Generate evaluation performance plots."""
    input:
        "extra/output/merged_results/eval_perf/{method_set}+classifier_type-{classifier_type}.csv",
    output:
        directory("extra/output/plot_results/eval_perf/{method_set}+classifier_type-{classifier_type}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval.py {input} {output}
        """


rule plot_time:
    """Generate timing analysis plots."""
    input:
        "extra/output/merged_results/all.csv",
    output:
        directory("extra/output/plot_results/time/all"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_total_time.py -i {input} {output}
        """
