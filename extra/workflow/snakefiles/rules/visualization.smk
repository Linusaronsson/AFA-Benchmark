"""
Visualization and plot generation rules.

Generates plots from aggregated results:
- Evaluation performance plots
- Timing analysis plots
"""


rule plot_eval:
    """Generate evaluation performance plots."""
    input:
        "extra/output/merged_results/eval_perf/rl_and_dummy+classifier_type-{classifier_type}.csv",
    output:
        directory("extra/output/plot_results/eval_perf/rl_and_dummy+classifier_type-{classifier_type}"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval.py {input} {output}
        """


rule plot_time:
    """Generate timing analysis plots."""
    input:
        "extra/output/merged_results/rl_and_dummy_time.csv",
    output:
        directory("extra/output/plot_results/time/rl_and_dummy"),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_total_time.py -i {input} {output}
        """
