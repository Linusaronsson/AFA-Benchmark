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
        execution_device=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_perf", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "visualization", "plot_eval_perf")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "visualization", "plot_eval_perf")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "visualization", "plot_eval_perf"),
        gres=lambda wc: EXECUTION.resource("gres", "visualization", "plot_eval_perf"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "visualization", "plot_eval_perf"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "visualization", "plot_eval_perf"),
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
        execution_device=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_eval_actions", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "visualization", "plot_eval_actions")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "visualization", "plot_eval_actions")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "visualization", "plot_eval_actions"),
        gres=lambda wc: EXECUTION.resource("gres", "visualization", "plot_eval_actions"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "visualization", "plot_eval_actions"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "visualization", "plot_eval_actions"),
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
        execution_device=lambda wc, resources: EXECUTION.checked_device("visualization", "plot_time", resources),
    resources:
        shell_exec="bash",
        slurm_partition=(lambda wc: EXECUTION.resource("slurm_partition", "visualization", "plot_time")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_partition", ""),
        slurm_account=(lambda wc: EXECUTION.resource("slurm_account", "visualization", "plot_time")) if EXECUTION.site else DEFAULT_RESOURCES.get("slurm_account", ""),
        gpu=lambda wc: EXECUTION.resource("gpu", "visualization", "plot_time"),
        gres=lambda wc: EXECUTION.resource("gres", "visualization", "plot_time"),
        gpu_model=lambda wc: EXECUTION.resource("gpu_model", "visualization", "plot_time"),
        slurm_extra=lambda wc: EXECUTION.resource("slurm_extra", "visualization", "plot_time"),
    shell:
        """
        python scripts/plotting/plot_total_time.py \
            input={input} output_folder={output} formats='[pdf,svg]'
        """
