"""Figures and tables from results that already exist.

This variant excludes dataset generation, classifier training, view
materialisation, restoration, method training, evaluation, and summarisation.
The four summary CSVs and the route descriptors are inputs, not products, so a
missing result fails immediately instead of scheduling the whole study.

Example:
    uv run snakemake \
      --profile extra/workflow/profiles/config/missing_data_plots --cores 4
"""

BUILD_EXPERIMENTS = False


include: "../rules/missing_data_common.smk"
include: "../rules/missing_data_results.smk"


localrules:
    plots,
    plot_missing_data,


rule plots:
    input:
        FIGURE_DIR,
        PAPER_OUTPUTS,
