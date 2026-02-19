"""
Compare evaluation performance across initializers.

Runtime config (--config / --configfile):
    initializers (list[str], required):
        Initializer names without prefix (e.g. ["missingness",
        "missingness_all_observed"]).
    method_sets (dict[str, list[str]], required):
        Method sets; only keys are used to locate merged method-set artifacts.
    eval_dataset_split (str, default="val"):
        Evaluation split used in upstream pipeline outputs.
    classifier_types (list[str], default=["external"]):
        Which classifier-specific merged parquet files to compare.
    pipeline_configfiles (list[str], optional):
        If provided, this workflow first materializes required upstream
        `pipeline.smk` merged parquet files for each initializer using these
        config files.
    pipeline_cores (int, default=8):
        Cores used by nested `pipeline.smk` invocations when
        `pipeline_configfiles` is provided.
    pipeline_rerun_incomplete (bool, default=True):
        Whether to pass `--rerun-incomplete` to nested `pipeline.smk`
        invocations.
    pipeline_extra_config (str, default=""):
        Optional raw `--config` key-value overrides appended to nested
        `pipeline.smk` invocations.

Input expectations:
    This workflow expects upstream outputs from pipeline.smk for each
    initializer at:
      extra/output/merged_results/eval_split-<split>/initializer-<initializer>/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>.parquet

Outputs:
    Merged cross-initializer parquet:
      extra/output/merged_results/eval_split-<split>/initializer-comparison/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>.parquet
    Plots:
      extra/output/plot_results/eval_split-<split>/initializer-comparison/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>/
"""

EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")
INITIALIZERS = config.get("initializers", None)
METHOD_SETS = config.get("method_sets", None)
CLASSIFIER_TYPES = config.get("classifier_types", ["external"])
PIPELINE_CONFIGFILES = config.get("pipeline_configfiles", [])
PIPELINE_CORES = config.get("pipeline_cores", 8)
PIPELINE_RERUN_INCOMPLETE = config.get("pipeline_rerun_incomplete", True)
PIPELINE_EXTRA_CONFIG = config.get("pipeline_extra_config", "")

if INITIALIZERS is None:
    raise ValueError("Expected 'initializers' in config.")
if METHOD_SETS is None:
    raise ValueError("Expected 'method_sets' in config.")

METHOD_SET_NAMES = list(METHOD_SETS.keys())


def _upstream_eval_perf_paths():
    return [
        (
            f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"initializer-{initializer}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}.parquet"
        )
        for initializer in INITIALIZERS
        for method_set in METHOD_SET_NAMES
        for classifier_type in CLASSIFIER_TYPES
    ]


rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-comparison/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}"
            for method_set in METHOD_SET_NAMES
            for classifier_type in CLASSIFIER_TYPES
        ]


if PIPELINE_CONFIGFILES:
    rule materialize_upstream_eval_perf:
        output:
            _upstream_eval_perf_paths()
        params:
            initializers=" ".join(INITIALIZERS),
            method_sets=" ".join(METHOD_SET_NAMES),
            classifier_types=" ".join(CLASSIFIER_TYPES),
            configfiles=" ".join(PIPELINE_CONFIGFILES),
            rerun_incomplete=(
                "--rerun-incomplete"
                if PIPELINE_RERUN_INCOMPLETE
                else ""
            ),
            pipeline_extra_config=PIPELINE_EXTRA_CONFIG,
            pipeline_cores=PIPELINE_CORES,
        resources:
            shell_exec="bash"
        shell:
            """
            set -euo pipefail
            for initializer in {params.initializers}; do
              for method_set in {params.method_sets}; do
                for classifier_type in {params.classifier_types}; do
                  uv run snakemake \
                    -s extra/workflow/snakefiles/orchestration/pipeline.smk \
                    {params.rerun_incomplete} \
                    --nolock \
                    --configfile {params.configfiles} \
                    --config initializer="${{initializer}}" \
                             eval_dataset_split="{EVAL_DATASET_SPLIT}" \
                             {params.pipeline_extra_config} \
                    --cores {params.pipeline_cores} \
                    extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-${{initializer}}/eval_perf/method_set-${{method_set}}+classifier_type-${{classifier_type}}.parquet
                done
              done
            done
            """


rule merge_initializer_eval_perf:
    input:
        lambda wc: [
            f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-{initializer}/eval_perf/"
            f"method_set-{wc.method_set}+classifier_type-{wc.classifier_type}.parquet"
            for initializer in INITIALIZERS
        ]
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-comparison/eval_perf/"
        "method_set-{method_set}+classifier_type-{classifier_type}.parquet"
    resources:
        shell_exec="bash"
    shell:
        """
        mkdir -p $(dirname {output})
        python scripts/misc/merge_dataframes.py {input} --output {output}
        """


rule plot_initializer_eval_perf:
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-comparison/eval_perf/"
        "method_set-{method_set}+classifier_type-{classifier_type}.parquet"
    output:
        directory(
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/initializer-comparison/eval_perf/"
            "method_set-{method_set}+classifier_type-{classifier_type}"
        )
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_perf_by_initializer.py \
            {input} {output} --budget-mode hard --keep-largest-hard-budget
        """
