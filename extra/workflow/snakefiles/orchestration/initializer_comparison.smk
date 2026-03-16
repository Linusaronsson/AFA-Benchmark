"""
Compare evaluation performance across initializers.

Runtime config (--config / --configfile):
    train_initializers (list[str], required):
        Train-stage initializer names without prefix. This workflow compares
        train-time support conditions while holding the eval initializer fixed.
    eval_initializer (str, default="cold"):
        Eval-stage initializer used together with train_initializers.
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
    pipeline_profile (str, default=""):
        Optional Snakemake profile passed to nested `pipeline.smk`
        invocations. Use this for SLURM-backed materialization, e.g. Alvis.
    pipeline_rerun_incomplete (bool, default=True):
        Whether to pass `--rerun-incomplete` to nested `pipeline.smk`
        invocations.
    pipeline_extra_config (str, default=""):
        Optional raw `--config` key-value overrides appended to nested
        `pipeline.smk` invocations.

Input expectations:
    This workflow expects upstream outputs from pipeline.smk for each
    condition at:
      extra/output/merged_results/eval_split-<split>/
      train_initializer-<train_initializer>+eval_initializer-<eval_initializer>/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>.parquet

Outputs:
    Merged cross-initializer parquet:
      extra/output/merged_results/eval_split-<split>/<comparison-tag>/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>.parquet
    Plots:
      extra/output/plot_results/eval_split-<split>/<comparison-tag>/
      eval_perf/method_set-<method_set>+classifier_type-<classifier>/
"""

import shlex

EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")
TRAIN_INITIALIZERS = config.get("train_initializers", None)
EVAL_INITIALIZER = config.get("eval_initializer", "cold")
METHOD_SETS = config.get("method_sets", None)
CLASSIFIER_TYPES = config.get("classifier_types", ["external"])
PIPELINE_CONFIGFILES = config.get("pipeline_configfiles", [])
PIPELINE_CORES = config.get("pipeline_cores", 8)
PIPELINE_PROFILE = config.get("pipeline_profile", "")
PIPELINE_RERUN_INCOMPLETE = config.get("pipeline_rerun_incomplete", True)
PIPELINE_EXTRA_CONFIG = config.get("pipeline_extra_config", "")

if TRAIN_INITIALIZERS is None:
    raise ValueError("Expected 'train_initializers' in config.")
if METHOD_SETS is None:
    raise ValueError("Expected 'method_sets' in config.")

INITIALIZERS = TRAIN_INITIALIZERS
METHOD_SET_NAMES = list(METHOD_SETS.keys())


def _initializer_tag(initializer: str) -> str:
    return (
        f"train_initializer-{initializer}+"
        f"eval_initializer-{EVAL_INITIALIZER}"
    )


def _comparison_tag() -> str:
    return (
        "train_initializer-comparison+"
        f"eval_initializer-{EVAL_INITIALIZER}"
    )


def _pipeline_execution_args() -> str:
    if PIPELINE_PROFILE:
        return "--profile " + shlex.quote(PIPELINE_PROFILE)
    return "--cores " + shlex.quote(str(PIPELINE_CORES))


def _pipeline_configfile_args() -> str:
    return " ".join(shlex.quote(path) for path in PIPELINE_CONFIGFILES)


def _pipeline_extra_config_args() -> str:
    if not PIPELINE_EXTRA_CONFIG:
        return ""
    return " ".join(shlex.quote(arg) for arg in shlex.split(PIPELINE_EXTRA_CONFIG))


def _upstream_eval_perf_paths():
    return [
        (
            f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_initializer_tag(initializer)}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}.parquet"
        )
        for initializer in INITIALIZERS
        for method_set in METHOD_SET_NAMES
        for classifier_type in CLASSIFIER_TYPES
    ]


rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
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
            configfiles=_pipeline_configfile_args(),
            eval_initializer=EVAL_INITIALIZER,
            rerun_incomplete=(
                "--rerun-incomplete"
                if PIPELINE_RERUN_INCOMPLETE
                else ""
            ),
            pipeline_extra_config=_pipeline_extra_config_args(),
            pipeline_execution_args=_pipeline_execution_args(),
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
                    --config train_initializer="${{initializer}}" \
                             eval_initializer="{params.eval_initializer}" \
                             eval_dataset_split="{EVAL_DATASET_SPLIT}" \
                             {params.pipeline_extra_config} \
                    {params.pipeline_execution_args} \
                    extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/train_initializer-${{initializer}}+eval_initializer-{params.eval_initializer}/eval_perf/method_set-${{method_set}}+classifier_type-${{classifier_type}}.parquet
                done
              done
            done
            """


rule merge_initializer_eval_perf:
    input:
        lambda wc: [
            f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/train_initializer-{initializer}+eval_initializer-{EVAL_INITIALIZER}/eval_perf/method_set-{wc.method_set}+classifier_type-{wc.classifier_type}.parquet"
            for initializer in INITIALIZERS
        ]
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
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
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
        "method_set-{method_set}+classifier_type-{classifier_type}.parquet"
    output:
        directory(
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
            "method_set-{method_set}+classifier_type-{classifier_type}"
        )
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_eval_perf_by_initializer.py \
            {input} {output} --budget-mode hard --keep-largest-hard-budget
        """
