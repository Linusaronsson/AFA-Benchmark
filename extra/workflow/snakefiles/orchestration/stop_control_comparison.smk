"""
Compare eval-time stop-control overlays while holding train/eval initializers fixed.

Runtime config (--config / --configfile):
    methods (list[str], required):
        Methods to compare. A synthetic method set is created from this list by
        default so the user-facing interface matches `pipeline.smk`.
    datasets (list[str], optional):
        Dataset subset forwarded to nested `pipeline.smk` invocations.
    train_initializer (str, default="cold"):
        Train-stage initializer used for all compared overlays.
    eval_initializer (str, default="cold"):
        Eval-stage initializer used for all compared overlays.
    stop_shield_deltas (list[float], default=[]):
        Threshold-shielded evaluation variants to compare.
    dual_lambdas (list[float], default=[]):
        Dualized-stop evaluation variants to compare.
    method_sets (dict[str, list[str]], optional):
        Optional explicit method sets. If omitted, a single synthetic method set
        is constructed from `methods`.
    comparison_method_set_name (str, default="selected_methods"):
        Method-set name used when synthesizing a method set from `methods`.
    eval_dataset_split (str, default="val"):
        Evaluation split used in upstream pipeline outputs.
    classifier_types (list[str], default=["external"]):
        Which classifier-specific merged parquet files to compare.
    pipeline_configfiles (list[str], optional):
        If provided, this workflow first materializes required upstream
        `pipeline.smk` merged parquet files for each stop-control condition.
    dataset_instance_indices / device / use_wandb / smoke_test (optional):
        Top-level pipeline-style config keys that are forwarded directly to
        nested `pipeline.smk` invocations.
    pipeline_cores (int, default=8):
        Cores used by nested `pipeline.smk` invocations when
        `pipeline_configfiles` is provided.
    pipeline_profile (str, default=""):
        Optional Snakemake profile passed to nested `pipeline.smk`
        invocations.
    pipeline_rerun_incomplete (bool, default=True):
        Whether to pass `--rerun-incomplete` to nested `pipeline.smk`
        invocations.
    pipeline_extra_config (str, default=""):
        Optional raw `--config` key-value overrides appended to nested
        `pipeline.smk` invocations.
    plot_budget_mode (str, default="soft"):
        Budget regime used when plotting merged comparison outputs.
    plot_keep_largest_hard_budget (bool, default=True):
        When plot_budget_mode="hard", keep only the largest hard budget per
        dataset in the comparison plot.
    plot_eval_hard_budget (float, optional):
        When set and plot_budget_mode="hard", keep only this hard budget in
        the comparison plot. This takes precedence over
        plot_keep_largest_hard_budget.
"""

import json
import re
import shlex

EVAL_DATASET_SPLIT = config.get("eval_dataset_split", "val")
TRAIN_INITIALIZER = config.get("train_initializer", "cold")
EVAL_INITIALIZER = config.get("eval_initializer", "cold")
STOP_SHIELD_DELTAS = [
    str(delta) for delta in config.get("stop_shield_deltas", [])
]
DUAL_LAMBDAS = [str(value) for value in config.get("dual_lambdas", [])]
METHODS = config.get("methods", None)
METHOD_SETS = config.get("method_sets", None)
COMPARISON_METHOD_SET_NAME = config.get(
    "comparison_method_set_name",
    "selected_methods",
)
CLASSIFIER_TYPES = config.get("classifier_types", ["external"])
PIPELINE_CONFIGFILES = config.get("pipeline_configfiles", [])
PIPELINE_CORES = config.get("pipeline_cores", 8)
PIPELINE_PROFILE = config.get("pipeline_profile", "")
PIPELINE_RERUN_INCOMPLETE = config.get("pipeline_rerun_incomplete", True)
PIPELINE_EXTRA_CONFIG = config.get("pipeline_extra_config", "")
PLOT_BUDGET_MODE = config.get("plot_budget_mode", "soft")
PLOT_KEEP_LARGEST_HARD_BUDGET = config.get(
    "plot_keep_largest_hard_budget", True
)
PLOT_EVAL_HARD_BUDGET = config.get("plot_eval_hard_budget", None)
PIPELINE_PASSTHROUGH_KEYS = (
    "datasets",
    "methods",
    "dataset_instance_indices",
    "device",
    "default_compute_platform",
    "compute_platform_devices",
    "classifier_compute_platform",
    "slurm_extra_by_compute_platform",
    "use_wandb",
    "smoke_test",
)

if METHODS is None:
    raise ValueError("Expected 'methods' in config.")
if METHOD_SETS is None:
    METHOD_SETS = {COMPARISON_METHOD_SET_NAME: METHODS}
if not STOP_SHIELD_DELTAS and not DUAL_LAMBDAS:
    raise ValueError(
        "Expected at least one of 'stop_shield_deltas' or 'dual_lambdas'."
    )
if PIPELINE_CONFIGFILES and "datasets" not in config:
    raise ValueError(
        "Expected 'datasets' in config when 'pipeline_configfiles' is set."
    )

METHOD_SET_NAMES = list(METHOD_SETS.keys())


def _initializer_tag() -> str:
    return (
        f"train_initializer-{TRAIN_INITIALIZER}+"
        f"eval_initializer-{EVAL_INITIALIZER}"
    )


def _comparison_tag() -> str:
    return (
        "stop_control-comparison+"
        f"train_initializer-{TRAIN_INITIALIZER}+"
        f"eval_initializer-{EVAL_INITIALIZER}"
    )


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", value)


def _plot_tag() -> str:
    if PLOT_BUDGET_MODE == "hard":
        if PLOT_EVAL_HARD_BUDGET is not None:
            return (
                "budget_mode-hard+eval_hard_budget-"
                f"{_slugify(str(PLOT_EVAL_HARD_BUDGET))}"
            )
        if PLOT_KEEP_LARGEST_HARD_BUDGET:
            return "budget_mode-hard+largest_hard_budget"
    return f"budget_mode-{PLOT_BUDGET_MODE}"


def _pipeline_execution_args() -> str:
    if PIPELINE_PROFILE:
        return "--profile " + shlex.quote(PIPELINE_PROFILE)
    return "--cores " + shlex.quote(str(PIPELINE_CORES))


def _pipeline_configfile_args() -> str:
    return " ".join(shlex.quote(path) for path in PIPELINE_CONFIGFILES)


def _pipeline_extra_config_args() -> str:
    if not PIPELINE_EXTRA_CONFIG:
        return ""
    return " ".join(
        shlex.quote(arg) for arg in shlex.split(PIPELINE_EXTRA_CONFIG)
    )


def _serialize_pipeline_config_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _pipeline_passthrough_config_args() -> str:
    args = []
    for key in PIPELINE_PASSTHROUGH_KEYS:
        if key not in config:
            continue
        value = config[key]
        if value is None:
            continue
        serialized_value = _serialize_pipeline_config_value(value)
        args.append(shlex.quote(f"{key}={serialized_value}"))
    args.append(
        shlex.quote(
            "method_sets=" + _serialize_pipeline_config_value(METHOD_SETS)
        )
    )
    return " ".join(args)


def _plot_eval_perf_args() -> str:
    args = [
        "--group-column",
        "comparison_group",
        "--group-label",
        shlex.quote("Stop control"),
        "--output-stem",
        "stop_control_comparison",
        "--budget-mode",
        shlex.quote(PLOT_BUDGET_MODE),
    ]
    if PLOT_EVAL_HARD_BUDGET is not None:
        args.extend(
            [
                "--eval-hard-budget",
                shlex.quote(str(PLOT_EVAL_HARD_BUDGET)),
            ]
        )
    elif PLOT_KEEP_LARGEST_HARD_BUDGET:
        args.append("--keep-largest-hard-budget")
    return " ".join(args)


def _comparison_conditions() -> list[tuple[str, str, str | None]]:
    return [("Unshielded", "unshielded", None)] + [
        (f"Shield delta={delta}", "shielded", delta)
        for delta in STOP_SHIELD_DELTAS
    ] + [
        (f"Dual lambda={dual_lambda}", "dualized", dual_lambda)
        for dual_lambda in DUAL_LAMBDAS
    ]


def _condition_input_path(
    *,
    method_set: str,
    classifier_type: str,
    condition_kind: str,
    condition_value: str | None,
) -> str:
    if condition_kind == "unshielded":
        return (
            f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_initializer_tag()}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}.parquet"
        )
    if condition_kind == "shielded":
        return (
            "extra/output/merged_results_shielded/"
            f"delta-{condition_value}/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_initializer_tag()}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}.parquet"
        )
    if condition_kind == "dualized":
        return (
            "extra/output/merged_results_dualized/"
            f"lambda-{condition_value}/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_initializer_tag()}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}.parquet"
        )
    raise ValueError(f"Unsupported condition kind: {condition_kind}")


def _upstream_eval_perf_paths() -> list[str]:
    return [
        _condition_input_path(
            method_set=method_set,
            classifier_type=classifier_type,
            condition_kind=condition_kind,
            condition_value=condition_value,
        )
        for method_set in METHOD_SET_NAMES
        for classifier_type in CLASSIFIER_TYPES
        for _label, condition_kind, condition_value in _comparison_conditions()
    ]


def _materialization_command(
    *,
    method_set: str,
    classifier_type: str,
    condition_kind: str,
    condition_value: str | None,
) -> str:
    target = _condition_input_path(
        method_set=method_set,
        classifier_type=classifier_type,
        condition_kind=condition_kind,
        condition_value=condition_value,
    )
    cmd = [
        "uv",
        "run",
        "snakemake",
        "-s",
        "extra/workflow/snakefiles/orchestration/pipeline.smk",
    ]
    if PIPELINE_RERUN_INCOMPLETE:
        cmd.append("--rerun-incomplete")
    cmd.extend(
        [
            "--nolock",
            "--configfile",
            *PIPELINE_CONFIGFILES,
            "--config",
            f"train_initializer={TRAIN_INITIALIZER}",
            f"eval_initializer={EVAL_INITIALIZER}",
            f"eval_dataset_split={EVAL_DATASET_SPLIT}",
        ]
    )
    passthrough_args = _pipeline_passthrough_config_args()
    if passthrough_args:
        cmd.extend(shlex.split(passthrough_args))
    extra_args = _pipeline_extra_config_args()
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    cmd.extend(shlex.split(_pipeline_execution_args()))
    cmd.append(target)
    return " ".join(shlex.quote(part) for part in cmd)


def _materialization_commands() -> str:
    return "\n".join(
        _materialization_command(
            method_set=method_set,
            classifier_type=classifier_type,
            condition_kind=condition_kind,
            condition_value=condition_value,
        )
        for method_set in METHOD_SET_NAMES
        for classifier_type in CLASSIFIER_TYPES
        for _label, condition_kind, condition_value in _comparison_conditions()
    )


def _merge_eval_perf_args(method_set: str, classifier_type: str) -> str:
    args = []
    for label, condition_kind, condition_value in _comparison_conditions():
        args.append("--input")
        args.append(shlex.quote(label))
        args.append(
            shlex.quote(
                _condition_input_path(
                    method_set=method_set,
                    classifier_type=classifier_type,
                    condition_kind=condition_kind,
                    condition_value=condition_value,
                )
            )
        )
    return " ".join(args)


rule all:
    input:
        [
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_comparison_tag()}/{_plot_tag()}/eval_perf/"
            f"method_set-{method_set}+classifier_type-{classifier_type}"
            for method_set in METHOD_SET_NAMES
            for classifier_type in CLASSIFIER_TYPES
        ]


if PIPELINE_CONFIGFILES:
    rule materialize_upstream_eval_perf:
        output:
            _upstream_eval_perf_paths()
        params:
            commands=_materialization_commands(),
        resources:
            shell_exec="bash"
        shell:
            """
            set -euo pipefail
            {params.commands}
            """


rule merge_stop_control_eval_perf:
    input:
        lambda wc: [
            _condition_input_path(
                method_set=wc.method_set,
                classifier_type=wc.classifier_type,
                condition_kind=condition_kind,
                condition_value=condition_value,
            )
            for _label, condition_kind, condition_value in _comparison_conditions()
        ]
    output:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
        "method_set-{method_set}+classifier_type-{classifier_type}.parquet"
    params:
        merge_args=lambda wc: _merge_eval_perf_args(
            wc.method_set,
            wc.classifier_type,
        ),
    resources:
        shell_exec="bash"
    shell:
        """
        mkdir -p $(dirname {output})
        uv run python scripts/misc/merge_eval_perf_comparison.py \
            {params.merge_args} \
            --group-column comparison_group \
            --output {output}
        """


rule plot_stop_control_eval_perf:
    input:
        f"extra/output/merged_results/eval_split-{EVAL_DATASET_SPLIT}/{_comparison_tag()}/eval_perf/"
        "method_set-{method_set}+classifier_type-{classifier_type}.parquet"
    output:
        directory(
            f"extra/output/plot_results/eval_split-{EVAL_DATASET_SPLIT}/"
            f"{_comparison_tag()}/{_plot_tag()}/eval_perf/"
            "method_set-{method_set}+classifier_type-{classifier_type}"
        )
    params:
        plot_args=_plot_eval_perf_args()
    resources:
        shell_exec="bash"
    shell:
        """
        uv run python scripts/plotting/plot_eval_perf_by_initializer.py \
            {input} {output} {params.plot_args}
        """
