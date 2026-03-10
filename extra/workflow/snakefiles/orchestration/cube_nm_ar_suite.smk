"""
One-command CUBE-NM-AR suite orchestration.

This workflow keeps `pipeline.smk` as the execution engine and adds a thin
suite layer that materializes:
- multiple train initializers,
- multiple eval initializers,
- hard and soft budget modes,
- optional stop-shielded evaluation variants,

and then writes cross-initializer CUBE-NM-AR comparison plots.

Runtime config (`--configfile` on this snakefile):
    cube_nm_ar_suite_train_initializers (list[str], default=["cube_nm_ar", "cold"])
    cube_nm_ar_suite_eval_initializers (list[str], default=["cold"])
    cube_nm_ar_suite_budget_modes (list[str], default=["hard", "soft"])
    cube_nm_ar_suite_dataset (str, default="cube_nm_ar")
    cube_nm_ar_suite_eval_dataset_split (str, default="val")
    cube_nm_ar_suite_methods (list[str], required)
    cube_nm_ar_suite_mode_methods (dict[str, list[str]], optional)
    cube_nm_ar_suite_stop_shield_deltas (list[float], default=[])
    cube_nm_ar_suite_mode_stop_shield_deltas (dict[str, list[float]], optional)
    cube_nm_ar_suite_pipeline_configfiles (list[str], required)
    cube_nm_ar_suite_mode_configfiles (dict[str, list[str]], required)
    cube_nm_ar_suite_pipeline_cores (int, default=8)
    cube_nm_ar_suite_pipeline_rerun_incomplete (bool, default=True)
    cube_nm_ar_suite_pipeline_config_overrides (list[str], default=[])
"""

import shlex

SUITE_TRAIN_INITIALIZERS = config.get(
    "cube_nm_ar_suite_train_initializers",
    ["cube_nm_ar", "cold"],
)
SUITE_EVAL_INITIALIZERS = config.get(
    "cube_nm_ar_suite_eval_initializers",
    ["cold"],
)
SUITE_BUDGET_MODES = config.get(
    "cube_nm_ar_suite_budget_modes",
    ["hard", "soft"],
)
SUITE_DATASET = config.get("cube_nm_ar_suite_dataset", "cube_nm_ar")
SUITE_EVAL_DATASET_SPLIT = config.get(
    "cube_nm_ar_suite_eval_dataset_split",
    "val",
)
SUITE_METHODS = config.get("cube_nm_ar_suite_methods", [])
SUITE_MODE_METHODS = config.get("cube_nm_ar_suite_mode_methods", {})
SUITE_STOP_SHIELD_DELTAS = [
    str(delta)
    for delta in config.get("cube_nm_ar_suite_stop_shield_deltas", [])
]
SUITE_MODE_STOP_SHIELD_DELTAS = {
    budget_mode: [str(delta) for delta in deltas]
    for budget_mode, deltas in config.get(
        "cube_nm_ar_suite_mode_stop_shield_deltas",
        {},
    ).items()
}
SUITE_PIPELINE_CONFIGFILES = config.get(
    "cube_nm_ar_suite_pipeline_configfiles",
    [],
)
SUITE_MODE_CONFIGFILES = config.get("cube_nm_ar_suite_mode_configfiles", {})
SUITE_PIPELINE_CORES = config.get("cube_nm_ar_suite_pipeline_cores", 8)
SUITE_PIPELINE_RERUN_INCOMPLETE = config.get(
    "cube_nm_ar_suite_pipeline_rerun_incomplete",
    True,
)
SUITE_PIPELINE_CONFIG_OVERRIDES = config.get(
    "cube_nm_ar_suite_pipeline_config_overrides",
    [],
)

if not SUITE_METHODS:
    raise ValueError("Expected 'cube_nm_ar_suite_methods' in config.")
if not SUITE_PIPELINE_CONFIGFILES:
    raise ValueError(
        "Expected 'cube_nm_ar_suite_pipeline_configfiles' in config."
    )
if not SUITE_MODE_CONFIGFILES:
    raise ValueError(
        "Expected 'cube_nm_ar_suite_mode_configfiles' in config."
    )
invalid_budget_modes = [
    budget_mode
    for budget_mode in SUITE_BUDGET_MODES
    if budget_mode not in {"hard", "soft"}
]
if invalid_budget_modes:
    raise ValueError(
        "Invalid cube_nm_ar_suite_budget_modes: "
        + ", ".join(invalid_budget_modes)
    )
missing_mode_configfiles = [
    budget_mode
    for budget_mode in SUITE_BUDGET_MODES
    if budget_mode not in SUITE_MODE_CONFIGFILES
]
if missing_mode_configfiles:
    raise ValueError(
        "Missing cube_nm_ar_suite_mode_configfiles entries for: "
        + ", ".join(missing_mode_configfiles)
    )


def _initializer_tag(train_initializer: str, eval_initializer: str) -> str:
    return (
        f"train_initializer-{train_initializer}+"
        f"eval_initializer-{eval_initializer}"
    )


def _mode_methods(budget_mode: str) -> list[str]:
    return SUITE_MODE_METHODS.get(budget_mode, SUITE_METHODS)


def _mode_stop_shield_deltas(budget_mode: str) -> list[str]:
    return SUITE_MODE_STOP_SHIELD_DELTAS.get(
        budget_mode,
        SUITE_STOP_SHIELD_DELTAS,
    )


def _format_cli_list(values: list[str]) -> str:
    return "[" + ",".join(repr(value) for value in values) + "]"


def _suite_marker() -> str:
    return (
        "extra/output/.suite_state/cube_nm_ar/"
        f"eval_split-{SUITE_EVAL_DATASET_SPLIT}/suite_materialized.done"
    )


def _upstream_plot_targets(
    train_initializer: str,
    eval_initializer: str,
    budget_mode: str,
) -> list[str]:
    base_dir = (
        "extra/output/plot_results/cube_nm_ar/"
        f"eval_split-{SUITE_EVAL_DATASET_SPLIT}/"
        f"{_initializer_tag(train_initializer, eval_initializer)}/"
        f"budget_mode-{budget_mode}"
    )
    return [f"{base_dir}/stop_shield-none"] + [
        f"{base_dir}/stop_shield-{delta}"
        for delta in _mode_stop_shield_deltas(budget_mode)
    ]


def _comparison_plot_dir(
    eval_initializer: str,
    budget_mode: str,
    stop_shield_tag: str,
) -> str:
    return (
        "extra/output/plot_results/cube_nm_ar/"
        f"eval_split-{SUITE_EVAL_DATASET_SPLIT}/"
        f"train_initializer-comparison+eval_initializer-{eval_initializer}/"
        f"budget_mode-{budget_mode}/stop_shield-{stop_shield_tag}"
    )


def _materialization_command(
    train_initializer: str,
    eval_initializer: str,
    budget_mode: str,
) -> str:
    configfiles = (
        SUITE_PIPELINE_CONFIGFILES
        + SUITE_MODE_CONFIGFILES[budget_mode]
    )
    cmd = [
        "uv",
        "run",
        "snakemake",
        "-s",
        "extra/workflow/snakefiles/orchestration/pipeline.smk",
    ]
    if SUITE_PIPELINE_RERUN_INCOMPLETE:
        cmd.append("--rerun-incomplete")
    cmd.extend(
        [
            "--nolock",
            "--configfile",
            *configfiles,
            "--config",
            *SUITE_PIPELINE_CONFIG_OVERRIDES,
            f"methods={_format_cli_list(_mode_methods(budget_mode))}",
            "stop_shield_deltas="
            f"{_format_cli_list(_mode_stop_shield_deltas(budget_mode))}",
            f"train_initializer={train_initializer}",
            f"eval_initializer={eval_initializer}",
            f"eval_dataset_split={SUITE_EVAL_DATASET_SPLIT}",
            f"cube_nm_ar_budget_mode={budget_mode}",
            "--cores",
            str(SUITE_PIPELINE_CORES),
            *_upstream_plot_targets(
                train_initializer,
                eval_initializer,
                budget_mode,
            ),
        ]
    )
    return " ".join(shlex.quote(part) for part in cmd)


MATERIALIZATION_COMMANDS = "\n".join(
    _materialization_command(
        train_initializer,
        eval_initializer,
        budget_mode,
    )
    for train_initializer in SUITE_TRAIN_INITIALIZERS
    for eval_initializer in SUITE_EVAL_INITIALIZERS
    for budget_mode in SUITE_BUDGET_MODES
)


rule all:
    input:
        [
            _comparison_plot_dir(eval_initializer, budget_mode, "none")
            for eval_initializer in SUITE_EVAL_INITIALIZERS
            for budget_mode in SUITE_BUDGET_MODES
        ]
        + [
            _comparison_plot_dir(
                eval_initializer,
                budget_mode,
                stop_shield_delta,
            )
            for eval_initializer in SUITE_EVAL_INITIALIZERS
            for budget_mode in SUITE_BUDGET_MODES
            for stop_shield_delta in _mode_stop_shield_deltas(budget_mode)
        ]


rule materialize_cube_nm_ar_suite:
    output:
        _suite_marker()
    params:
        commands=MATERIALIZATION_COMMANDS,
    resources:
        shell_exec="bash"
    shell:
        """
        set -euo pipefail
        {params.commands}
        mkdir -p $(dirname {output})
        touch {output}
        """


rule plot_cube_nm_ar_initializer_comparison:
    input:
        _suite_marker()
    output:
        directory(
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{SUITE_EVAL_DATASET_SPLIT}/"
            "train_initializer-comparison+eval_initializer-{eval_initializer}/"
            "budget_mode-{budget_mode}/stop_shield-none"
        )
    params:
        methods=lambda wildcards: " ".join(
            _mode_methods(wildcards.budget_mode)
        ),
        train_initializers=" ".join(SUITE_TRAIN_INITIALIZERS),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_cube_nm_ar_results.py \
            extra/output/eval_results/eval_split-{SUITE_EVAL_DATASET_SPLIT} \
            {output} \
            --train-initializers {params.train_initializers} \
            --eval-initializers {wildcards.eval_initializer} \
            --budget-mode {wildcards.budget_mode} \
            --dataset {SUITE_DATASET} \
            --methods {params.methods}
        """


rule plot_cube_nm_ar_initializer_comparison_shielded:
    input:
        _suite_marker()
    wildcard_constraints:
        stop_shield_delta="(?!none$)[^/]+"
    output:
        directory(
            "extra/output/plot_results/cube_nm_ar/"
            f"eval_split-{SUITE_EVAL_DATASET_SPLIT}/"
            "train_initializer-comparison+eval_initializer-{eval_initializer}/"
            "budget_mode-{budget_mode}/stop_shield-{stop_shield_delta}"
        )
    params:
        methods=lambda wildcards: " ".join(
            _mode_methods(wildcards.budget_mode)
        ),
        train_initializers=" ".join(SUITE_TRAIN_INITIALIZERS),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_cube_nm_ar_results.py \
            extra/output/eval_results_shielded/delta-{wildcards.stop_shield_delta}/eval_split-{SUITE_EVAL_DATASET_SPLIT} \
            {output} \
            --train-initializers {params.train_initializers} \
            --eval-initializers {wildcards.eval_initializer} \
            --budget-mode {wildcards.budget_mode} \
            --dataset {SUITE_DATASET} \
            --methods {params.methods}
        """
