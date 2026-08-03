"""Generic experiments with missing values in training data only.

The workflow consumes the same dataset, method, pretraining, classifier,
unmasker, and hard-budget configuration files as the ordinary evaluation
pipeline. Missingness-specific scientific choices live in
``conf/missing_data/design.yaml``; runtime scale and devices live in the
selected runtime config and execution profile.

Required shared config keys:
    datasets, methods, method_options, pretrain_mapping, eval_hard_budgets,
    soft_budget_params, unmaskers, classifier_names

Missing-data config keys:
    artifact_namespace, missingness, strategies, eval_dataset_split
    Optional ``classifier_scope`` is ``shared`` for legacy experiments or
    ``per_instance`` to fit classifiers on each dataset split independently.
    Optional ``run_confirmatory_analysis`` writes paired path, restoration,
    stepwise, and DIME-invariance tables after aggregation.
    Optional ``strategy_filters`` restricts one strategy by datasets, methods,
    mechanisms, or probabilities without duplicating the experiment workflow.
    Optional ``include_method_variants`` omits missingness-specific controls
    from a focused experiment while retaining them by default.

Every non-local production rule writes a Snakemake benchmark TSV under
``extra/output/missing_data/benchmark/<namespace>``. These records measure the
wall time, CPU time, and peak resident memory of the exact artifact-producing
job without becoming part of the rule's completeness contract.
Hydra's per-invocation configuration and log files are redirected to temporary
storage because the scheduler log and benchmark TSV already retain the command
and resource record.

Device routing:
    ``device`` is the default. Optional ``device_overrides`` may contain
    ``methods``, ``datasets``, ``method_datasets``, and ``pretrained_models``.
    ``AFABENCH_DEVICE`` may replace only that default at execution time without
    replacing a profile's scientific config. Every CUDA-resolved rule consumes
    one ``gpu`` resource; the runner caps those tokens to physical devices.

Example:
    uv run snakemake \
      --profile extra/workflow/profiles/config/missing_data_smoke --cores 4
"""

import os
import re
import sys

snakefile_dir = workflow.basedir
workflow_dir = os.path.dirname(os.path.dirname(snakefile_dir))
src_dir = os.path.join(workflow_dir, "src")
sys.path.insert(0, src_dir)

from config import load_config, resolve_device
from missing_data_config import (
    build_method_specs,
    largest_hard_budget,
    strategy_enabled,
)


required_missing_config = {
    "artifact_namespace",
    "eval_dataset_split",
    "missingness",
    "strategies",
}
missing_config = sorted(required_missing_config - set(config))
if missing_config:
    raise ValueError(
        "Missing required missing-data config keys: "
        + ", ".join(missing_config)
    )

if runtime_device := os.environ.get("AFABENCH_DEVICE"):
    config["device"] = runtime_device
_config = load_config(config)

NAMESPACE = str(config["artifact_namespace"])
INSTANCES = [int(value) for value in _config["DATASET_INSTANCE_INDICES"]]
if not INSTANCES:
    raise ValueError("dataset_instance_indices must not be empty")
CLASSIFIER_INSTANCE = INSTANCES[0]
CLASSIFIER_SCOPE = str(config.get("classifier_scope", "shared"))
if CLASSIFIER_SCOPE not in {"shared", "per_instance"}:
    raise ValueError("classifier_scope must be 'shared' or 'per_instance'")

UNSUPPORTED_DATASETS = set(
    str(value)
    for value in config.get("missing_data_unsupported_datasets", [])
)
SELECTED_DATASETS = [str(value) for value in _config["DATASETS"]]
DATASETS = [
    dataset
    for dataset in SELECTED_DATASETS
    if dataset not in UNSUPPORTED_DATASETS
]
excluded_datasets = sorted(set(SELECTED_DATASETS) - set(DATASETS))
if excluded_datasets:
    print(
        "Skipping datasets without flat-feature missingness support: "
        + ", ".join(excluded_datasets)
    )
if not DATASETS:
    raise ValueError("No missing-data-compatible datasets were selected")

BASE_METHODS = [str(value) for value in _config["METHODS"]]
METHOD_OPTIONS = _config["METHOD_OPTIONS"]
METHOD_SPECS = build_method_specs(
    BASE_METHODS,
    METHOD_OPTIONS,
    config.get("missing_data_method_overrides", {}),
    (
        config.get("missing_data_method_variants", {})
        if config.get("include_method_variants", True)
        else {}
    ),
)
PRETRAINED_METHODS = [
    name
    for name, spec in METHOD_SPECS.items()
    if spec.pretrained_model_name is not None
]
UNPRETRAINED_METHODS = [
    name
    for name, spec in METHOD_SPECS.items()
    if spec.pretrained_model_name is None
]
COMMON_STRATEGIES = [str(value) for value in config["strategies"]]
STRATEGY_FILTERS = config.get("strategy_filters", {})
INCLUDE_COMPLETE_DATA = bool(config.get("include_complete_data", True))

UNMASKERS = _config["UNMASKERS"]
CLASSIFIER_NAMES = _config["CLASSIFIER_NAMES"]
METHOD_CLASSIFIER_SCRIPT_NAMES = _config["METHOD_CLASSIFIER_SCRIPT_NAMES"]
METHOD_CLASSIFIER_SCRIPT_PARAMS = _config["METHOD_CLASSIFIER_SCRIPT_PARAMS"]
PRETRAIN_SCRIPT_NAMES = _config["PRETRAIN_SCRIPT_NAMES"]
PRETRAIN_PARAMS = _config["PRETRAIN_PARAMS"]
EVAL_BATCH_SIZES = _config["EVAL_BATCH_SIZES"]
HARD_BUDGET_IGNORED_DATASETS = _config[
    "HARD_BUDGET_IGNORED_DATASETS"
]

DEFAULT_DEVICE = str(_config["DEVICE"])
DEVICE_OVERRIDES = _config["DEVICE_OVERRIDES"]
USE_WANDB = str(bool(_config["USE_WANDB"])).lower()
SMOKE_TEST = bool(_config["SMOKE_TEST"])
SMOKE_TEST_STR = str(SMOKE_TEST).lower()
INITIALIZER = str(_config["INITIALIZER"])
EVAL_SPLIT = str(_config["EVAL_DATASET_SPLIT"])
if EVAL_SPLIT not in {"val", "test"}:
    raise ValueError("eval_dataset_split must be either 'val' or 'test'")

ROOT = "extra/output/missing_data"
BENCHMARK_ROOT = f"{ROOT}/benchmark/{NAMESPACE}"
HYDRA_WORKFLOW_OVERRIDES = (
    "hydra/job_logging=workflow_console "
    "hydra.run.dir=${SNIC_TMP:-/tmp}/afabench-hydra/"
    "${SLURM_JOB_ID:-local} hydra.output_subdir=null"
)
MISSINGNESS = config["missingness"]
MECHANISMS = [str(value) for value in MISSINGNESS["mechanisms"]]
PROBABILITIES = [str(value) for value in MISSINGNESS["probabilities"]]
MISSING_COMBINATIONS = [
    (mechanism, probability)
    for mechanism in MECHANISMS
    for probability in PROBABILITIES
]

DATASET_GENERATION_PARAMS = config.get("dataset_generation_params", {})
CLASSIFIER_PARAMS = config.get("classifier_params", {})
RESTORATION_PVAE_PARAMS = config.get("restoration_pvae_params", {})
PRETRAIN_RUNTIME_PARAMS = config.get("pretrain_runtime_params", {})
TRAIN_RUNTIME_PARAMS = config.get("train_runtime_params", {})
EVAL_PARAMS = config.get("eval_params", {})
RESTORATION_BATCH_SIZE = int(config.get("restoration_batch_size", 1024))
EVAL_BATCH_SIZE_OVERRIDE = config.get("eval_batch_size")
STEPWISE_EVAL_BATCH_SIZE = int(config.get("stepwise_eval_batch_size", 16))
RUN_CONFIRMATORY_ANALYSIS = bool(
    config.get("run_confirmatory_analysis", False)
)


def wildcard_pattern(values):
    unique_values = dict.fromkeys(str(value) for value in values)
    return "(?:" + "|".join(re.escape(value) for value in unique_values) + ")"


def runtime_params(mapping, *keys):
    if isinstance(mapping, list):
        return " ".join(str(value) for value in mapping)
    values = [str(value) for value in mapping.get("default", [])]
    for key in dict.fromkeys(str(value) for value in keys if value is not None):
        values.extend(str(value) for value in mapping.get(key, []))
    return " ".join(values)


def method_device(dataset, method):
    spec = METHOD_SPECS[method]
    method_datasets = DEVICE_OVERRIDES.get("method_datasets", {})
    methods = DEVICE_OVERRIDES.get("methods", {})
    for candidate in dict.fromkeys([method, spec.base_method]):
        if dataset in method_datasets.get(candidate, {}):
            return str(method_datasets[candidate][dataset])
        if candidate in methods:
            return str(methods[candidate])
    return resolve_device(
        DEFAULT_DEVICE,
        DEVICE_OVERRIDES,
        dataset=dataset,
    )


def dataset_device(dataset, pretrained_model=None):
    return resolve_device(
        DEFAULT_DEVICE,
        DEVICE_OVERRIDES,
        dataset=dataset,
        pretrained_model=pretrained_model,
    )


def gpu_for_device(device):
    """Reserve one GPU slot exactly when the resolved device uses CUDA."""
    return int(str(device).split(":", maxsplit=1)[0] == "cuda")


def classifier_script_name(dataset):
    classifier = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier, dict):
        return classifier["script_name"]
    return classifier


def classifier_script_params(dataset):
    classifier = CLASSIFIER_NAMES[dataset]
    if isinstance(classifier, dict):
        configured = classifier.get("script_params", [])
    else:
        configured = []
    runtime = runtime_params(CLASSIFIER_PARAMS, dataset)
    return " ".join([*[str(value) for value in configured], runtime]).strip()


def classifier_instance(wildcards):
    if CLASSIFIER_SCOPE == "per_instance":
        return int(wildcards.instance)
    return CLASSIFIER_INSTANCE


def classifier_path(dataset, instance, method=None):
    instance_suffix = (
        f"/instance-{instance}.bundle"
        if CLASSIFIER_SCOPE == "per_instance"
        else ".bundle"
    )
    if method is not None:
        base_method = METHOD_SPECS[method].base_method
        if base_method in METHOD_CLASSIFIER_SCRIPT_NAMES:
            return (
                f"{ROOT}/classifier/{NAMESPACE}/"
                f"method-{base_method}+dataset-{dataset}{instance_suffix}"
            )
    return (
        f"{ROOT}/classifier/{NAMESPACE}/dataset-{dataset}{instance_suffix}"
    )


CLASSIFIER_OUTPUT = (
    f"{ROOT}/classifier/{NAMESPACE}/dataset-{{dataset}}/"
    "instance-{instance}.bundle"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{ROOT}/classifier/{NAMESPACE}/dataset-{{dataset}}.bundle"
)
METHOD_CLASSIFIER_OUTPUT = (
    f"{ROOT}/classifier/{NAMESPACE}/"
    "method-{base_method}+dataset-{dataset}/instance-{instance}.bundle"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{ROOT}/classifier/{NAMESPACE}/"
    "method-{base_method}+dataset-{dataset}.bundle"
)
CLASSIFIER_BENCHMARK = (
    f"{BENCHMARK_ROOT}/train_shared_classifier/dataset-{{dataset}}/"
    "instance-{instance}.tsv"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{BENCHMARK_ROOT}/train_shared_classifier/dataset-{{dataset}}.tsv"
)
METHOD_CLASSIFIER_BENCHMARK = (
    f"{BENCHMARK_ROOT}/train_method_classifier/"
    "method-{base_method}+dataset-{dataset}/instance-{instance}.tsv"
    if CLASSIFIER_SCOPE == "per_instance"
    else f"{BENCHMARK_ROOT}/train_method_classifier/"
    "method-{base_method}+dataset-{dataset}.tsv"
)


def raw_dataset(dataset, instance, split):
    return (
        f"{ROOT}/datasets/{NAMESPACE}/{dataset}/{instance}/{split}.bundle"
    )


def base_view(dataset, mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/base/{NAMESPACE}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        f"{strategy}/{split}.bundle"
    )


def restored_view(
    dataset, mechanism, probability, instance, strategy, split
):
    return (
        f"{ROOT}/views/restored/{NAMESPACE}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        f"{strategy}/{split}.bundle"
    )


def training_view(wildcards, split):
    if wildcards.strategy == "complete":
        return raw_dataset(wildcards.dataset, wildcards.instance, split)
    if wildcards.strategy == "pvae_stepwise":
        return base_view(
            wildcards.dataset,
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
            "restricted",
            split,
        )
    if wildcards.strategy.startswith("pvae_"):
        return restored_view(
            wildcards.dataset,
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
            wildcards.strategy,
            split,
        )
    return base_view(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
        wildcards.strategy,
        split,
    )


def incomplete_pvae(dataset, mechanism, probability, instance):
    return (
        f"{ROOT}/restoration_pvae/{NAMESPACE}/incomplete/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/"
        "model.bundle"
    )


def oracle_pvae(dataset, instance):
    return (
        f"{ROOT}/restoration_pvae/{NAMESPACE}/oracle/dataset-{dataset}/"
        f"instance-{instance}/model.bundle"
    )


def method_pretrain(wildcards):
    key = METHOD_SPECS[wildcards.method].pretrained_model_name
    strategy = (
        "restricted"
        if wildcards.strategy == "pvae_stepwise"
        else wildcards.strategy
    )
    return (
        f"{ROOT}/pretrained/{NAMESPACE}/{key}/dataset-{wildcards.dataset}/"
        f"mechanism-{wildcards.mechanism}+p-{wildcards.p}+"
        f"strategy-{strategy}+instance-{wildcards.instance}/"
        "model.bundle"
    )


def trained_method(
    dataset,
    method,
    mechanism,
    probability,
    strategy,
    instance,
    train_budget,
):
    return (
        f"{ROOT}/trained/{NAMESPACE}/{method}/dataset-{dataset}/"
        f"mechanism-{mechanism}+p-{probability}+strategy-{strategy}+"
        f"instance-{instance}+train_hard_budget-{train_budget}/method.bundle"
    )


def trained_method_input(wildcards):
    return trained_method(
        wildcards.dataset,
        wildcards.method,
        wildcards.mechanism,
        wildcards.p,
        wildcards.strategy,
        wildcards.instance,
        wildcards.train_budget,
    )


def evaluation_path(
    dataset,
    method,
    mechanism,
    probability,
    strategy,
    instance,
    train_budget,
    eval_budget,
):
    return (
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{dataset}/"
        f"method-{method}+mechanism-{mechanism}+p-{probability}+"
        f"strategy-{strategy}+instance-{instance}+"
        f"train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "eval_data.parquet"
    )


BUDGETS = {}
for method in BASE_METHODS:
    for dataset in DATASETS:
        if dataset in HARD_BUDGET_IGNORED_DATASETS[method]:
            continue
        budget = largest_hard_budget(_config["BUDGET_PARAMS"][method][dataset])
        if budget is not None:
            BUDGETS[(method, dataset)] = budget


def experiment_matrix():
    rows = []
    for dataset in DATASETS:
        for method, spec in METHOD_SPECS.items():
            budget = BUDGETS.get((spec.base_method, dataset))
            if budget is None:
                continue
            train_budget, eval_budget = budget
            if INCLUDE_COMPLETE_DATA and spec.include_complete_data:
                for instance in INSTANCES:
                    rows.append(
                        (
                            dataset,
                            method,
                            "none",
                            "0.0",
                            "complete",
                            instance,
                            train_budget,
                            eval_budget,
                        )
                    )
            strategies = (
                list(spec.allowed_strategies)
                if spec.allowed_strategies is not None
                else COMMON_STRATEGIES
            )
            strategies = list(strategies) + list(spec.extra_strategies)
            for mechanism, probability in MISSING_COMBINATIONS:
                for instance in INSTANCES:
                    for strategy in dict.fromkeys(strategies):
                        if not strategy_enabled(
                            STRATEGY_FILTERS,
                            strategy,
                            dataset=dataset,
                            method=method,
                            base_method=spec.base_method,
                            mechanism=mechanism,
                            probability=probability,
                        ):
                            continue
                        rows.append(
                            (
                                dataset,
                                method,
                                mechanism,
                                probability,
                                strategy,
                                instance,
                                train_budget,
                                eval_budget,
                            )
                        )
    return rows


EXPERIMENTS = experiment_matrix()
if not EXPERIMENTS:
    raise ValueError("The selected methods and datasets produce no experiments")
EVALUATIONS = [evaluation_path(*row) for row in EXPERIMENTS]
SUMMARY_DIR = f"{ROOT}/summary/{EVAL_SPLIT}/{NAMESPACE}"
FIGURE_DIR = f"{ROOT}/figures/{EVAL_SPLIT}/{NAMESPACE}"
ANALYSIS_OUTPUTS = (
    [
        f"{ROOT}/analysis/path_fidelity_{NAMESPACE}_{EVAL_SPLIT}.csv",
        f"{ROOT}/analysis/stepwise_effects_{NAMESPACE}_{EVAL_SPLIT}.csv",
        f"{ROOT}/analysis/generator_quality_{NAMESPACE}_{EVAL_SPLIT}.csv",
        f"{ROOT}/analysis/dime_invariance_{NAMESPACE}_{EVAL_SPLIT}.csv",
    ]
    if RUN_CONFIRMATORY_ANALYSIS
    else []
)
MECHANISM_FIGURE_DIR = (
    f"{ROOT}/analysis_figures/{EVAL_SPLIT}/{NAMESPACE}"
    if RUN_CONFIRMATORY_ANALYSIS
    else []
)

configured_strategies = ["complete", *COMMON_STRATEGIES]
for spec in METHOD_SPECS.values():
    configured_strategies.extend(spec.allowed_strategies or ())
    configured_strategies.extend(spec.extra_strategies)

wildcard_constraints:
    dataset=wildcard_pattern(DATASETS),
    mechanism=wildcard_pattern(["none", *MECHANISMS]),
    p=wildcard_pattern(["0.0", *PROBABILITIES]),
    strategy=wildcard_pattern(configured_strategies),
    method=wildcard_pattern(METHOD_SPECS),
    pretrain_key=wildcard_pattern(PRETRAIN_SCRIPT_NAMES),
    train_budget=r"[0-9.]+",
    eval_budget=r"[0-9.]+"


# Aggregation is seconds of pandas over files the submitting host can already
# read. On a cluster each of these would otherwise cost a full queue round trip.
localrules:
    all,
    analyze_missing_data_mechanisms,
    plot_missing_data_mechanisms,
    summarize_missing_data,
    plot_missing_data,


# Arrhenius has x86_64 CPU nodes and aarch64 Grace Hopper GPU nodes, and
# Snakemake exports the submitting environment to both, which would put the
# login node's venv on the wrong architecture. This lets a cluster select the
# right one per job without any rule knowing about it. Empty everywhere else.
shell.prefix(os.environ.get("AFABENCH_SHELL_PREFIX", ""))


rule all:
    input:
        f"{SUMMARY_DIR}/instance_metrics.csv",
        f"{SUMMARY_DIR}/summary.csv",
        f"{SUMMARY_DIR}/action_rates.csv",
        f"{SUMMARY_DIR}/restoration_rmse.csv",
        FIGURE_DIR,
        ANALYSIS_OUTPUTS,
        MECHANISM_FIGURE_DIR,


rule analyze_missing_data_mechanisms:
    input:
        evaluations=EVALUATIONS,
        instances=f"{SUMMARY_DIR}/instance_metrics.csv",
        restoration=f"{SUMMARY_DIR}/restoration_rmse.csv",
        datasets=[
            raw_dataset(dataset, instance, EVAL_SPLIT)
            for dataset in DATASETS
            for instance in INSTANCES
        ],
    output:
        ANALYSIS_OUTPUTS,
    params:
        require_invariance=(
            "--require-dime-invariance"
            if RUN_CONFIRMATORY_ANALYSIS
            else ""
        ),
    shell:
        """
        python scripts/analysis/analyze_missing_data_mechanisms.py \
            --namespace={NAMESPACE} --split={EVAL_SPLIT} \
            {params.require_invariance}
        """


rule plot_missing_data_mechanisms:
    input:
        ANALYSIS_OUTPUTS,
    output:
        directory(MECHANISM_FIGURE_DIR),
    shell:
        """
        python scripts/plotting/plot_missing_data_mechanisms.py \
            --analysis-dir={ROOT}/analysis --namespace={NAMESPACE} \
            --split={EVAL_SPLIT} --output-dir={output}
        """


rule generate_missing_data_dataset:
    output:
        [
            directory(
                f"{ROOT}/datasets/{NAMESPACE}/{{dataset}}/{instance}/{split}.bundle"
            )
            for instance in INSTANCES
            for split in ["train", "val", "test"]
        ]
    benchmark:
        f"{BENCHMARK_ROOT}/generate_dataset/dataset-{{dataset}}.tsv"
    params:
        instances="[" + ",".join(str(value) for value in INSTANCES) + "]",
        seeds="[" + ",".join(str(value) for value in INSTANCES) + "]",
        save_path=lambda wc: f"{ROOT}/datasets/{NAMESPACE}/{wc.dataset}",
        extra=lambda wc: runtime_params(DATASET_GENERATION_PARAMS, wc.dataset),
    shell:
        """
        python scripts/dataset_generation/generate_dataset.py \
            dataset={wildcards.dataset} instance_indices={params.instances} \
            seeds={params.seeds} save_path={params.save_path} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_shared_classifier:
    input:
        train=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "train"
        ),
        val=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "val"
        ),
    output:
        directory(CLASSIFIER_OUTPUT),
    benchmark:
        CLASSIFIER_BENCHMARK
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset)),
    params:
        instance=classifier_instance,
        script=lambda wc: classifier_script_name(wc.dataset),
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset),
        extra=lambda wc: classifier_script_params(wc.dataset),
    shell:
        """
        python scripts/train_classifier/{params.script}.py \
            train_dataset_path={input.train} val_dataset_path={input.val} \
            save_path={output} components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={params.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_method_classifier:
    input:
        train=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "train"
        ),
        val=lambda wc: raw_dataset(
            wc.dataset, classifier_instance(wc), "val"
        ),
    output:
        directory(METHOD_CLASSIFIER_OUTPUT),
    benchmark:
        METHOD_CLASSIFIER_BENCHMARK
    resources:
        gpu=lambda wc: gpu_for_device(
            resolve_device(
                DEFAULT_DEVICE,
                DEVICE_OVERRIDES,
                dataset=wc.dataset,
                method=wc.base_method,
            )
        ),
    params:
        instance=classifier_instance,
        script=lambda wc: METHOD_CLASSIFIER_SCRIPT_NAMES[wc.base_method],
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: resolve_device(
            DEFAULT_DEVICE,
            DEVICE_OVERRIDES,
            dataset=wc.dataset,
            method=wc.base_method,
        ),
        extra=lambda wc: METHOD_CLASSIFIER_SCRIPT_PARAMS[wc.base_method],
    shell:
        """
        python scripts/train_classifier/{params.script}.py \
            train_dataset_path={input.train} val_dataset_path={input.val} \
            save_path={output} components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={params.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule materialize_missing_training_view:
    input:
        train=lambda wc: raw_dataset(wc.dataset, wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/base/{NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/base/{NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/val.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/materialize_view/dataset-{{dataset}}/"
        "mechanism-{mechanism}+p-{p}+instance-{instance}+"
        "strategy-{strategy}.tsv"
    params:
        p_obs=MISSINGNESS["p_obs"],
        p_params=MISSINGNESS["p_params"],
        exclude_inputs=str(bool(MISSINGNESS["exclude_inputs"])).lower(),
    shell:
        """
        python scripts/missing_values/materialize_training_views.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            train_save_path={output.train} val_save_path={output.val} \
            strategy={wildcards.strategy} seed={wildcards.instance} \
            missingness.mechanism={wildcards.mechanism} \
            missingness.p={wildcards.p} missingness.p_obs={params.p_obs} \
            missingness.p_params={params.p_params} \
            missingness.exclude_inputs={params.exclude_inputs} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule pretrain_incomplete_restoration_pvae:
    input:
        train=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/restoration_pvae/{NAMESPACE}/incomplete/"
            "dataset-{dataset}/mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_restoration_pvae/incomplete/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
        extra=lambda wc: runtime_params(
            RESTORATION_PVAE_PARAMS, wc.dataset
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            respect_source_availability=true \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule pretrain_oracle_restoration_pvae:
    input:
        train=lambda wc: raw_dataset(wc.dataset, wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/restoration_pvae/{NAMESPACE}/oracle/"
            "dataset-{dataset}/instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_restoration_pvae/oracle/"
        "dataset-{dataset}+instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
        extra=lambda wc: runtime_params(
            RESTORATION_PVAE_PARAMS, wc.dataset
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} \
            respect_source_availability=false \
            experiment@_global_={wildcards.dataset} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def restoration_pvae_input(wildcards):
    if wildcards.strategy == "pvae_oracle":
        return oracle_pvae(wildcards.dataset, wildcards.instance)
    return incomplete_pvae(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
    )


rule restore_missing_training_view:
    input:
        train=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.dataset, wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        pvae=restoration_pvae_input,
        reference_train=lambda wc: raw_dataset(
            wc.dataset, wc.instance, "train"
        ),
        reference_val=lambda wc: raw_dataset(wc.dataset, wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/restored/{NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/restored/{NAMESPACE}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/"
            "{strategy}/val.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/restore_view/dataset-{{dataset}}/"
        "mechanism-{mechanism}+p-{p}+instance-{instance}+"
        "strategy-{strategy}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(dataset_device(wc.dataset, "pvae")),
    params:
        device=lambda wc: dataset_device(wc.dataset, "pvae"),
    shell:
        """
        python scripts/missing_values/restore_training_views.py \
            train_view_bundle_path={input.train} \
            val_view_bundle_path={input.val} pvae_bundle_path={input.pvae} \
            train_save_path={output.train} val_save_path={output.val} \
            strategy={wildcards.strategy} seed={wildcards.instance} \
            batch_size={RESTORATION_BATCH_SIZE} device={params.device} \
            reference_train_dataset_bundle_path={input.reference_train} \
            reference_val_dataset_bundle_path={input.reference_val} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def pretrain_extra(wildcards):
    key = wildcards.pretrain_key
    script = PRETRAIN_SCRIPT_NAMES[key]
    params = [PRETRAIN_PARAMS[key]] if PRETRAIN_PARAMS[key] else []
    if script == "odin":
        respect = wildcards.strategy == "restricted"
        params.append(
            f"respect_source_availability={str(respect).lower()}"
        )
    params.append(f"experiment@_global_={wildcards.dataset}")
    runtime = runtime_params(
        PRETRAIN_RUNTIME_PARAMS,
        key,
        script,
        wildcards.dataset,
        f"{key}/{wildcards.dataset}",
    )
    if runtime:
        params.append(runtime)
    return " ".join(params)


rule pretrain_missing_data_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/pretrained/{NAMESPACE}/{{pretrain_key}}/"
            "dataset-{dataset}/mechanism-{mechanism}+p-{p}+"
            "strategy-{strategy}+instance-{instance}/model.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/pretrain_method/pretrain-{{pretrain_key}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(
            dataset_device(wc.dataset, wc.pretrain_key)
        ),
    params:
        script=lambda wc: PRETRAIN_SCRIPT_NAMES[wc.pretrain_key],
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: dataset_device(wc.dataset, wc.pretrain_key),
        extra=pretrain_extra,
    shell:
        """
        python scripts/pretrain_model/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def training_extra(wildcards):
    spec = METHOD_SPECS[wildcards.method]
    params = list(spec.train_params)
    params.append(f"experiment@_global_={wildcards.dataset}")
    if wildcards.strategy == "pvae_stepwise":
        params.append(
            "stepwise_pvae_bundle_path="
            + incomplete_pvae(
                wildcards.dataset,
                wildcards.mechanism,
                wildcards.p,
                wildcards.instance,
            )
        )
    runtime = runtime_params(
        TRAIN_RUNTIME_PARAMS,
        spec.base_method,
        wildcards.method,
        wildcards.dataset,
        f"{wildcards.method}/{wildcards.dataset}",
    )
    if runtime:
        params.append(runtime)
    return " ".join(params)


def stepwise_pvae_input(wildcards):
    if wildcards.strategy != "pvae_stepwise":
        return []
    return incomplete_pvae(
        wildcards.dataset,
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
    )


rule train_missing_data_method_with_pretraining:
    wildcard_constraints:
        method=wildcard_pattern(PRETRAINED_METHODS)
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        pretrained=method_pretrain,
        stepwise_pvae=stepwise_pvae_input,
        classifier=lambda wc: classifier_path(wc.dataset, wc.instance),
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/{{method}}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}+train_hard_budget-{train_budget}/"
            "method.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/train_method/method-{{method}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        script=lambda wc: METHOD_SPECS[wc.method].train_script_name,
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        extra=training_extra,
    shell:
        """
        python scripts/train_method/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            pretrained_model_bundle_path={input.pretrained} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_budget} soft_budget_param=null \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule train_missing_data_method_without_pretraining:
    wildcard_constraints:
        method=wildcard_pattern(UNPRETRAINED_METHODS)
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=lambda wc: classifier_path(
            wc.dataset, wc.instance, wc.method
        ),
        stepwise_pvae=stepwise_pvae_input,
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/{{method}}/dataset-{{dataset}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}+train_hard_budget-{train_budget}/"
            "method.bundle"
        ),
    benchmark:
        f"{BENCHMARK_ROOT}/train_method/method-{{method}}/"
        "dataset-{dataset}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        script=lambda wc: METHOD_SPECS[wc.method].train_script_name,
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        extra=training_extra,
    shell:
        """
        python scripts/train_method/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.train_budget} soft_budget_param=null \
            device={params.device} seed={wildcards.instance} \
            use_wandb={USE_WANDB} smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


def eval_batch_size(wildcards):
    if wildcards.strategy == "pvae_stepwise":
        return STEPWISE_EVAL_BATCH_SIZE
    if EVAL_BATCH_SIZE_OVERRIDE is not None:
        if isinstance(EVAL_BATCH_SIZE_OVERRIDE, dict):
            return int(
                EVAL_BATCH_SIZE_OVERRIDE.get(
                    wildcards.dataset,
                    EVAL_BATCH_SIZE_OVERRIDE.get("default"),
                )
            )
        return int(EVAL_BATCH_SIZE_OVERRIDE)
    base_method = METHOD_SPECS[wildcards.method].base_method
    return int(EVAL_BATCH_SIZES[base_method][wildcards.dataset])


rule eval_missing_data_method:
    input:
        dataset=lambda wc: raw_dataset(wc.dataset, wc.instance, EVAL_SPLIT),
        method=trained_method_input,
        classifier=lambda wc: classifier_path(
            wc.dataset, wc.instance, wc.method
        ),
    output:
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{{dataset}}/"
        "method-{method}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "eval_data.parquet",
    # Total subprocess wall time, plus max RSS and CPU time. `benchmark:` rather
    # than a second `output:` so the DAG's completeness check is unaffected.
    # Subtract the eval_data.timing.json phases from this to get fixed startup.
    benchmark:
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/dataset-{{dataset}}/"
        "method-{method}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}+"
        "train_hard_budget-{train_budget}+eval_hard_budget-{eval_budget}/"
        "benchmark.tsv"
    resources:
        gpu=lambda wc: gpu_for_device(method_device(wc.dataset, wc.method)),
    params:
        unmasker=lambda wc: UNMASKERS[wc.dataset],
        device=lambda wc: method_device(wc.dataset, wc.method),
        batch_size=eval_batch_size,
        respect_native=lambda wc: str(wc.mechanism == "native").lower(),
        extra=lambda wc: runtime_params(
            EVAL_PARAMS,
            METHOD_SPECS[wc.method].base_method,
            wc.method,
            wc.dataset,
            f"{wc.method}/{wc.dataset}",
        ),
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input.method} dataset_bundle_path={input.dataset} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={params.unmasker} \
            hard_budget={wildcards.eval_budget} soft_budget_param=null \
            batch_size={params.batch_size} device={params.device} \
            respect_native_availability={params.respect_native} \
            seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {params.extra} \
            {HYDRA_WORKFLOW_OVERRIDES}
        """


rule summarize_missing_data:
    input:
        EVALUATIONS,
    output:
        instances=f"{SUMMARY_DIR}/instance_metrics.csv",
        summary=f"{SUMMARY_DIR}/summary.csv",
        actions=f"{SUMMARY_DIR}/action_rates.csv",
        restoration=f"{SUMMARY_DIR}/restoration_rmse.csv",
    params:
        root=f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}",
        restored=f"{ROOT}/views/restored/{NAMESPACE}",
        variants=" ".join(
            f"--base-method {name}={spec.base_method}"
            for name, spec in METHOD_SPECS.items()
            if name != spec.base_method
        ),
    shell:
        """
        python scripts/analysis/summarize_missing_data.py \
            --input-root {params.root} --restored-root {params.restored} \
            --instance-output {output.instances} \
            --summary-output {output.summary} \
            --action-output {output.actions} \
            --restoration-output {output.restoration} {params.variants}
        """


rule plot_missing_data:
    input:
        instances=f"{SUMMARY_DIR}/instance_metrics.csv",
        summary=f"{SUMMARY_DIR}/summary.csv",
        actions=f"{SUMMARY_DIR}/action_rates.csv",
        restoration=f"{SUMMARY_DIR}/restoration_rmse.csv",
    output:
        directory(FIGURE_DIR),
    resources:
        shell_exec="bash"
    shell:
        """
        python scripts/plotting/plot_missing_data.py \
            instance_metrics={input.instances} summary={input.summary} \
            action_rates={input.actions} restoration_rmse={input.restoration} \
            output_folder={output} formats='[pdf,svg]' \
            {HYDRA_WORKFLOW_OVERRIDES}
        """
