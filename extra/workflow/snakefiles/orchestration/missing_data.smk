"""Config-driven experiments with missing values in training data only.

This workflow intentionally does not merge missingness into the ordinary main
pipeline. It materializes immutable train/validation views, trains every method
against those views, and always evaluates from a cold start on a complete
validation or test bundle.
"""

import re

required_config = {
    "artifact_namespace",
    "dataset",
    "dataset_instance_indices",
    "device",
    "eval_dataset_split",
    "method_options",
    "methods",
    "missingness",
    "pretrain_options",
    "strategies",
}
missing_config = sorted(required_config - set(config))
if missing_config:
    raise ValueError(
        "Missing required missing-data config keys: " + ", ".join(missing_config)
    )

NAMESPACE = str(config["artifact_namespace"])
DATASET = str(config["dataset"])
INSTANCES = [int(value) for value in config["dataset_instance_indices"]]
if not INSTANCES:
    raise ValueError("dataset_instance_indices must not be empty")
DATASET_SEEDS = [
    int(value) for value in config.get("dataset_seeds", INSTANCES)
]
if len(DATASET_SEEDS) != len(INSTANCES):
    raise ValueError("dataset_seeds must align with dataset_instance_indices")
CLASSIFIER_INSTANCE = INSTANCES[0]
CLASSIFIER_SEED = DATASET_SEEDS[0]
DEVICE = str(config["device"])
USE_WANDB = str(bool(config.get("use_wandb", False))).lower()
SMOKE_TEST = bool(config.get("smoke_test", False))
SMOKE_TEST_STR = str(SMOKE_TEST).lower()
HARD_BUDGET = int(config.get("hard_budget", 14))
EVAL_BATCH_SIZE = int(config.get("eval_batch_size", 128))
RESTORATION_BATCH_SIZE = int(config.get("restoration_batch_size", 1024))
UNMASKER = str(config.get("unmasker", DATASET))
INITIALIZER = str(config.get("initializer", "cold"))
EVAL_SPLIT = str(config["eval_dataset_split"])
if EVAL_SPLIT not in {"val", "test"}:
    raise ValueError("eval_dataset_split must be either 'val' or 'test'")
ROOT = "extra/output/missing_data"
CLASSIFIER = f"{ROOT}/classifier/{NAMESPACE}/dataset-{DATASET}.bundle"
CLASSIFIER_SCRIPT = str(config.get("classifier_script", "masked_mlp_classifier"))

DATASET_GENERATION_PARAMS = " ".join(
    str(value) for value in config.get("dataset_generation_params", [])
)
CLASSIFIER_PARAMS = " ".join(
    str(value) for value in config.get("classifier_params", [])
)
RESTORATION_PVAE_PARAMS = " ".join(
    str(value) for value in config.get("restoration_pvae_params", [])
)
PRETRAIN_RUNTIME_PARAMS = config.get("pretrain_runtime_params", {})
TRAIN_RUNTIME_PARAMS = config.get("train_runtime_params", {})
EVAL_PARAMS = " ".join(str(value) for value in config.get("eval_params", []))

MISSINGNESS = config["missingness"]
MECHANISMS = list(MISSINGNESS["mechanisms"])
PROBABILITIES = [str(value) for value in MISSINGNESS["probabilities"]]
MISSING_COMBINATIONS = [
    (mechanism, probability)
    for mechanism in MECHANISMS
    for probability in PROBABILITIES
]

METHODS = list(config["methods"])
COMMON_STRATEGIES = list(config["strategies"])
INCLUDE_COMPLETE_DATA = bool(config.get("include_complete_data", True))
METHOD_OPTIONS = config["method_options"]
PRETRAIN_OPTIONS = config["pretrain_options"]
unknown_methods = sorted(set(METHODS) - set(METHOD_OPTIONS))
if unknown_methods:
    raise ValueError("Unknown methods: " + ", ".join(unknown_methods))


def wildcard_pattern(values):
    unique_values = dict.fromkeys(str(value) for value in values)
    return "(?:" + "|".join(re.escape(value) for value in unique_values) + ")"


configured_strategies = ["complete", *COMMON_STRATEGIES]
for method in METHODS:
    options = METHOD_OPTIONS[method]
    configured_strategies.extend(options.get("allowed_strategies", []))
    configured_strategies.extend(options.get("extra_strategies", []))

MECHANISM_PATTERN = wildcard_pattern(["none", *MECHANISMS])
PROBABILITY_PATTERN = wildcard_pattern(["0.0", *PROBABILITIES])
STRATEGY_PATTERN = wildcard_pattern(configured_strategies)
METHOD_PATTERN = wildcard_pattern(METHODS)
PRETRAIN_PATTERN = wildcard_pattern(PRETRAIN_OPTIONS)


def runtime_params(mapping, key):
    return " ".join(
        [str(value) for value in mapping.get("default", [])]
        + [str(value) for value in mapping.get(key, [])]
    )


def raw_dataset(instance, split):
    return (
        f"{ROOT}/datasets/{NAMESPACE}/{DATASET}/"
        f"{instance}/{split}.bundle"
    )


def base_view(mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/base/{NAMESPACE}/"
        f"mechanism-{mechanism}+p-{probability}/"
        f"instance-{instance}/{strategy}/{split}.bundle"
    )


def restored_view(mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/restored/{NAMESPACE}/"
        f"mechanism-{mechanism}+p-{probability}/"
        f"instance-{instance}/{strategy}/{split}.bundle"
    )


def training_view(wildcards, split):
    if wildcards.strategy == "complete":
        return raw_dataset(wildcards.instance, split)
    if wildcards.strategy.startswith("pvae_"):
        return restored_view(
            wildcards.mechanism,
            wildcards.p,
            wildcards.instance,
            wildcards.strategy,
            split,
        )
    return base_view(
        wildcards.mechanism,
        wildcards.p,
        wildcards.instance,
        wildcards.strategy,
        split,
    )


def incomplete_pvae(mechanism, probability, instance):
    return (
        f"{ROOT}/restoration_pvae/{NAMESPACE}/incomplete/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/model.bundle"
    )


def oracle_pvae(instance):
    return (
        f"{ROOT}/restoration_pvae/{NAMESPACE}/oracle/"
        f"instance-{instance}/model.bundle"
    )


def method_pretrain(wildcards):
    pretrain_key = METHOD_OPTIONS[wildcards.method]["pretrained_model_name"]
    return (
        f"{ROOT}/pretrained/{NAMESPACE}/{pretrain_key}/"
        f"mechanism-{wildcards.mechanism}+p-{wildcards.p}+"
        f"strategy-{wildcards.strategy}+instance-{wildcards.instance}/model.bundle"
    )


def trained_method(method, mechanism, probability, strategy, instance):
    family = METHOD_OPTIONS[method]["family"]
    return (
        f"{ROOT}/trained/{NAMESPACE}/{family}/{method}/"
        f"mechanism-{mechanism}+p-{probability}+strategy-{strategy}+"
        f"instance-{instance}/method.bundle"
    )


def trained_method_input(wildcards):
    return trained_method(
        wildcards.method,
        wildcards.mechanism,
        wildcards.p,
        wildcards.strategy,
        wildcards.instance,
    )


def evaluation_path(method, mechanism, probability, strategy, instance):
    return (
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/"
        f"method-{method}+mechanism-{mechanism}+p-{probability}+"
        f"strategy-{strategy}+instance-{instance}/eval_data.parquet"
    )


def experiment_matrix():
    rows = []
    if INCLUDE_COMPLETE_DATA:
        for instance in INSTANCES:
            for method in METHODS:
                options = METHOD_OPTIONS[method]
                if options.get("include_complete_data", True):
                    rows.append(
                        (method, "none", "0.0", "complete", instance)
                    )
    for mechanism, probability in MISSING_COMBINATIONS:
        for instance in INSTANCES:
            for method in METHODS:
                options = METHOD_OPTIONS[method]
                strategies = options.get(
                    "allowed_strategies", COMMON_STRATEGIES
                )
                strategies = list(strategies) + list(
                    options.get("extra_strategies", [])
                )
                for strategy in dict.fromkeys(strategies):
                    rows.append(
                        (method, mechanism, probability, strategy, instance)
                    )
    return rows


EXPERIMENTS = experiment_matrix()
EVALUATIONS = [evaluation_path(*row) for row in EXPERIMENTS]
SUMMARY_DIR = f"{ROOT}/summary/{EVAL_SPLIT}/{NAMESPACE}"

wildcard_constraints:
    mechanism=MECHANISM_PATTERN,
    p=PROBABILITY_PATTERN,
    strategy=STRATEGY_PATTERN,
    method=METHOD_PATTERN,
    pretrain_key=PRETRAIN_PATTERN


rule all:
    input:
        f"{SUMMARY_DIR}/instance_metrics.csv",
        f"{SUMMARY_DIR}/summary.csv",
        f"{SUMMARY_DIR}/action_rates.csv",
        f"{SUMMARY_DIR}/restoration_rmse.csv",


rule generate_missing_data_dataset:
    output:
        [
            directory(raw_dataset(instance, split))
            for instance in INSTANCES
            for split in ["train", "val", "test"]
        ]
    params:
        instances="[" + ",".join(str(value) for value in INSTANCES) + "]",
        seeds="[" + ",".join(str(value) for value in DATASET_SEEDS) + "]",
        save_path=f"{ROOT}/datasets/{NAMESPACE}/{DATASET}",
        extra=DATASET_GENERATION_PARAMS,
    shell:
        """
        python scripts/dataset_generation/generate_dataset.py \
            dataset={DATASET} \
            instance_indices={params.instances} \
            seeds={params.seeds} \
            save_path={params.save_path} \
            {params.extra}
        """


rule train_missing_data_shared_classifier:
    input:
        train=raw_dataset(CLASSIFIER_INSTANCE, "train"),
        val=raw_dataset(CLASSIFIER_INSTANCE, "val"),
    output:
        directory(CLASSIFIER),
    shell:
        """
        python scripts/train_classifier/{CLASSIFIER_SCRIPT}.py \
            train_dataset_path={input.train} \
            val_dataset_path={input.val} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            device={DEVICE} seed={CLASSIFIER_SEED} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} experiment@_global_={DATASET} \
            {CLASSIFIER_PARAMS}
        """


rule materialize_missing_training_view:
    input:
        train=lambda wc: raw_dataset(wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/base/{NAMESPACE}/"
            "mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/base/{NAMESPACE}/"
            "mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/{strategy}/val.bundle"
        ),
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
            missingness.p={wildcards.p} \
            missingness.p_obs={params.p_obs} \
            missingness.p_params={params.p_params} \
            missingness.exclude_inputs={params.exclude_inputs}
        """


rule pretrain_incomplete_restoration_pvae:
    input:
        train=lambda wc: base_view(
            wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/restoration_pvae/{NAMESPACE}/incomplete/"
            "mechanism-{mechanism}+p-{p}/instance-{instance}/model.bundle"
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            device={DEVICE} seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} respect_source_availability=true \
            experiment@_global_={DATASET} {RESTORATION_PVAE_PARAMS}
        """


rule pretrain_oracle_restoration_pvae:
    input:
        train=lambda wc: raw_dataset(wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.instance, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/restoration_pvae/{NAMESPACE}/oracle/"
            "instance-{instance}/model.bundle"
        ),
    shell:
        """
        python scripts/pretrain_model/odin.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            device={DEVICE} seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} respect_source_availability=false \
            experiment@_global_={DATASET} {RESTORATION_PVAE_PARAMS}
        """


def restoration_pvae_input(wildcards):
    if wildcards.strategy == "pvae_oracle":
        return oracle_pvae(wildcards.instance)
    return incomplete_pvae(wildcards.mechanism, wildcards.p, wildcards.instance)


rule restore_missing_training_view:
    input:
        train=lambda wc: base_view(
            wc.mechanism, wc.p, wc.instance, "restricted", "train"
        ),
        val=lambda wc: base_view(
            wc.mechanism, wc.p, wc.instance, "restricted", "val"
        ),
        pvae=restoration_pvae_input,
        reference_train=lambda wc: raw_dataset(wc.instance, "train"),
        reference_val=lambda wc: raw_dataset(wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/restored/{NAMESPACE}/"
            "mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/restored/{NAMESPACE}/"
            "mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/{strategy}/val.bundle"
        ),
    shell:
        """
        python scripts/missing_values/restore_training_views.py \
            train_view_bundle_path={input.train} \
            val_view_bundle_path={input.val} pvae_bundle_path={input.pvae} \
            train_save_path={output.train} val_save_path={output.val} \
            strategy={wildcards.strategy} seed={wildcards.instance} \
            batch_size={RESTORATION_BATCH_SIZE} device={DEVICE} \
            reference_train_dataset_bundle_path={input.reference_train} \
            reference_val_dataset_bundle_path={input.reference_val}
        """


def pretrain_extra(wildcards):
    key = wildcards.pretrain_key
    options = PRETRAIN_OPTIONS[key]
    params = [str(value) for value in options.get("pretrain_params", [])]
    if key == "odin":
        respect = wildcards.strategy == "restricted"
        params.append(f"respect_source_availability={str(respect).lower()}")
    if options.get("use_experiment_config", False):
        params.append(f"experiment@_global_={DATASET}")
    runtime = runtime_params(PRETRAIN_RUNTIME_PARAMS, key)
    if runtime:
        params.append(runtime)
    return " ".join(params)


rule pretrain_missing_data_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/pretrained/{NAMESPACE}/{{pretrain_key}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}/model.bundle"
        ),
    params:
        script=lambda wc: PRETRAIN_OPTIONS[wc.pretrain_key]["script_name"],
        extra=pretrain_extra,
    shell:
        """
        python scripts/pretrain_model/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            device={DEVICE} seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {params.extra}
        """


def learned_training_extra(wildcards):
    options = METHOD_OPTIONS[wildcards.method]
    params = [str(value) for value in options.get("train_params", [])]
    if options.get("use_experiment_config", False):
        params.append(f"experiment@_global_={DATASET}")
    runtime = runtime_params(TRAIN_RUNTIME_PARAMS, wildcards.method)
    if runtime:
        params.append(runtime)
    return " ".join(params)


rule train_missing_data_learned_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        pretrained=method_pretrain,
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/learned/{{method}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}/method.bundle"
        ),
    params:
        script=lambda wc: METHOD_OPTIONS[wc.method]["train_script_name"],
        extra=learned_training_extra,
    shell:
        """
        python scripts/train_method/{params.script}.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            pretrained_model_bundle_path={input.pretrained} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            hard_budget={HARD_BUDGET} soft_budget_param=null \
            device={DEVICE} seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {params.extra}
        """


def aaco_training_extra(wildcards):
    options = METHOD_OPTIONS[wildcards.method]
    params = [str(value) for value in options.get("train_params", [])]
    if options.get("use_experiment_config", False):
        params.append(f"experiment@_global_={DATASET}")
    runtime = runtime_params(TRAIN_RUNTIME_PARAMS, wildcards.method)
    if runtime:
        params.append(runtime)
    return " ".join(params)


rule train_missing_data_aaco:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/trained/{NAMESPACE}/aaco/{{method}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}/method.bundle"
        ),
    params:
        extra=aaco_training_extra,
    shell:
        """
        python scripts/train_method/aaco.py \
            train_dataset_bundle_path={input.train} \
            val_dataset_bundle_path={input.val} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            hard_budget={HARD_BUDGET} soft_budget_param=null \
            device={DEVICE} seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {params.extra}
        """


rule eval_missing_data_method:
    input:
        dataset=lambda wc: raw_dataset(wc.instance, EVAL_SPLIT),
        method=trained_method_input,
        classifier=CLASSIFIER,
    output:
        f"{ROOT}/eval/{EVAL_SPLIT}/{NAMESPACE}/"
        "method-{method}+mechanism-{mechanism}+p-{p}+"
        "strategy-{strategy}+instance-{instance}/eval_data.parquet",
    shell:
        """
        python scripts/eval/eval_afa_method.py \
            method_bundle_path={input.method} \
            dataset_bundle_path={input.dataset} \
            classifier_bundle_path={input.classifier} save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            hard_budget={HARD_BUDGET} soft_budget_param=null \
            batch_size={EVAL_BATCH_SIZE} device={DEVICE} \
            seed={wildcards.instance} use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} {EVAL_PARAMS}
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
    shell:
        """
        python scripts/analysis/summarize_missing_data.py \
            --input-root {params.root} \
            --instance-output {output.instances} \
            --summary-output {output.summary} \
            --action-output {output.actions} \
            --restoration-output {output.restoration}
        """
