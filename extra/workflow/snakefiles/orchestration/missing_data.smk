"""CUBE-NM experiments with missing values in training data only.

This workflow intentionally does not merge missingness into the ordinary main
pipeline. It materializes immutable train/validation views, trains every method
against those views, and always evaluates from a cold start on a complete
validation or test bundle.
"""

configfile: "extra/workflow/conf/missing_data.yaml"


STAGE = int(config.get("stage", 1))
if STAGE not in {1, 2, 3, 4}:
    raise ValueError("missing-data stage must be one of 1, 2, 3, or 4")

DATASET = config.get("dataset", "cube_nm")
if DATASET != "cube_nm":
    raise ValueError("the missing-data study is intentionally scoped to cube_nm")

ALL_INSTANCES = [int(value) for value in config["dataset_instance_indices"]]
INSTANCES = ALL_INSTANCES[:1] if STAGE == 1 else ALL_INSTANCES
DEVICE = config.get("device", "cuda")
USE_WANDB = str(bool(config.get("use_wandb", False))).lower()
configured_smoke = config.get("smoke_test")
SMOKE_TEST = STAGE == 1 if configured_smoke is None else bool(configured_smoke)
SMOKE_TEST_STR = str(SMOKE_TEST).lower()
HARD_BUDGET = int(config.get("hard_budget", 14))
EVAL_BATCH_SIZE = int(config.get("eval_batch_size", 128))
UNMASKER = config.get("unmasker", "cube_nm")
INITIALIZER = config.get("initializer", "cold")
MODE = "smoke" if STAGE == 1 else "full"
EVAL_SPLIT = "test" if STAGE == 4 else "val"
ROOT = "extra/output/missing_data"
CLASSIFIER = (
    f"{ROOT}/classifier/smoke/dataset-{DATASET}.bundle"
    if MODE == "smoke"
    else (
        f"extra/output/trained_classifiers/initializer-{INITIALIZER}/"
        f"dataset-{DATASET}.bundle"
    )
)

MISSINGNESS = config["missingness"]
MECHANISMS = list(MISSINGNESS["mechanisms"])
PROBABILITIES = [str(value) for value in MISSINGNESS["probabilities"]]
if STAGE == 1:
    MISSING_COMBINATIONS = [("mcar", "0.5")]
elif STAGE == 2:
    MISSING_COMBINATIONS = [(mechanism, "0.5") for mechanism in MECHANISMS]
else:
    MISSING_COMBINATIONS = [
        (mechanism, probability)
        for mechanism in MECHANISMS
        for probability in PROBABILITIES
    ]

BASE_METHODS = list(config["methods"])
COMMON_STRATEGIES = list(config["strategies"])
REWEIGHTING_CONTROLS = list(config["reweighting_controls"])
AACO_METHODS = {"aaco", "aaco_doubly_robust"}

METHOD_INFO = {
    "dime": {"script": "dime", "pretrain": "dime"},
    "dime_feature_marginal_ipw": {"script": "dime", "pretrain": "dime"},
    "gdfs": {"script": "gdfs", "pretrain": "gdfs"},
    "jafa": {"script": "jafa", "pretrain": "jafa"},
    "odin_model_free": {"script": "odin", "pretrain": "odin"},
    "odin_model_based": {"script": "odin", "pretrain": "odin"},
    "ol_without_mask": {"script": "ol", "pretrain": "ol_without_mask"},
    "ol_with_mask": {"script": "ol", "pretrain": "ol_with_mask"},
}
PRETRAIN_INFO = {
    "dime": {"script": "dime", "experiment": True},
    "gdfs": {"script": "gdfs", "experiment": True},
    "jafa": {"script": "jafa", "experiment": False},
    "odin": {"script": "odin", "experiment": True},
    "ol_without_mask": {"script": "ol", "experiment": False},
    "ol_with_mask": {"script": "ol", "experiment": False},
}


def raw_dataset(instance, split):
    return f"extra/output/datasets/{DATASET}/{instance}/{split}.bundle"


def base_view(mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/base/mechanism-{mechanism}+p-{probability}/"
        f"instance-{instance}/{strategy}/{split}.bundle"
    )


def restored_view(mechanism, probability, instance, strategy, split):
    return (
        f"{ROOT}/views/restored/{MODE}/"
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
        f"{ROOT}/restoration_pvae/{MODE}/incomplete/"
        f"mechanism-{mechanism}+p-{probability}/instance-{instance}/model.bundle"
    )


def oracle_pvae(instance):
    return (
        f"{ROOT}/restoration_pvae/{MODE}/oracle/"
        f"instance-{instance}/model.bundle"
    )


def method_pretrain(wildcards):
    pretrain_key = METHOD_INFO[wildcards.method]["pretrain"]
    return (
        f"{ROOT}/pretrained/{MODE}/{pretrain_key}/"
        f"mechanism-{wildcards.mechanism}+p-{wildcards.p}+"
        f"strategy-{wildcards.strategy}+instance-{wildcards.instance}/model.bundle"
    )


def trained_method(method, mechanism, probability, strategy, instance):
    family = "aaco" if method in AACO_METHODS else "learned"
    return (
        f"{ROOT}/trained/{MODE}/{family}/{method}/"
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
        f"{ROOT}/eval/{EVAL_SPLIT}/{MODE}/"
        f"method-{method}+mechanism-{mechanism}+p-{probability}+"
        f"strategy-{strategy}+instance-{instance}/eval_data.parquet"
    )


def experiment_matrix():
    rows = []
    if STAGE >= 2:
        for instance in INSTANCES:
            for method in BASE_METHODS:
                rows.append((method, "none", "0.0", "complete", instance))
    for mechanism, probability in MISSING_COMBINATIONS:
        for instance in INSTANCES:
            for method in BASE_METHODS:
                for strategy in COMMON_STRATEGIES:
                    rows.append(
                        (method, mechanism, probability, strategy, instance)
                    )
            # zero-fill is an AACO k-NN control only.
            rows.append(("aaco", mechanism, probability, "zero_fill", instance))
            if "aaco_doubly_robust" in REWEIGHTING_CONTROLS:
                rows.append(
                    (
                        "aaco_doubly_robust",
                        mechanism,
                        probability,
                        "restricted",
                        instance,
                    )
                )
            if "dime_feature_marginal_ipw" in REWEIGHTING_CONTROLS:
                rows.append(
                    (
                        "dime_feature_marginal_ipw",
                        mechanism,
                        probability,
                        "restricted",
                        instance,
                    )
                )
    return rows


EXPERIMENTS = experiment_matrix()
EVALUATIONS = [evaluation_path(*row) for row in EXPERIMENTS]
SUMMARY_DIR = f"{ROOT}/summary/{EVAL_SPLIT}/{MODE}"

wildcard_constraints:
    mechanism="none|mcar|mar|mnar_logistic|mnar_self",
    p="0\\.0|0\\.3|0\\.5|0\\.7",
    strategy=(
        "complete|restricted|mean_fill|zero_fill|pvae_label_conditioned|"
        "pvae_label_free|pvae_oracle|true_completion"
    ),
    method=(
        "aaco|aaco_doubly_robust|dime|dime_feature_marginal_ipw|gdfs|"
        "jafa|odin_model_free|odin_model_based|ol_without_mask|ol_with_mask"
    ),
    pretrain_key="dime|gdfs|jafa|odin|ol_without_mask|ol_with_mask"


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
        save_path=f"extra/output/datasets/{DATASET}",
    shell:
        """
        python scripts/dataset_generation/generate_dataset.py \
            dataset={DATASET} \
            instance_indices={params.instances} \
            seeds={params.instances} \
            save_path={params.save_path}
        """


rule train_missing_data_shared_classifier:
    input:
        train=raw_dataset(0, "train"),
        val=raw_dataset(0, "val"),
    output:
        directory(CLASSIFIER),
    shell:
        """
        python scripts/train_classifier/masked_mlp_classifier.py \
            train_dataset_path={input.train} \
            val_dataset_path={input.val} \
            save_path={output} \
            components/initializers@initializer={INITIALIZER} \
            components/unmaskers@unmasker={UNMASKER} \
            device={DEVICE} seed=0 use_wandb={USE_WANDB} \
            smoke_test={SMOKE_TEST_STR} experiment@_global_={DATASET}
        """


rule materialize_missing_training_view:
    input:
        train=lambda wc: raw_dataset(wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.instance, "val"),
    output:
        train=directory(
            f"{ROOT}/views/base/mechanism-{{mechanism}}+p-{{p}}/"
            "instance-{instance}/{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/base/mechanism-{{mechanism}}+p-{{p}}/"
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
            f"{ROOT}/restoration_pvae/{MODE}/incomplete/"
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
            experiment@_global_={DATASET}
        """


rule pretrain_oracle_restoration_pvae:
    input:
        train=lambda wc: raw_dataset(wc.instance, "train"),
        val=lambda wc: raw_dataset(wc.instance, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/restoration_pvae/{MODE}/oracle/"
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
            experiment@_global_={DATASET}
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
            f"{ROOT}/views/restored/{MODE}/"
            "mechanism-{mechanism}+p-{p}/"
            "instance-{instance}/{strategy}/train.bundle"
        ),
        val=directory(
            f"{ROOT}/views/restored/{MODE}/"
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
            batch_size=1024 device={DEVICE} \
            reference_train_dataset_bundle_path={input.reference_train} \
            reference_val_dataset_bundle_path={input.reference_val}
        """


def pretrain_extra(wildcards):
    key = wildcards.pretrain_key
    params = []
    if key == "odin":
        respect = wildcards.strategy == "restricted"
        params.append(f"respect_source_availability={str(respect).lower()}")
    if key == "ol_without_mask":
        params.append("pq_module.use_feature_mask=false")
    if key == "ol_with_mask":
        params.append("pq_module.use_feature_mask=true")
    if PRETRAIN_INFO[key]["experiment"]:
        params.append(f"experiment@_global_={DATASET}")
    return " ".join(params)


rule pretrain_missing_data_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/pretrained/{MODE}/{{pretrain_key}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}/model.bundle"
        ),
    params:
        script=lambda wc: PRETRAIN_INFO[wc.pretrain_key]["script"],
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
    params = []
    if wildcards.method == "odin_model_free":
        params.append("additional_generation_fraction=0.0")
    elif wildcards.method == "odin_model_based":
        params.append("additional_generation_fraction=1.0")
    elif wildcards.method == "dime_feature_marginal_ipw":
        params.append("ipw_mode=feature_marginal")
    if wildcards.method in {"dime", "dime_feature_marginal_ipw", "gdfs"}:
        params.append(f"experiment@_global_={DATASET}")
    return " ".join(params)


rule train_missing_data_learned_method:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        pretrained=method_pretrain,
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/trained/{MODE}/learned/{{method}}/"
            "mechanism-{mechanism}+p-{p}+strategy-{strategy}+"
            "instance-{instance}/method.bundle"
        ),
    params:
        script=lambda wc: METHOD_INFO[wc.method]["script"],
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
    objective = (
        "doubly_robust"
        if wildcards.method == "aaco_doubly_robust"
        else "support_aware"
    )
    return f"aco.missingness_objective={objective}"


rule train_missing_data_aaco:
    input:
        train=lambda wc: training_view(wc, "train"),
        val=lambda wc: training_view(wc, "val"),
        classifier=CLASSIFIER,
    output:
        directory(
            f"{ROOT}/trained/{MODE}/aaco/{{method}}/"
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
            smoke_test={SMOKE_TEST_STR} experiment@_global_={DATASET} \
            {params.extra}
        """


rule eval_missing_data_method:
    input:
        dataset=lambda wc: raw_dataset(wc.instance, EVAL_SPLIT),
        method=trained_method_input,
        classifier=CLASSIFIER,
    output:
        f"{ROOT}/eval/{EVAL_SPLIT}/{MODE}/"
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
            smoke_test={SMOKE_TEST_STR}
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
        root=f"{ROOT}/eval/{EVAL_SPLIT}/{MODE}",
    shell:
        """
        python scripts/analysis/summarize_missing_data.py \
            --input-root {params.root} \
            --instance-output {output.instances} \
            --summary-output {output.summary} \
            --action-output {output.actions} \
            --restoration-output {output.restoration}
        """
