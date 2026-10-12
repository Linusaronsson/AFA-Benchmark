"""
The workflow config loader: Snakemake's config, validated and typed.

`load_config` turns the merged Snakemake config into `WorkflowSettings`, the
values every orchestration Snakefile and the release manifest read. A config
mistake raises here, naming the method and the key, instead of surfacing as
a `KeyError` while Snakemake builds the DAG or as a silently different run.
The rules are documented in `docs/reference/pipeline_configuration.md`.
Snakemake imports this module at parse time, so it must not import torch.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import dacite

from afabench.core.checkout import require_checkout
from afabench.core.output_layout import (
    EvaluationRun,
    OutputLayout,
    PathValue,
    TrainingRun,
    default_output_root,
    pretrain_seed_folder,
    pretrain_seed_in_folder,
)

type BudgetParam = int | float | str
type NullableParam = BudgetParam | None
# (train_hard_budget, eval_hard_budget, train_soft_budget_param,
# eval_soft_budget_param), with "null" for an unset budget.
type BudgetCombination = tuple[
    BudgetParam, BudgetParam, BudgetParam, BudgetParam
]

# A dataset key is known when it has a file here.
DATASET_KEY_DIR = (
    require_checkout("the workflow settings") / "conf/components/dataset_key"
)


@dataclass(frozen=True, kw_only=True)
class ClassifierScript:
    """A classifier training script and its extra arguments."""

    script_name: str
    script_params: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class ScriptInvocation:
    """A pretraining or training script and the arguments it gets appended."""

    script_name: str
    # Space-joined, as the rule appends them to the command.
    script_params: str


@dataclass(frozen=True, kw_only=True)
class PretrainedModelOptions:
    """One `pretrain_mapping` entry."""

    pretrain_script_name: str
    pretrain_params: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class MethodOptions:
    """One `method_options` entry."""

    train_script_name: str
    pretrained_model_name: str | None = None
    method_specific_params: list[str] = field(default_factory=list)
    # A scalar for every dataset, or per dataset with a `default`; None
    # falls back to 1 with a warning.
    eval_batch_size: int | dict[str, int] | None = None
    hard_budget_ignored_datasets: list[str] = field(default_factory=list)
    soft_budget_ignored_datasets: list[str] = field(default_factory=list)
    # dataset -> eval hard budget -> train hard budget
    eval_to_train_hard_budget_mapping: dict[
        str, dict[int | float, int | float]
    ] = field(default_factory=dict)
    use_max_hard_budget_when_training_soft_budget: bool = False
    # The method's built-in classifier, trained instead of the external one.
    classifier: ClassifierScript | None = None


@dataclass(frozen=True, kw_only=True)
class PipelineStages:
    """
    The `stages` mapping: which optional pipeline stages define rules.

    A stage that is off defines no rules, so the run reads its outputs as
    they are on disk and never schedules its jobs.
    """

    dataset_generation: bool = True
    # Pretraining, training and classifier training.
    training: bool = True
    evaluation: bool = True


# The method set that always exists and holds every method
ALL_METHOD_SET = "all"
# The dataset set that always exists and holds every dataset of the run
ALL_DATASET_SET = "all"


@dataclass(frozen=True, kw_only=True)
class WorkflowSettings:
    stages: PipelineStages
    dataset_realization_indices: list[int]
    initializer: str
    eval_dataset_split: str
    use_wandb: bool
    # The W&B group of the invocation's W&B runs; set whenever use_wandb is
    wandb_group: str | None
    smoke_test: bool
    # Whether the terminal shows every job's messages, not only the job
    # counts and progress
    list_jobs: bool
    # Whether the run trains and evaluates in the soft-budget setting too,
    # not only in the hard-budget setting
    soft_budget_setting: bool
    # Where the run writes its artifacts and job records
    output_root: str
    # Only the pretrained models a selected method uses.
    pretrain_names: list[str]
    pretrain_script_names: dict[str, str]
    pretrain_params: dict[str, str]
    methods: list[str]
    method_train_script_names: dict[str, str]
    method_classifier_script_names: dict[str, str]
    method_classifier_script_params: dict[str, str]
    method_to_pretrained_model: dict[str, str]
    method_specific_params: dict[str, str]
    datasets: list[str]
    unmaskers: dict[str, str]
    # method -> dataset -> budget combinations
    budget_params: dict[str, dict[str, list[BudgetCombination]]]
    classifier_names: dict[str, ClassifierScript]
    method_sets: dict[str, list[str]]
    dataset_sets: dict[str, list[str]]
    eval_batch_sizes: dict[str, dict[str, int]]
    datasets_used_per_pretrain_name: dict[str, list[str]]

    def evaluation_run(
        self,
        *,
        method: str,
        dataset: str,
        dataset_realization_index: PathValue,
        budget_combination: BudgetCombination,
    ) -> EvaluationRun:
        """
        Name one evaluation of `method` on a dataset realization.

        Its pretraining, training and evaluation are all seeded with the
        dataset realization index, and its pretrain folder is that seed's
        only if the method has a pretraining stage.
        """
        (
            train_hard_budget,
            eval_hard_budget,
            train_soft_budget_param,
            eval_soft_budget_param,
        ) = budget_combination
        return EvaluationRun(
            training=TrainingRun(
                method=method,
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                pretrain_folder=pretrain_seed_folder(
                    dataset_realization_index
                    if method in self.method_to_pretrained_model
                    else None
                ),
                train_seed=dataset_realization_index,
                train_hard_budget=train_hard_budget,
                train_soft_budget_param=train_soft_budget_param,
            ),
            eval_seed=dataset_realization_index,
            eval_hard_budget=eval_hard_budget,
            eval_soft_budget_param=eval_soft_budget_param,
        )

    def classifier_bundle(
        self,
        layout: OutputLayout,
        *,
        method: str,
        dataset: str,
        dataset_realization_index: PathValue,
    ) -> str:
        """Address the classifier `method` uses on a dataset realization."""
        return layout.classifier_bundle(
            dataset=dataset,
            dataset_realization_index=dataset_realization_index,
            method=(
                method
                if method in self.method_classifier_script_names
                else None
            ),
        )

    def training_script(self, method: str) -> ScriptInvocation:
        return ScriptInvocation(
            script_name=self.method_train_script_names[method],
            script_params=self.method_specific_params[method],
        )

    def pretraining_script(
        self, pretrained_model_name: str
    ) -> ScriptInvocation:
        return ScriptInvocation(
            script_name=self.pretrain_script_names[pretrained_model_name],
            script_params=self.pretrain_params[pretrained_model_name],
        )

    def pretrained_model_bundle(
        self,
        layout: OutputLayout,
        *,
        method: str,
        dataset: str,
        dataset_realization_index: PathValue,
        pretrain_folder: str,
    ) -> list[str]:
        """
        Address the pretrained model a training run needs: one bundle or none.

        `pretrain_folder` must be the kind `evaluation_run` names for the
        method, a pretraining seed's folder exactly if it has a pretraining
        stage.
        """
        pretrained_model_name = self.method_to_pretrained_model.get(method)
        pretrain_seed = pretrain_seed_in_folder(pretrain_folder)
        if pretrain_seed is None:
            if pretrained_model_name is not None:
                message = (
                    f"Method {method!r} has a pretraining stage, so its "
                    f"bundles live under {pretrain_seed_folder('<seed>')}/, "
                    f"not {pretrain_folder}/."
                )
                raise ValueError(message)
            return []
        if pretrained_model_name is None:
            message = (
                f"Method {method!r} has no pretraining stage, so its "
                f"bundles live under {pretrain_seed_folder(None)}/, not "
                f"{pretrain_folder}/."
            )
            raise ValueError(message)
        return [
            layout.pretrained_model_bundle(
                pretrained_model_name=pretrained_model_name,
                dataset=dataset,
                dataset_realization_index=dataset_realization_index,
                pretrain_seed=pretrain_seed,
            )
        ]

    def unmasker(self, dataset: str) -> str:
        return self.unmaskers[dataset]

    def eval_batch_size(self, method: str, dataset: str) -> int:
        return self.eval_batch_sizes[method][dataset]

    def summary(self, wandb_group_url: str | None = None) -> str:
        """
        Describe what this run covers, for a reader checking a dry run.

        Each method is listed with its classifier and pretrained model, and
        each method and dataset with the hard budgets it is evaluated at and
        the soft-budget parameters it is trained and evaluated with. With
        W&B logging, the W&B group is linked by `wandb_group_url`, if given.
        """
        lines = [
            "Resolved workflow configuration:",
            f"  output root: {self.output_root}",
            "  stages off: "
            + _join(
                [
                    stage.name
                    for stage in fields(self.stages)
                    if not getattr(self.stages, stage.name)
                ]
            ),
            f"  initializer: {self.initializer}",
            f"  eval dataset split: {self.eval_dataset_split}",
            f"  smoke test: {self.smoke_test}",
            f"  W&B logging: {self.use_wandb}",
            *(
                [f"  W&B group: {wandb_group_url or self.wandb_group}"]
                if self.use_wandb
                else []
            ),
            f"  soft-budget setting: {self.soft_budget_setting}",
            "  dataset realizations: "
            + _join(self.dataset_realization_indices),
            "  datasets:",
            *(
                line
                for dataset in self.datasets
                for line in [
                    f"    {dataset}:",
                    f"      unmasker: {self.unmaskers[dataset]}",
                    "      classifier: "
                    + self.classifier_names[dataset].script_name,
                ]
            ),
            "  methods:",
            *(
                line
                for method in self.methods
                for line in [
                    f"    {method}:",
                    *(
                        f"      {line}"
                        for line in self._describe_method(method)
                    ),
                ]
            ),
            "  method sets:",
            *(
                f"    {name}: {_join(methods)}"
                for name, methods in self.method_sets.items()
            ),
            "  dataset sets:",
            *(
                f"    {name}: {_join(datasets)}"
                for name, datasets in self.dataset_sets.items()
            ),
            "  budgets (hard budgets are eval budgets; soft-budget",
            "  parameters are train -> eval):",
        ]
        for method in self.methods:
            for dataset in self.datasets:
                lines += [
                    f"    {method} on {dataset}:",
                    *(
                        f"      {line}"
                        for line in _describe_budgets(
                            self.budget_params[method][dataset]
                        )
                    ),
                ]
        return "\n".join(lines)

    def _describe_method(self, method: str) -> list[str]:
        classifier = self.method_classifier_script_names.get(method)
        pretrained_model = self.method_to_pretrained_model.get(method)
        return [
            f"train script: {self.method_train_script_names[method]}",
            "classifier: "
            + ("external" if classifier is None else f"built-in {classifier}"),
            f"pretrained model: {pretrained_model or 'none'}",
        ]


def _join(values: Sequence[object]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _describe_budgets(
    combinations: Sequence[BudgetCombination],
) -> list[str]:
    """Render a method's budget combinations on one dataset."""
    hard: list[str] = []
    soft: list[str] = []
    # Every soft-budget run of a method on a dataset shares one train hard
    # budget (see _create_budget_combinations).
    soft_train_hard: set[BudgetParam] = set()
    for train_hard, eval_hard, train_soft, eval_soft in combinations:
        if eval_hard != "null":
            hard.append(
                str(eval_hard)
                if train_hard == eval_hard
                else f"{eval_hard} (trained at {train_hard})"
            )
        else:
            soft.append(f"{train_soft} -> {eval_soft}")
            soft_train_hard.add(train_hard)
    soft_label = "soft-budget parameters"
    if soft_train_hard - {"null"}:
        train_hard_budgets = _join(sorted(soft_train_hard, key=str))
        soft_label += f" (trained at hard budgets {train_hard_budgets})"
    return [f"hard budgets: {_join(hard)}", f"{soft_label}: {_join(soft)}"]


def load_config(config: Mapping[str, Any]) -> WorkflowSettings:
    """Validate the merged Snakemake `config` and resolve its settings."""
    pretrain_mapping = {
        name: _parse_strictly(
            PretrainedModelOptions,
            model_config,
            f"pretrain_mapping[{name!r}]",
        )
        for name, model_config in _required_config(
            config, "pretrain_mapping"
        ).items()
    }
    known_dataset_keys = {path.stem for path in DATASET_KEY_DIR.glob("*.yaml")}
    method_options = {
        method: _method_options(method, options, known_dataset_keys)
        for method, options in _required_config(
            config, "method_options"
        ).items()
    }

    smoke_test = _config_bool(config.get("smoke_test", False), "smoke_test")
    use_wandb = _config_bool(config.get("use_wandb", False), "use_wandb")
    wandb_group = _wandb_group(config.get("wandb_group"), use_wandb=use_wandb)
    output_root = _output_root(
        config.get("output_root"), smoke_test=smoke_test
    )
    methods: list[str] | None = config.get("methods", [])
    if methods is None:
        message = "Expected methods to be provided."
        raise ValueError(message)
    raw_soft_budget_params: dict[str, Any] = _required_config(
        config, "soft_budget_params"
    )
    for method in methods:
        for required, label in [
            (method_options, "method_options"),
            (raw_soft_budget_params, "soft_budget_params"),
        ]:
            if method not in required:
                message = f"Method {method!r} is not in {label}."
                raise ValueError(message)

    method_sets = _method_sets(config, methods)
    datasets: list[str] = _required_config(config, "datasets")

    soft_budget_setting = _config_bool(
        config.get("soft_budget_setting", True), "soft_budget_setting"
    )
    selected_options = {
        method: options
        if soft_budget_setting
        # As if every method ignored every dataset in the soft-budget
        # setting, which also skips pretraining no hard-budget run needs.
        else replace(options, soft_budget_ignored_datasets=list(datasets))
        for method, options in method_options.items()
        if method in methods
    }

    for method, options in selected_options.items():
        if (
            options.pretrained_model_name is not None
            and options.pretrained_model_name not in pretrain_mapping
        ):
            message = (
                f"method_options[{method!r}]: pretrained_model_name "
                f"{options.pretrained_model_name!r} is not in "
                f"pretrain_mapping {sorted(pretrain_mapping)}."
            )
            raise ValueError(message)

    method_to_pretrained_model = {
        method: options.pretrained_model_name
        for method, options in selected_options.items()
        if options.pretrained_model_name is not None
    }
    pretrain_names = [
        name
        for name in pretrain_mapping
        if name in set(method_to_pretrained_model.values())
    ]

    dataset_sets = _dataset_sets(config, datasets)
    eval_hard_budgets = _fill_missing_datasets_with_default(
        _required_config(config, "eval_hard_budgets"), datasets
    )
    soft_budget_params = {
        method: _fill_missing_datasets_with_default(method_params, datasets)
        for method, method_params in raw_soft_budget_params.items()
    }
    classifier_names = _fill_missing_datasets_with_default(
        {
            key: _classifier_script(key, classifier)
            for key, classifier in _required_config(
                config, "classifier_names"
            ).items()
        },
        datasets,
    )

    budget_params = {
        method: {
            dataset: _create_budget_combinations(
                options,
                dataset,
                eval_hard_budgets[dataset],
                soft_budget_params[method][dataset],
            )
            for dataset in datasets
        }
        for method, options in selected_options.items()
    }

    return WorkflowSettings(
        stages=_stages(config.get("stages", {})),
        dataset_realization_indices=config.get(
            "dataset_realization_indices", [0, 1, 2, 3, 4]
        ),
        initializer=config.get("initializer", "cold"),
        # Switch to val while developing, and train if debugging.
        eval_dataset_split=config.get("eval_dataset_split", "test"),
        use_wandb=use_wandb,
        wandb_group=wandb_group,
        smoke_test=smoke_test,
        list_jobs=_config_bool(config.get("list_jobs", False), "list_jobs"),
        soft_budget_setting=soft_budget_setting,
        output_root=output_root,
        pretrain_names=pretrain_names,
        pretrain_script_names={
            name: model_config.pretrain_script_name
            for name, model_config in pretrain_mapping.items()
        },
        pretrain_params={
            name: " ".join(model_config.pretrain_params)
            for name, model_config in pretrain_mapping.items()
        },
        methods=methods,
        method_train_script_names={
            method: options.train_script_name
            for method, options in selected_options.items()
        },
        method_classifier_script_names={
            method: options.classifier.script_name
            for method, options in selected_options.items()
            if options.classifier is not None
        },
        method_classifier_script_params={
            method: " ".join(options.classifier.script_params)
            for method, options in selected_options.items()
            if options.classifier is not None
        },
        method_to_pretrained_model=method_to_pretrained_model,
        method_specific_params={
            method: " ".join(options.method_specific_params)
            for method, options in selected_options.items()
        },
        datasets=datasets,
        unmaskers=_fill_missing_datasets_with_default(
            _required_config(config, "unmaskers"), datasets
        ),
        budget_params=budget_params,
        classifier_names=classifier_names,
        method_sets=method_sets,
        dataset_sets=dataset_sets,
        eval_batch_sizes={
            method: _eval_batch_sizes(method, options, datasets)
            for method, options in selected_options.items()
        },
        datasets_used_per_pretrain_name=_datasets_used_per_pretrain_name(
            pretrain_names,
            method_to_pretrained_model,
            _datasets_used_per_method(selected_options, datasets),
        ),
    )


def _method_options(
    method: str, options: Mapping[str, Any], known_dataset_keys: set[str]
) -> MethodOptions:
    if "eval_to_train_hard_budget_mapping" in options:
        options = {
            **options,
            "eval_to_train_hard_budget_mapping": {
                dataset: {
                    _hard_budget(eval_budget): train_budget
                    for eval_budget, train_budget in mapping.items()
                }
                for dataset, mapping in options[
                    "eval_to_train_hard_budget_mapping"
                ].items()
            },
        }
    parsed = _parse_strictly(
        MethodOptions, options, f"method_options[{method!r}]"
    )
    if (
        isinstance(parsed.eval_batch_size, dict)
        and "default" not in parsed.eval_batch_size
    ):
        message = (
            f"method_options[{method!r}]: eval_batch_size "
            f"{parsed.eval_batch_size} has no default for the other datasets."
        )
        raise ValueError(message)
    # Any known dataset key, not only this run's datasets: `datasets` is a
    # runtime filter, the ignored datasets a property of the method.
    for label, ignored in [
        ("hard_budget_ignored_datasets", parsed.hard_budget_ignored_datasets),
        ("soft_budget_ignored_datasets", parsed.soft_budget_ignored_datasets),
    ]:
        unknown = sorted(set(ignored) - known_dataset_keys)
        if unknown:
            message = (
                f"method_options[{method!r}]: {label} {unknown} are not "
                f"dataset keys (files in {DATASET_KEY_DIR})."
            )
            raise ValueError(message)
    return parsed


def _classifier_script(
    key: str, classifier: str | Mapping[str, Any]
) -> ClassifierScript:
    if isinstance(classifier, str):
        return ClassifierScript(script_name=classifier)
    return _parse_strictly(
        ClassifierScript, classifier, f"classifier_names[{key!r}]"
    )


def _hard_budget(budget: object) -> object:
    """Read a hard budget that JSON turned into a string key back."""
    if not isinstance(budget, str):
        return budget
    try:
        return int(budget)
    except ValueError:
        try:
            return float(budget)
        except ValueError:
            return budget


def _parse_strictly[T](
    data_class: type[T], data: Mapping[str, Any], label: str
) -> T:
    """Parse `data`, rejecting unknown keys; errors start with `label`."""
    try:
        return dacite.from_dict(
            data_class, data, config=dacite.Config(strict=True)
        )
    except dacite.DaciteError as error:
        message = f"{label}: {error}"
        raise ValueError(message) from error


def _output_root(output_root: str | None, *, smoke_test: bool) -> str:
    """
    Resolve the run's output root, apart from the other kind of run's.

    A smoke test whose outputs share a tree with production would leave
    smoke artifacts that a later real run takes as its own and skips their
    jobs, and the reverse would mix production artifacts into a smoke
    release.
    """
    if output_root is None:
        return default_output_root(smoke_test=smoke_test)
    other = default_output_root(smoke_test=not smoke_test)
    # Relative paths are relative to the checkout, Snakemake's working
    # directory.
    resolved, other_resolved = (
        Path(output_root).resolve(),
        Path(other).resolve(),
    )
    if resolved.is_relative_to(
        other_resolved
    ) or other_resolved.is_relative_to(resolved):
        own = default_output_root(smoke_test=smoke_test)
        message = (
            f"smoke_test={str(smoke_test).lower()} cannot write into "
            f"output_root={output_root}, which overlaps {other}; omit "
            f"output_root to use {own}."
        )
        raise ValueError(message)
    return output_root


def _stages(stages: object) -> PipelineStages:
    if not isinstance(stages, Mapping):
        message = f"Expected stages to be a mapping, got {stages!r}."
        raise TypeError(message)
    known = [stage.name for stage in fields(PipelineStages)]
    unknown = [stage for stage in stages if stage not in known]
    if unknown:
        message = (
            f"stages: {unknown} are not pipeline stages a run can turn "
            f"off; expected some of {known}."
        )
        raise ValueError(message)
    return PipelineStages(
        **{
            stage: _config_bool(value, f"stages.{stage}")
            for stage, value in stages.items()
        }
    )


def _config_bool(value: object, key: str) -> bool:
    """Read a bool that a `--config` override may pass as a string."""
    if isinstance(value, bool):
        return value
    # Snakemake can pass scalar and mapping override values as strings.
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    message = f"Expected {key} to be true or false, got {value!r}."
    raise ValueError(message)


def _wandb_group(value: object, *, use_wandb: bool) -> str | None:
    """Read the W&B group, which W&B logging needs."""
    # Snakemake parses an override such as wandb_group=2026 as a number.
    if isinstance(value, int | float) and not isinstance(value, bool):
        value = str(value)
    if value is not None and not isinstance(value, str):
        message = f"Expected wandb_group to be a name, got {value!r}."
        raise ValueError(message)
    if use_wandb and not value:
        message = (
            "use_wandb=true needs wandb_group, the W&B group every W&B run "
            "of this invocation joins; workflow/run.py --wandb names a new "
            "one, and --wandb-group <name> continues an existing one."
        )
        raise ValueError(message)
    return value or None


def _method_sets(
    config: Mapping[str, Any], methods: list[str]
) -> dict[str, list[str]]:
    """
    Resolve the method sets of a run, the reserved "all" set first.

    Keep the methods of the run in each method set and remove method_sets
    without one. The "all" method set holds every method of the run.
    """
    method_sets: dict[str, list[str]] = {}
    for key, method_set in {
        ALL_METHOD_SET: methods,
        **_configured_sets(config, "method_sets", ALL_METHOD_SET),
    }.items():
        filtered_methods = [
            method for method in method_set if method in methods
        ]
        if filtered_methods:
            method_sets[key] = filtered_methods
    return method_sets


def _dataset_sets(
    config: Mapping[str, Any], datasets: list[str]
) -> dict[str, list[str]]:
    """
    Resolve the dataset sets of a run, the reserved "all" set first.

    Keep the datasets of the run in each dataset set and remove dataset_sets
    without one. The "all" dataset set holds every dataset of the run.
    """
    dataset_sets: dict[str, list[str]] = {ALL_DATASET_SET: datasets}
    for key, dataset_set in _configured_sets(
        config, "dataset_sets", ALL_DATASET_SET
    ).items():
        filtered_datasets = [
            dataset for dataset in dataset_set if dataset in datasets
        ]
        if filtered_datasets:
            dataset_sets[key] = filtered_datasets
    return dataset_sets


def _configured_sets(
    config: Mapping[str, Any], key: str, reserved: str
) -> dict[str, list[str]]:
    """Read the named sets under `key`, none of which may be `reserved`."""
    sets: dict[str, list[str]] = config.get(key, {})
    if reserved in sets:
        message = (
            f"{key}: {reserved!r} is reserved for the set the workflow "
            "builds itself and cannot be configured."
        )
        raise ValueError(message)
    return sets


# Snakemake config values are heterogeneous YAML; callers annotate them.
def _required_config(config: Mapping[str, Any], key: str) -> Any:  # noqa: ANN401
    value = config.get(key, None)
    if value is None:
        message = f"Expected {key} to be provided."
        raise ValueError(message)
    return value


def _fill_missing_datasets_with_default(
    config_dict: dict[str, Any], datasets: Sequence[str]
) -> dict[str, Any]:
    return config_dict | {
        dataset: config_dict["default"]
        for dataset in datasets
        if dataset not in config_dict
    }


def _eval_batch_sizes(
    method: str, options: MethodOptions, datasets: Sequence[str]
) -> dict[str, int]:
    batch_size = options.eval_batch_size
    if batch_size is None:
        # Loudly, because a silent fallback to 1 is indistinguishable from a
        # working config while being orders of magnitude slower for any
        # method that batches its acquisition.
        warnings.warn(
            f"method_options['{method}'] has no eval_batch_size; "
            "falling back to 1, which evaluates one instance at a "
            "time. Add an explicit eval_batch_size entry for the "
            "method in the method_options config in use.",
            stacklevel=3,
        )
        return dict.fromkeys(datasets, 1)
    if isinstance(batch_size, dict):
        return batch_size | {
            dataset: batch_size["default"]
            for dataset in datasets
            if dataset not in batch_size
        }
    return dict.fromkeys(datasets, batch_size)


def _create_budget_combinations(
    options: MethodOptions,
    dataset: str,
    eval_hard_budgets: Sequence[int | float],
    soft_budget_params: Sequence[Sequence[NullableParam]],
) -> list[BudgetCombination]:
    """
    Pair train and eval budgets for one method and dataset.

    Hard-budget combinations map each eval hard budget to its train hard
    budget and leave the soft-budget parameters "null"; soft-budget
    combinations leave the hard budgets "null", except the train hard budget
    of a method that trains soft-budget runs under its largest one. A dataset
    the method ignores in a setting gets no combinations for it.
    """
    dataset_mapping = options.eval_to_train_hard_budget_mapping.get(
        dataset, {}
    )
    # An eval hard budget without a mapping is also the train hard budget.
    train_hard_budgets = [
        dataset_mapping.get(eval_budget, eval_budget)
        for eval_budget in eval_hard_budgets
    ]
    result: list[BudgetCombination] = []
    if dataset not in options.hard_budget_ignored_datasets:
        result.extend(
            (train_budget, eval_budget, "null", "null")
            for train_budget, eval_budget in zip(
                train_hard_budgets, eval_hard_budgets, strict=True
            )
        )
    if dataset not in options.soft_budget_ignored_datasets:
        train_hard_budget = (
            max(train_hard_budgets)
            if options.use_max_hard_budget_when_training_soft_budget
            and train_hard_budgets
            else "null"
        )
        result.extend(
            (
                train_hard_budget,
                "null",
                _normalize_nullable_param(train_soft_budget_param),
                _normalize_nullable_param(eval_soft_budget_param),
            )
            for train_soft_budget_param, eval_soft_budget_param in (
                soft_budget_params
            )
        )
    return result


def _normalize_nullable_param(value: NullableParam) -> BudgetParam:
    if value is None:
        return "null"
    return value


def _datasets_used_per_method(
    options: Mapping[str, MethodOptions], datasets: Sequence[str]
) -> dict[str, list[str]]:
    """Leave out the datasets a method ignores in both budget settings."""
    return {
        method: [
            dataset
            for dataset in datasets
            if not (
                dataset in method_options.hard_budget_ignored_datasets
                and dataset in method_options.soft_budget_ignored_datasets
            )
        ]
        for method, method_options in options.items()
    }


def _datasets_used_per_pretrain_name(
    pretrain_names: Sequence[str],
    method_to_pretrained_model: Mapping[str, str],
    datasets_used_per_method: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    """Compute which datasets are needed for each shared pretrained model."""
    datasets_used: dict[str, list[str]] = {
        pretrain_name: [] for pretrain_name in pretrain_names
    }
    for method, pretrain_name in method_to_pretrained_model.items():
        for dataset in datasets_used_per_method[method]:
            if dataset not in datasets_used[pretrain_name]:
                datasets_used[pretrain_name].append(dataset)
    return datasets_used
