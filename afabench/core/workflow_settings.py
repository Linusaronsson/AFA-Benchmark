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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dacite

from afabench.core.output_layout import (
    EvaluationRun,
    OutputLayout,
    PathValue,
    TrainingRun,
    default_output_root,
    pretrain_seed_folder,
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
    Path(__file__).resolve().parents[2] / "conf/components/dataset_key"
)


@dataclass(frozen=True, kw_only=True)
class ClassifierScript:
    """A classifier training script and its extra arguments."""

    script_name: str
    script_params: list[str] = field(default_factory=list)


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
class WorkflowSettings:
    dataset_realization_indices: list[int]
    initializer: str
    eval_dataset_split: str
    use_wandb: bool
    smoke_test: bool
    # Where the run writes its artifacts and job records
    output_root: str
    # Only the pretrained models a selected method uses.
    pretrain_names: list[str]
    pretrain_script_names: dict[str, str]
    pretrain_params: dict[str, str]
    methods: list[str]
    # Methods whose plotting-ready tables are restored from a benchmark
    # release: they join method sets and aggregation, but the workflow never
    # produces anything for them.
    reference_methods: list[str]
    compared_methods_with_pretraining_stage: list[str]
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
                    if method in self.compared_methods_with_pretraining_stage
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

    def summary(self) -> str:
        """
        Describe what this run covers, for a reader checking a dry run.

        Each method is listed with its classifier and pretrained model, and
        each method and dataset with the hard budgets it is evaluated at and
        the soft-budget parameters it is trained and evaluated with.
        """
        lines = [
            "Resolved workflow configuration:",
            f"  output root: {self.output_root}",
            f"  initializer: {self.initializer}",
            f"  eval dataset split: {self.eval_dataset_split}",
            f"  smoke test: {self.smoke_test}",
            f"  W&B logging: {self.use_wandb}",
            "  dataset realizations: "
            + _join(self.dataset_realization_indices),
            "  datasets:",
            *(
                f"    {dataset}: unmasker {self.unmaskers[dataset]}, "
                f"classifier {self.classifier_names[dataset].script_name}"
                for dataset in self.datasets
            ),
            "  methods:",
            *(
                f"    {method}: {self._method_description(method)}"
                for method in self.methods
            ),
            f"  reference methods: {_join(self.reference_methods)}",
            "  method sets:",
            *(
                f"    {name}: {_join(methods)}"
                for name, methods in self.method_sets.items()
            ),
            "  budgets (hard budgets are eval budgets; soft-budget",
            "  parameters are train -> eval):",
        ]
        for method in [*self.methods, *self.reference_methods]:
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

    def _method_description(self, method: str) -> str:
        classifier = self.method_classifier_script_names.get(method)
        pretrained_model = self.method_to_pretrained_model.get(method)
        return ", ".join(
            [
                f"train script {self.method_train_script_names[method]}",
                "external classifier"
                if classifier is None
                else f"built-in classifier {classifier}",
                "no pretrained model"
                if pretrained_model is None
                else f"pretrained model {pretrained_model}",
            ]
        )


def _join(values: Sequence[object]) -> str:
    return ", ".join(str(value) for value in values) or "none"


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
        soft_label += f" (trained at hard budget {train_hard_budgets})"
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

    smoke_test: bool = config.get("smoke_test", False)
    output_root = _output_root(
        config.get("output_root"), smoke_test=smoke_test
    )
    methods: list[str] | None = config.get("methods", [])
    if methods is None:
        message = "Expected methods to be provided."
        raise ValueError(message)
    reference_methods = config.get("reference_methods", [])
    _check_reference_methods(reference_methods, methods, method_options)
    compared_methods = [*methods, *reference_methods]
    raw_soft_budget_params: dict[str, Any] = _required_config(
        config, "soft_budget_params"
    )
    for method in compared_methods:
        for required, label in [
            (method_options, "method_options"),
            (raw_soft_budget_params, "soft_budget_params"),
        ]:
            if method not in required:
                message = f"Method {method!r} is not in {label}."
                raise ValueError(message)

    # Filter out methods that are neither enabled by the "methods" option nor
    # reference methods, and remove method_sets without an enabled method:
    # a set of reference methods only is already plotted in its release.
    method_sets: dict[str, list[str]] = {}
    for key, method_set in config.get("method_sets", {}).items():
        filtered_methods = [
            method for method in method_set if method in compared_methods
        ]
        if any(method in methods for method in filtered_methods):
            method_sets[key] = filtered_methods

    selected_options = {
        method: options
        for method, options in method_options.items()
        if method in methods
    }
    compared_options = {
        method: options
        for method, options in method_options.items()
        if method in compared_methods
    }

    for method, options in compared_options.items():
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

    datasets: list[str] = _required_config(config, "datasets")
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
        for method, options in compared_options.items()
    }

    return WorkflowSettings(
        dataset_realization_indices=config.get(
            "dataset_realization_indices", [0, 1, 2, 3, 4]
        ),
        initializer=config.get("initializer", "cold"),
        # Switch to val while developing, and train if debugging.
        eval_dataset_split=config.get("eval_dataset_split", "test"),
        use_wandb=config.get("use_wandb", True),
        smoke_test=smoke_test,
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
        reference_methods=reference_methods,
        # Reference tables live under the same pretraining folder as they
        # would if the method were produced here.
        compared_methods_with_pretraining_stage=[
            method
            for method, options in compared_options.items()
            if options.pretrained_model_name is not None
        ],
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


def _check_reference_methods(
    reference_methods: object,
    methods: Sequence[str],
    method_options: Mapping[str, MethodOptions],
) -> None:
    if not isinstance(reference_methods, list):
        message = (
            "Expected reference_methods to be a list, got "
            f"{reference_methods!r}."
        )
        raise TypeError(message)
    unknown = [
        method for method in reference_methods if method not in method_options
    ]
    if unknown:
        message = f"Reference methods {unknown} are not in method_options."
        raise ValueError(message)
    # Producing a reference method too would compare its restored and its
    # local tables as one method, duplicating its rows.
    produced = [method for method in reference_methods if method in methods]
    if produced:
        message = (
            f"Methods {produced} are both in methods and in "
            "reference_methods; name each method in one of them."
        )
        raise ValueError(message)


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
