"""Site-owned requested walltime, independent of measured job durations."""

from collections.abc import Callable, Collection, Mapping
from decimal import ROUND_CEILING, Decimal

# Only rules that run commands request walltime, not aggregate targets.
RULE_SELECTORS = {
    "dataset_generation": None,
    "train_classifier": None,
    "train_classifier_for_method": "methods",
    "pretrain_model": "pretrained_models",
    "train_method": "methods",
    "eval_method": "methods",
    "transform_eval_data": "methods",
    "merge_eval_perf": None,
    "split_by_classifier_type": None,
    "collect_job_records": None,
    "plot_eval_perf": None,
    "plot_eval_actions": None,
    "plot_time": None,
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        message = f"Expected {label} mapping, got {value!r}"
        raise TypeError(message)
    return value


def _known_keys(
    values: Mapping[str, object], allowed: Collection[str], label: str
) -> None:
    unknown = values.keys() - set(allowed)
    if unknown:
        message = f"Unknown {label} keys: {sorted(map(str, unknown))}"
        raise ValueError(message)


def _minutes(hours: object, label: str) -> int:
    if type(hours) not in (int, float):
        message = (
            f"{label} must be positive finite numeric hours, got {hours!r}"
        )
        raise ValueError(message)
    value = Decimal(str(hours))
    if not value.is_finite() or value <= 0:
        message = (
            f"{label} must be positive finite numeric hours, got {hours!r}"
        )
        raise ValueError(message)
    # Decimal avoids adding a minute to exact decimal minute boundaries.
    return int((value * 60).to_integral_value(rounding=ROUND_CEILING))


class WalltimePolicy:
    def __init__(
        self,
        config: object,
        *,
        methods: Collection[str],
        pretrained_models: Collection[str],
        datasets: Collection[str],
        default_runtime: object = None,
    ) -> None:
        table = _mapping(config, "walltime")
        _known_keys(table, {"default_hours", "rules"}, "walltime")
        self.default_runtime = default_runtime
        self.site_default = self._default(table, "walltime")
        self.rule_defaults: dict[str, int | None] = {}
        self.limits: dict[tuple[str, str, str | None], int] = {}
        rules = _mapping(table.get("rules", {}), "walltime.rules")
        _known_keys(rules, RULE_SELECTORS, "walltime.rules")
        for rule, value in rules.items():
            label = f"walltime.rules.{rule}"
            options = _mapping(value, label)
            selector = RULE_SELECTORS[rule]
            allowed = {"default_hours"}
            if selector is not None:
                allowed.add(selector)
            _known_keys(options, allowed, label)
            self.rule_defaults[rule] = self._default(options, label)
            if selector is None:
                continue
            identities = _mapping(options.get(selector, {}), label)
            _known_keys(
                identities,
                pretrained_models
                if selector == "pretrained_models"
                else methods,
                f"{label}.{selector}",
            )
            for identity, overrides in identities.items():
                identity_label = f"{label}.{selector}.{identity}"
                values = _mapping(overrides, identity_label)
                _known_keys(
                    values, {"default_hours", "datasets"}, identity_label
                )
                default = self._default(values, identity_label)
                if default is not None:
                    self.limits[rule, identity, None] = default
                dataset_limits = _mapping(
                    values.get("datasets", {}),
                    f"{identity_label}.datasets",
                )
                _known_keys(
                    dataset_limits, datasets, f"{identity_label}.datasets"
                )
                for dataset, hours in dataset_limits.items():
                    self.limits[rule, identity, dataset] = _minutes(
                        hours, f"{identity_label}.datasets.{dataset}"
                    )

    @staticmethod
    def _default(values: Mapping[str, object], label: str) -> int | None:
        if "default_hours" not in values:
            return None
        return _minutes(values["default_hours"], f"{label}.default_hours")

    def minutes(
        self, rule: str, identity: str | None, dataset: str | None
    ) -> int | None:
        """Return a table match, or None to retain Snakemake's default."""
        if identity is not None:
            for key in [(rule, identity, dataset), (rule, identity, None)]:
                if key in self.limits:
                    return self.limits[key]
        default = self.rule_defaults.get(rule)
        return self.site_default if default is None else default

    def resources(self, rule: str) -> dict[str, Callable[..., object]]:
        """Set runtime only where the table can affect this rule."""
        if self.site_default is None and rule not in self.rule_defaults:
            return {}

        def runtime(wildcards: object) -> object:
            selector = RULE_SELECTORS[rule]
            identity = (
                getattr(
                    wildcards,
                    "pretrained_model_name"
                    if selector == "pretrained_models"
                    else "method",
                    None,
                )
                if selector is not None
                else None
            )
            minutes = self.minutes(
                rule, identity, getattr(wildcards, "dataset", None)
            )
            return self.default_runtime if minutes is None else minutes

        # An input argument makes Snakemake defer dry-run evaluation. Use it
        # only when a sparse table can fall back to an input-based expression.
        if (
            callable(self.default_runtime)
            and self.site_default is None
            and self.rule_defaults.get(rule) is None
        ):

            def runtime_with_input(
                wildcards: object,
                input: object,  # noqa: A002 - Snakemake resource API
                attempt: int,
                threads: int,
                rulename: str,
            ) -> object:
                value = runtime(wildcards)
                if callable(value):
                    return value(
                        wildcards=wildcards,
                        input=input,
                        attempt=attempt,
                        threads=threads,
                        rulename=rulename,
                    )
                return value

            return {"runtime": runtime_with_input}
        return {"runtime": runtime}
