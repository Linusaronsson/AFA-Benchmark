"""Configuration helpers for the missing-training-data workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MissingMethodSpec:
    """A base benchmark method or a missingness-specific training variant."""

    name: str
    base_method: str
    train_script_name: str
    pretrained_model_name: str | None
    train_params: tuple[str, ...]
    allowed_strategies: tuple[str, ...] | None
    extra_strategies: tuple[str, ...]
    include_complete_data: bool


def build_method_specs(
    methods: list[str],
    method_options: dict[str, dict[str, Any]],
    method_overrides: dict[str, dict[str, Any]],
    method_variants: dict[str, dict[str, Any]],
) -> dict[str, MissingMethodSpec]:
    """Build selected method specifications from the shared method catalog."""
    specs: dict[str, MissingMethodSpec] = {}
    for method in methods:
        if method not in method_options:
            msg = f"Unknown selected method: {method}"
            raise ValueError(msg)
        options = method_options[method]
        override = method_overrides.get(method, {})
        specs[method] = _method_spec(method, method, options, override)

    for name, variant in method_variants.items():
        base_method = str(variant["base_method"])
        if base_method not in methods:
            continue
        options = method_options[base_method]
        specs[name] = _method_spec(name, base_method, options, variant)
    return specs


def _method_spec(
    name: str,
    base_method: str,
    options: dict[str, Any],
    override: dict[str, Any],
) -> MissingMethodSpec:
    allowed = override.get("allowed_strategies")
    return MissingMethodSpec(
        name=name,
        base_method=base_method,
        train_script_name=str(options["train_script_name"]),
        pretrained_model_name=options.get("pretrained_model_name"),
        train_params=tuple(
            str(value)
            for value in [
                *options.get("method_specific_params", []),
                *override.get("train_params", []),
            ]
        ),
        allowed_strategies=(
            tuple(str(value) for value in allowed)
            if allowed is not None
            else None
        ),
        extra_strategies=tuple(
            str(value) for value in override.get("extra_strategies", [])
        ),
        include_complete_data=bool(
            override.get("include_complete_data", name == base_method)
        ),
    )


def largest_hard_budget(
    combinations: list[tuple[Any, Any, Any, Any]],
) -> tuple[str, str] | None:
    """Return the train/evaluation pair with the largest evaluation budget."""
    hard = [
        (train, evaluation)
        for train, evaluation, train_soft, eval_soft in combinations
        if evaluation != "null"
        and train_soft == "null"
        and eval_soft == "null"
    ]
    if not hard:
        return None
    train, evaluation = max(hard, key=lambda values: float(values[1]))
    return str(train), str(evaluation)
