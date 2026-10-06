"""Declared per-job execution needs and profile-owned SLURM allocation mapping."""

import re
import shlex
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# Workflow YAML and Snakemake resources are heterogeneous mappings.
# ruff: noqa: ANN401

METHOD_STAGES = {"training", "evaluation", "classifier"}
CPU_ONLY_STAGES = {
    "dataset_generation",
    "transformation",
    "aggregation",
    "visualization",
}
ALLOCATION_RESOURCES = {
    "slurm_partition": "",
    "slurm_account": "",
    "gpu": 0,
    "gres": "",
    "gpu_model": "",
    "slurm_extra": "",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        message = f"Expected {label} mapping, got {value!r}"
        raise TypeError(message)
    return value


def _known_keys(
    value: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = value.keys() - allowed
    if unknown:
        message = f"Unknown {label} keys: {sorted(unknown)}"
        raise ValueError(message)


class ExecutionPolicy:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.execution = _mapping(config.get("execution", {}), "execution")
        _known_keys(
            self.execution,
            {"defaults", "methods", "pretrained_models"},
            "execution",
        )
        self.defaults = _mapping(
            self.execution.get("defaults", {}), "execution.defaults"
        )
        _known_keys(
            self.defaults,
            METHOD_STAGES | {"pretraining"},
            "execution.defaults",
        )
        self.methods = _mapping(
            self.execution.get("methods", {}), "execution.methods"
        )
        self.pretrained_models = _mapping(
            self.execution.get("pretrained_models", {}),
            "execution.pretrained_models",
        )
        site_config = config
        if "execution_site_file" in config:
            if "execution_site" in config:
                message = (
                    "Use only one of execution_site_file and execution_site"
                )
                raise ValueError(message)
            site_config = _mapping(
                yaml.safe_load(
                    Path(config["execution_site_file"]).read_text()
                ),
                "execution_site_file",
            )
        self.site = _mapping(
            site_config.get("execution_site", {}), "execution_site"
        )
        _known_keys(self.site, {"cpu", "gpu"}, "execution_site")
        self.legacy_device = config.get("device", "cpu")
        if "device" in config:
            if "execution" in config:
                message = "Deprecated global device cannot be combined with execution"
                raise ValueError(message)
            print(
                "Global device is deprecated; use execution stage defaults and overrides",
                file=sys.stderr,
            )
        for method in config.get("methods", []):
            overrides = _mapping(
                self.methods.get(method, {}), f"execution.methods.{method}"
            )
            _known_keys(
                overrides, METHOD_STAGES, f"execution.methods.{method}"
            )
            self._validate_method_parameters(config, method)

    def _validate_method_parameters(
        self, config: Mapping[str, Any], method: str
    ) -> None:
        params = (
            config.get("method_options", {})
            .get(method, {})
            .get("method_specific_params", [])
        )
        for param in params:
            for argument in shlex.split(param):
                if argument.split("=", 1)[0].lstrip("+~") == "device":
                    message = f"method_specific_params for {method!r} cannot set device; use execution"
                    raise ValueError(message)

    def device(self, stage: str, identity: str | None) -> str:
        # Processing never inherits GPU intent, including legacy global device.
        if stage in CPU_ONLY_STAGES:
            return "cpu"
        choice = self.defaults.get(stage, self.legacy_device)
        # Shared pretraining is named independently of its downstream methods.
        # A None identity selects the external classifier stage default only.
        if stage == "pretraining":
            choice = self.pretrained_models.get(identity, choice)
        elif identity is not None:
            choice = self.methods.get(identity, {}).get(stage, choice)
        if choice not in ("cpu", "cuda"):
            message = f"Invalid execution choice {choice!r} for {stage}/{identity}; expected cpu or cuda"
            raise ValueError(message)
        return choice

    def _validate_allocation(self, hardware: str) -> None:
        site = _mapping(
            self.site.get(hardware, {}), f"execution_site.{hardware}"
        )
        _known_keys(
            site, set(ALLOCATION_RESOURCES), f"execution_site.{hardware}"
        )
        gpu = site.get("gpu", 0)
        gres = site.get("gres", "")
        gpu_model = site.get("gpu_model", "")
        if hardware == "cpu":
            if gpu or gres or gpu_model or site.get("slurm_extra"):
                message = f"CPU allocation cannot request GPUs or slurm_extra: {dict(site)!r}"
                raise ValueError(message)
            return
        valid_gpu = type(gpu) is int and gpu > 0 and not gres
        valid_gres = (
            isinstance(gres, str)
            and re.fullmatch(r"gpu(:[a-zA-Z0-9_]+)?:[1-9]\d*", gres)
            and not gpu
            and not gpu_model
        )
        if not (valid_gpu or valid_gres) or site.get("slurm_extra"):
            message = f"GPU allocation requires exactly one positive gpu count or GPU gres, not slurm_extra: {dict(site)!r}"
            raise ValueError(message)
        if gpu_model and not re.fullmatch(r"[a-zA-Z0-9_]+", str(gpu_model)):
            message = f"Invalid GPU allocation gpu_model: {gpu_model!r}"
            raise ValueError(message)

    def resource(
        self, name: str, stage: str, identity: str | None
    ) -> int | str:
        hardware = "gpu" if self.device(stage, identity) == "cuda" else "cpu"
        if self.site:
            self._validate_allocation(hardware)
        site = self.site.get(hardware, {})
        default = ALLOCATION_RESOURCES[name]
        if not self.site and name == "gpu" and hardware == "gpu":
            default = 1
        return site.get(name, default)

    def checked_device(
        self, stage: str, identity: str | None, resources: Mapping[str, Any]
    ) -> str:
        for name, default in ALLOCATION_RESOURCES.items():
            if not self.site and name in {"slurm_partition", "slurm_account"}:
                continue
            expected = self.resource(name, stage, identity)
            actual = resources.get(name, default)
            if actual != expected:
                message = f"Conflicting allocation for {stage}/{identity}: {name}={actual!r}, expected {expected!r}; configure execution_site instead of rule overrides"
                raise ValueError(message)
        return self.device(stage, identity)
