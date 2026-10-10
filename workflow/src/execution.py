"""Declared per-job execution needs and profile-owned SLURM allocation mapping."""

import re
import shlex
import warnings
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Literal

import yaml

from afabench.core.job_record import Device

type Hardware = Literal["cpu", "gpu"]
# Pipeline stages: the kinds of job whose hardware this policy resolves.
type Stage = Literal[
    "classifier_training",
    "pretraining",
    "training",
    "evaluation",
    "dataset_generation",
    "transformation",
    "aggregation",
    "visualization",
]
type ResourceFunction = Callable[[object], int | str]

METHOD_STAGES = {"training", "evaluation", "classifier_training"}
COMPUTATIONAL_STAGES = METHOD_STAGES | {"pretraining"}
PROCESSING_STAGES = {
    "dataset_generation",
    "transformation",
    "aggregation",
    "visualization",
}
REPO_ROOT = Path(__file__).parents[2]
DEFAULT_EXECUTION_FILE = (
    Path(__file__).parents[1] / "profiles" / "execution" / "default.yaml"
)
ALLOCATION_RESOURCES: dict[str, int | str] = {
    "slurm_partition": "",
    "slurm_account": "",
    "gpu": 0,
    "gres": "",
    "gpu_model": "",
    "slurm_extra": "",
}
# Allocation keys that are no SLURM resource: the image the job runs in
# (workflow/src/images.py), relative to the repository root
ALLOCATION_SETTINGS = {"image"}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        message = f"Expected {label} mapping, got {value!r}"
        raise TypeError(message)
    return value


def _known_keys(
    value: Mapping[str, object], allowed: set[str], label: str
) -> None:
    unknown = value.keys() - allowed
    if unknown:
        message = f"Unknown {label} keys: {sorted(unknown)}"
        raise ValueError(message)


def checked_script_params(params: str, label: str) -> str:
    """Return script arguments, rejecting any that bypass the resolved device."""
    for argument in shlex.split(params):
        if argument.split("=", 1)[0].lstrip("+~") == "device":
            message = f"{label} cannot set device; use execution"
            raise ValueError(message)
    return params


class ExecutionPolicy:
    def __init__(
        self,
        config: Mapping[str, object],
        *,
        method_classifiers: Collection[str],
        default_resources: Mapping[str, object],
        submits_to_cluster: Callable[[], bool],
    ) -> None:
        if "slurm_extra" in default_resources:
            # Every rule sets slurm_extra from its allocation, replacing it.
            message = "default-resources slurm_extra would be replaced by every job's allocation; move scheduler flags to execution_site.<cpu|gpu>.slurm_extra"
            raise ValueError(message)
        # Called per job: the executor is unknown while the Snakefile parses.
        self.submits_to_cluster = submits_to_cluster
        execution = config.get("execution")
        if execution is None and "device" not in config:
            execution_file = Path(
                str(config.get("execution_file", DEFAULT_EXECUTION_FILE))
            )
            execution = _mapping(
                yaml.safe_load(execution_file.read_text()), "execution_file"
            ).get("execution")
        self.execution = _mapping(execution or {}, "execution")
        _known_keys(
            self.execution,
            {"defaults", "methods", "pretrained_models"},
            "execution",
        )
        self.defaults = _mapping(
            self.execution.get("defaults", {}), "execution.defaults"
        )
        _known_keys(self.defaults, COMPUTATIONAL_STAGES, "execution.defaults")
        self.methods = _mapping(
            self.execution.get("methods", {}), "execution.methods"
        )
        self.pretrained_models = _mapping(
            self.execution.get("pretrained_models", {}),
            "execution.pretrained_models",
        )
        _known_keys(
            self.pretrained_models,
            set(
                _mapping(
                    config.get("pretrain_mapping", {}), "pretrain_mapping"
                )
            ),
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
                    Path(str(config["execution_site_file"])).read_text()
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
            # Attributed to this module: Snakemake 9.12.0 drops warnings
            # attributed to Snakefile frames, as stacklevel=2 would be.
            warnings.warn(
                "Global device is deprecated; use per-stage execution "
                "defaults and overrides",
                stacklevel=1,
            )
        method_options = _mapping(
            config.get("method_options", {}), "method_options"
        )
        for method in config.get("methods", []):
            overrides = self._method_overrides(method)
            _known_keys(
                overrides, METHOD_STAGES, f"execution.methods.{method}"
            )
            if (
                "classifier_training" in overrides
                and method not in method_classifiers
            ):
                message = f"execution.methods.{method}.classifier_training is set, but {method!r} has no method-specific classifier; external classifiers use execution.defaults.classifier_training"
                raise ValueError(message)
            options = _mapping(
                method_options.get(method, {}), f"method_options.{method}"
            )
            for param in options.get("method_specific_params", []):
                checked_script_params(
                    param, f"method_specific_params for {method!r}"
                )

    def _method_overrides(self, method: str) -> Mapping[str, object]:
        return _mapping(
            self.methods.get(method, {}), f"execution.methods.{method}"
        )

    def device(self, stage: Stage, identity: str | None) -> Device:
        # Processing never inherits GPU intent, including legacy global device.
        if stage in PROCESSING_STAGES:
            return "cpu"
        choice = self.defaults.get(stage, self.legacy_device)
        # Shared pretraining is named independently of its downstream methods.
        # A None identity selects the external classifier default only.
        if stage == "pretraining":
            choice = self.pretrained_models.get(str(identity), choice)
        elif identity is not None:
            choice = self._method_overrides(identity).get(stage, choice)
        if choice in ("cpu", "cuda"):
            return choice
        message = f"Invalid execution choice {choice!r} for {stage}/{identity}; expected cpu or cuda"
        raise ValueError(message)

    def _validate_allocation(
        self, hardware: Hardware, stage: Stage, identity: str | None
    ) -> None:
        if hardware not in self.site:
            message = f"execution_site has no {hardware} allocation for {stage}/{identity}"
            raise ValueError(message)
        site = _mapping(
            self.site.get(hardware, {}), f"execution_site.{hardware}"
        )
        _known_keys(
            site,
            set(ALLOCATION_RESOURCES) | ALLOCATION_SETTINGS,
            f"execution_site.{hardware}",
        )
        if "image" in site:
            image = site["image"]
            if not isinstance(image, str) or not image:
                message = f"execution_site.{hardware}.image must be a path: {image!r}"
                raise ValueError(message)
            if not (REPO_ROOT / image).is_file():
                message = f"execution_site.{hardware}.image {image!r} does not exist under {REPO_ROOT}; build it with containers/build.sbatch"
                raise ValueError(message)
        slurm_extra = site.get("slurm_extra", "")
        if not isinstance(slurm_extra, str) or any(
            re.match(r"--gres|--gpus|-G", argument)
            for argument in shlex.split(slurm_extra)
        ):
            message = f"{hardware.upper()} allocation slurm_extra must not request GPUs; use gpu or gres: {dict(site)!r}"
            raise ValueError(message)
        gpu = site.get("gpu", 0)
        gres = site.get("gres", "")
        gpu_model = site.get("gpu_model", "")
        if hardware == "cpu":
            if gpu or gres or gpu_model:
                message = f"CPU allocation cannot request GPUs: {dict(site)!r}"
                raise ValueError(message)
            return
        valid_gpu = type(gpu) is int and gpu > 0 and not gres
        valid_gres = (
            isinstance(gres, str)
            and re.fullmatch(r"gpu(:[a-zA-Z0-9_]+)?:[1-9]\d*", gres)
            and not gpu
            and not gpu_model
        )
        if not (valid_gpu or valid_gres):
            message = f"GPU allocation requires exactly one positive gpu count or GPU gres: {dict(site)!r}"
            raise ValueError(message)
        if gpu_model and not re.fullmatch(r"[a-zA-Z0-9_]+", str(gpu_model)):
            message = f"Invalid GPU allocation gpu_model: {gpu_model!r}"
            raise ValueError(message)

    def hardware(self, stage: Stage, identity: str | None) -> Hardware:
        """Return which allocation the job resolves to."""
        return "gpu" if self.device(stage, identity) == "cuda" else "cpu"

    def image(self, stage: Stage, identity: str | None) -> Path | None:
        """Return the image the job's allocation names, if any."""
        if not self.site:
            return None
        hardware = self.hardware(stage, identity)
        self._validate_allocation(hardware, stage, identity)
        image = _mapping(self.site[hardware], "execution_site").get("image")
        return None if image is None else REPO_ROOT / str(image)

    def resource(
        self, name: str, stage: Stage, identity: str | None
    ) -> int | str:
        hardware = self.hardware(stage, identity)
        if self.site:
            self._validate_allocation(hardware, stage, identity)
        elif self.submits_to_cluster():
            message = f"Cluster submission of {stage}/{identity} needs an execution_site allocation map; set execution_site_file in the site profile, and repeat it whenever passing --config, which replaces the profile's config"
            raise ValueError(message)
        site = _mapping(self.site.get(hardware, {}), "execution_site")
        default = ALLOCATION_RESOURCES[name]
        if not self.site and name == "gpu" and hardware == "gpu":
            # Local runs: lets --resources gpu=<n> bound concurrent cuda jobs.
            default = 1
        return site.get(name, default)

    def allocation_resources(
        self,
        stage: Stage,
        identity: Callable[[object], str | None],
    ) -> dict[str, ResourceFunction]:
        """Return a rule's allocation resources, resolved per job."""
        names = list(ALLOCATION_RESOURCES)
        if not self.site:
            # Without a site map, the profile's partition and account apply.
            names.remove("slurm_partition")
            names.remove("slurm_account")
        return {
            name: self._resource_function(name, stage, identity)
            for name in names
        }

    def _resource_function(
        self,
        name: str,
        stage: Stage,
        identity: Callable[[object], str | None],
    ) -> ResourceFunction:
        def resolve(wildcards: object) -> int | str:
            return self.resource(name, stage, identity(wildcards))

        return resolve

    def checked_device(
        self,
        stage: Stage,
        identity: str | None,
        resources: Mapping[str, object],
    ) -> Device:
        """Return the job's device after checking its final allocation."""
        for name, default in ALLOCATION_RESOURCES.items():
            if not self.site and name in {"slurm_partition", "slurm_account"}:
                continue
            expected = self.resource(name, stage, identity)
            actual = resources.get(name, default)
            if actual != expected:
                message = f"Conflicting allocation for {stage}/{identity}: {name}={actual!r}, expected {expected!r}; configure execution_site instead of rule overrides"
                raise ValueError(message)
        return self.device(stage, identity)
