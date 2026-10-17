"""
Record Snakemake's effective configuration and consume it for releases.

Snakemake is a development dependency, so its config merging is mirrored
here: config files are merged recursively in order, then `--config` values
are merged over them. A CLI `--configfile` or `--config` replaces the
profile's value for that option, as Snakemake's own option parsing does.

The pipeline records its actual merged configuration before executing
jobs. Release creation uses that record and checks any optional supplied
configuration against it. Nothing per artifact is derived from it
(`docs/adr/0006-release-manifest-indexes-artifact-provenance.md`).
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import dacite
import yaml

WORKFLOW_CONFIG_FILENAME = "workflow_config.json"
BENCHMARK_CONFIG_FILENAME = "benchmark.yaml"
PIPELINE_PROFILE_ENV = "AFABENCH_PIPELINE_PROFILE"


@dataclass(frozen=True, kw_only=True)
class ConfigFileRecord:
    path: str
    sha256: str


@dataclass(frozen=True, kw_only=True)
class WorkflowConfigRecord:
    """The workflow configuration as given, and as Snakemake merges it."""

    profile: str | None
    configfiles: list[ConfigFileRecord]
    overrides: dict[str, Any]
    merged: dict[str, Any]
    benchmark: dict[str, Any]


def resolve_workflow_config(
    *,
    profile: Path | None,
    configfiles: Sequence[Path],
    overrides: Sequence[str],
) -> WorkflowConfigRecord:
    # Reading release manifests needs no checkout; recording defaults does.
    from afabench.core.workflow_settings import (  # noqa: PLC0415
        benchmark_config,
    )

    profile_configfiles: list[Path] = []
    profile_overrides: dict[str, Any] = {}
    if profile is not None:
        profile_configfiles, profile_overrides = _read_profile(profile)

    used_configfiles = list(configfiles) or profile_configfiles
    if not used_configfiles:
        msg = (
            "No workflow config file to record: pass --configfile or a "
            f"--profile whose config.yaml lists configfile (got {profile})."
        )
        raise ValueError(msg)
    used_overrides = (
        _parse_overrides(overrides) if overrides else profile_overrides
    )

    merged: dict[str, Any] = {}
    records: list[ConfigFileRecord] = []
    for path in used_configfiles:
        content = path.read_bytes()
        records.append(
            ConfigFileRecord(
                path=str(path), sha256=hashlib.sha256(content).hexdigest()
            )
        )
        loaded = yaml.safe_load(content) or {}
        if not isinstance(loaded, dict):
            msg = f"Workflow config file {path} is not a mapping."
            raise TypeError(msg)
        _merge(merged, loaded)
    _merge(merged, used_overrides)

    return WorkflowConfigRecord(
        profile=None if profile is None else str(profile),
        configfiles=records,
        overrides=used_overrides,
        merged=merged,
        benchmark=benchmark_config(merged),
    )


def record_workflow_config(
    output_root: Path,
    *,
    profile: str | None,
    configfiles: Sequence[Path],
    overrides: Mapping[str, Any],
    merged: Mapping[str, Any],
) -> None:
    """Record Snakemake's actual configuration before it executes jobs."""
    from afabench.core.workflow_settings import (  # noqa: PLC0415
        benchmark_config,
    )

    record = WorkflowConfigRecord(
        profile=profile,
        configfiles=[
            ConfigFileRecord(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in configfiles
        ],
        overrides=dict(overrides),
        merged=dict(merged),
        benchmark=benchmark_config(merged),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / WORKFLOW_CONFIG_FILENAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(asdict(record), indent=2) + "\n")
    temporary.replace(path)


def release_workflow_config(
    output_root: Path,
    *,
    profile: Path | None = None,
    configfiles: Sequence[Path] = (),
    overrides: Sequence[str] = (),
) -> WorkflowConfigRecord:
    """Use the run's saved configuration, checking any supplied assertion."""
    path = output_root / WORKFLOW_CONFIG_FILENAME
    if not path.is_file():
        msg = (
            f"No recorded workflow configuration at {path}. Execute pipeline "
            "jobs to record their workflow configuration before saving a "
            "release; a dry run or no-op invocation does not create it."
        )
        raise FileNotFoundError(msg)
    recorded = dacite.from_dict(
        WorkflowConfigRecord,
        json.loads(path.read_text()),
        config=dacite.Config(strict=True),
    )
    if profile is not None or configfiles or overrides:
        supplied = resolve_workflow_config(
            profile=profile, configfiles=configfiles, overrides=overrides
        )
        recorded_files = [
            (Path(item.path).resolve(), item.sha256)
            for item in recorded.configfiles
        ]
        supplied_files = [
            (Path(item.path).resolve(), item.sha256)
            for item in supplied.configfiles
        ]
        if (
            supplied.merged != recorded.merged
            or supplied.overrides != recorded.overrides
            or supplied_files != recorded_files
            or (
                profile is not None
                and (
                    recorded.profile is None
                    or profile.resolve() != Path(recorded.profile).resolve()
                )
            )
        ):
            msg = (
                f"Supplied workflow configuration does not match {path}. "
                "Omit --profile, --configfile and --config to use the "
                "recorded run configuration."
            )
            raise ValueError(msg)
    return recorded


def _read_profile(profile: Path) -> tuple[list[Path], dict[str, Any]]:
    profile_file = profile / "config.yaml"
    if not profile_file.is_file():
        msg = f"Snakemake profile has no config.yaml: {profile_file}"
        raise FileNotFoundError(msg)
    content = yaml.safe_load(profile_file.read_text()) or {}
    configfile = content.get("configfile", [])
    configfiles = [configfile] if isinstance(configfile, str) else configfile
    config = content.get("config", {})
    overrides = _parse_overrides(
        [f"{key}={value}" for key, value in config.items()]
        if isinstance(config, dict)
        else config
    )
    return [Path(path) for path in configfiles], overrides


def _parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key:
            msg = f"Workflow config override is not KEY=VALUE: {override!r}"
            raise ValueError(msg)
        parsed[key] = _config_value(value)
    return parsed


def _config_value(value: str) -> object:
    # Match Snakemake's parse_config: numeric scalars, Python booleans,
    # then YAML BaseLoader (which leaves nested scalars as strings).
    if not value:
        return None
    for parse in (int, float):
        try:
            return parse(value)
        except ValueError:
            pass
    if value in {"True", "False"}:
        return value == "True"
    try:
        # BaseLoader constructs only strings and containers, never objects.
        return yaml.load(value, Loader=yaml.BaseLoader)  # noqa: S506
    except yaml.YAMLError:
        return value


def _merge(config: dict[str, Any], update: Mapping[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(config.get(key), dict):
            _merge(config[key], value)
        else:
            config[key] = (
                _copy_mapping(value) if isinstance(value, Mapping) else value
            )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    _merge(copied, value)
    return copied
