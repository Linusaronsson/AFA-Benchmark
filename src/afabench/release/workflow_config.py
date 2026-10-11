"""
Resolve the Snakemake workflow configuration a pipeline run used.

Snakemake is a development dependency, so its config merging is mirrored
here: config files are merged recursively in order, then `--config` values
are merged over them. A CLI `--configfile` or `--config` replaces the
profile's value for that option, as Snakemake's own option parsing does.

A release manifest records the result as the configuration whose targets
lay out the release; nothing per artifact is derived from it
(`docs/adr/0006-release-manifest-indexes-artifact-provenance.md`).
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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


def resolve_workflow_config(
    *,
    profile: Path | None,
    configfiles: Sequence[Path],
    overrides: Sequence[str],
) -> WorkflowConfigRecord:
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
    )


def _read_profile(profile: Path) -> tuple[list[Path], dict[str, Any]]:
    profile_file = profile / "config.yaml"
    if not profile_file.is_file():
        msg = f"Snakemake profile has no config.yaml: {profile_file}"
        raise FileNotFoundError(msg)
    content = yaml.safe_load(profile_file.read_text()) or {}
    configfile = content.get("configfile", [])
    configfiles = [configfile] if isinstance(configfile, str) else configfile
    config = content.get("config", {})
    overrides = (
        dict(config) if isinstance(config, dict) else _parse_overrides(config)
    )
    return [Path(path) for path in configfiles], overrides


def _parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key:
            msg = f"Workflow config override is not KEY=VALUE: {override!r}"
            raise ValueError(msg)
        parsed[key] = yaml.safe_load(value)
    return parsed


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
