"""Prepare datasets before content-checked reuse of a downloaded benchmark."""

import json
import shlex
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from afabench.core.workflow_settings import load_config
from afabench.release.manifest import RELEASE_MANIFEST_FILENAME
from afabench.release.workflow_config import resolve_workflow_config


def _with_config(arguments: list[str], **values: object) -> list[str]:
    split = (
        arguments.index("--config")
        if "--config" in arguments
        else len(arguments)
    )
    overrides = [
        word
        for word in arguments[split + 1 :]
        if word.split("=", 1)[0] not in values
    ]
    overrides.extend(
        f"{key}={json.dumps(value)}" for key, value in values.items()
    )
    return [*arguments[:split], "--config", *overrides]


def config_run_commands(
    arguments: Sequence[str], *, preview: bool = False
) -> list[list[str]]:
    """Plan one config invocation, with a preparation phase only for a cache."""
    command = list(arguments)
    if "--configfile" not in command:
        return [command]
    config_index = command.index("--configfile")
    # Config-file invocations put the target first so --configfile, which
    # accepts several files, cannot consume it as another filename.
    target_index = 0
    record = resolve_workflow_config(
        profile=None,
        configfiles=[Path(command[config_index + 1])],
        overrides=command[command.index("--config") + 1 :]
        if "--config" in command
        else [],
    )
    settings = load_config(record.merged)
    if not (Path(settings.output_root) / RELEASE_MANIFEST_FILENAME).is_file():
        return [command]
    # An explicit dataset-only target never consumes released artifacts.
    if command[target_index] == "all_generate_datasets":
        return [_with_config(command, reuse_release=False)]
    if preview:
        if settings.stages.dataset_generation:
            command.insert(1, "all_generate_datasets")
        if not any(flag in command for flag in ("-n", "--dry-run")):
            index = (
                command.index("--config")
                if "--config" in command
                else len(command)
            )
            command.insert(index, "--dry-run")
        return [
            _with_config(command, reuse_release=True, release_preview=True)
        ]
    commands = []
    if settings.stages.dataset_generation:
        prepare = _with_config(command, reuse_release=False)
        prepare[target_index] = "all_generate_datasets"
        commands.append(prepare)
    commands.append(
        _with_config(
            command,
            reuse_release=True,
            release_preview=False,
            stages={**asdict(settings.stages), "dataset_generation": False},
        )
    )
    return commands


def run_config(command: list[str]) -> int:
    """Execute the phases with the interpreter selected by the host launcher."""
    preview = any(flag in command for flag in ("-n", "--dry-run"))
    for phase in config_run_commands(command, preview=preview):
        invocation = [sys.executable, "-m", "snakemake", *phase]
        print(shlex.join(invocation), file=sys.stderr, flush=True)
        result = subprocess.run(invocation, check=False)
        if result.returncode:
            return result.returncode
    return 0
