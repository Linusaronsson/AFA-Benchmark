#!/usr/bin/env python3
"""
Submit workflow/venvs/build.sbatch once per venv a site profile names.

    python3 workflow/venvs/submit_builds.py arrhenius [--dry-run]

Each allocation of the site file (`execution_site.<cpu|gpu>`) that names a
venv is built on that allocation, with its account, partition and GPU
request, so they are written down once, in the site profile: a cluster
whose nodes differ in architecture needs a venv built on each (ADR 0015).
The argument is a site profile under workflow/profiles/site/, whose
site.yaml is read, or the path of a site file.

This runs on the login node before any venv exists, and the pipeline
launcher imports it to check the venvs, so it runs on the host's python3
(3.9 on Arrhenius) with the standard library only: argparse instead of
typer, and a reader for the subset of YAML that site files use instead of
PyYAML.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITES = REPO_ROOT / "workflow" / "profiles" / "site"
# build.sbatch builds only .venv-<arch>, <arch> being the node's `uname -m`,
# the venv workflow/venvs/bin/python runs.
VENV_NAME = re.compile(r"\.venv-[A-Za-z0-9_]+")
BUILD_SCRIPT = "workflow/venvs/build.sbatch"


def read_site_yaml(text: str, label: str) -> dict[str, object]:
    """
    Read nested mappings of scalars, the only YAML site files use.

    Anything else, such as a list or a multi-line value, fails rather than
    being misread.
    """
    root: dict[str, object] = {}
    # (indentation of its entries, mapping) of every mapping still open
    stack: list[tuple[int, dict[str, object]]] = [(0, root)]
    # The mapping a `key:` line opened, with that key's indentation
    pending: tuple[int, dict[str, object]] | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        entry = _entry(raw, f"{label}:{number}")
        if entry is None:
            continue
        indent, key, value = entry
        if pending is not None:
            if indent <= pending[0]:
                message = f"{label}:{number}: expected the entries of the mapping above"
                raise ValueError(message)
            stack.append((indent, pending[1]))
            pending = None
        while indent < stack[-1][0]:
            stack.pop()
        if indent != stack[-1][0]:
            message = f"{label}:{number}: indentation matches no open mapping"
            raise ValueError(message)
        mapping = stack[-1][1]
        if key in mapping:
            message = f"{label}:{number}: duplicate key {key!r}"
            raise ValueError(message)
        if value is None:
            child: dict[str, object] = {}
            mapping[key] = child
            pending = (indent, child)
        else:
            mapping[key] = _scalar(value.strip(), f"{label}:{number}")
    if pending is not None:
        message = f"{label}: the last key has no value or entries"
        raise ValueError(message)
    return root


def _entry(raw: str, label: str) -> tuple[int, str, str | None] | None:
    """Return a line's indentation, key and value, or None if it is blank."""
    line = re.sub(r"(^|\s)#.*", "", raw).rstrip()
    if not line.strip():
        return None
    indent = len(line) - len(line.lstrip(" "))
    match = re.fullmatch(r"([A-Za-z0-9_]+):(?: (.*))?", line.strip())
    if match is None or "\t" in line[:indent]:
        message = f"{label}: unsupported YAML {raw!r}; site files hold nested mappings of scalars only"
        raise ValueError(message)
    return indent, match.group(1), match.group(2)


def _scalar(value: str, label: str) -> object:
    if re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    quoted = re.fullmatch(r"'([^']*)'|\"([^\"\\]*)\"", value)
    if quoted is not None:
        return (
            quoted.group(1) if quoted.group(1) is not None else quoted.group(2)
        )
    if re.fullmatch(r"[A-Za-z0-9_./:@+-]+", value) is None or value in {
        "true",
        "false",
        "null",
        "~",
    }:
        message = f"{label}: unsupported YAML value {value!r}; quote strings with other characters"
        raise ValueError(message)
    return value


def site_file(site: str) -> Path:
    path = Path(site)
    if path.is_file():
        return path
    path = SITES / site / "site.yaml"
    if not path.is_file():
        message = f"No site file {site!r}: expected a file or a site profile with {path}"
        raise FileNotFoundError(message)
    return path


def lock_record(venv: Path) -> Path:
    """Return where build.sbatch records the uv.lock `venv` was synced from."""
    return venv / "afabench.uv.lock"


def site_venvs(site: dict[str, object], label: str) -> dict[str, str]:
    """Return the venv each allocation of `site` names, by its hardware."""
    allocations = site.get("execution_site")
    if not isinstance(allocations, dict):
        message = f"{label} has no execution_site mapping"
        raise TypeError(message)
    venvs: dict[str, str] = {}
    for hardware in ["cpu", "gpu"]:
        allocation = allocations.get(hardware)
        if isinstance(allocation, dict) and "venv" in allocation:
            venvs[hardware] = str(allocation["venv"])
    return venvs


def venv_problem(checkout: Path, venv: str) -> str | None:
    """
    Return why the venv `venv` of `checkout` cannot run jobs, or None.

    The record is written only once the venv is synced, and is read rather
    than the venv's packages, because a login node cannot run a venv of
    another architecture.
    """
    record = lock_record(checkout / venv)
    if not record.is_file():
        return f"No venv built at {checkout / venv}"
    if record.read_bytes() != (checkout / "uv.lock").read_bytes():
        return f"The venv {checkout / venv} was built from another uv.lock than {checkout / 'uv.lock'}"
    return None


def build_commands(site: dict[str, object], label: str) -> list[list[str]]:
    """Return one sbatch command per distinct venv the site names."""
    allocations = site.get("execution_site")
    if not isinstance(allocations, dict):
        message = f"{label} has no execution_site mapping"
        raise TypeError(message)
    commands: list[list[str]] = []
    built: set[str] = set()
    for hardware in ["cpu", "gpu"]:
        allocation = allocations.get(hardware)
        if not isinstance(allocation, dict) or "venv" not in allocation:
            continue
        venv = str(allocation["venv"])
        if not VENV_NAME.fullmatch(venv):
            message = f"execution_site.{hardware}.venv {venv!r} in {label} is not named .venv-<arch> in the checkout root, the venv build.sbatch builds"
            raise ValueError(message)
        if venv in built:
            # Both allocations run on one architecture: one build serves both.
            continue
        built.add(venv)
        commands.append(_sbatch_command(allocation, venv))
    if not commands:
        message = f"{label} names no venv in execution_site.cpu or execution_site.gpu"
        raise ValueError(message)
    return commands


def _sbatch_command(allocation: dict[str, object], venv: str) -> list[str]:
    command = ["sbatch"]
    if allocation.get("slurm_account"):
        command += ["-A", str(allocation["slurm_account"])]
    if allocation.get("slurm_partition"):
        command += ["-p", str(allocation["slurm_partition"])]
    # The GPU request the Snakemake SLURM executor makes of a job.
    if allocation.get("gres"):
        command.append(f"--gres={allocation['gres']}")
    elif allocation.get("gpu"):
        model = allocation.get("gpu_model")
        count = allocation["gpu"]
        command.append(
            f"--gpus={model}:{count}" if model else f"--gpus={count}"
        )
    command += shlex.split(str(allocation.get("slurm_extra", "")))
    return [
        *command,
        f"--output={REPO_ROOT}/workflow/venvs/build-%j.log",
        BUILD_SCRIPT,
        venv,
    ]


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Submit workflow/venvs/build.sbatch once per venv a site profile names."
    )
    parser.add_argument(
        "site",
        help="Site profile under workflow/profiles/site/, or a site file",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Print the sbatch commands without submitting them",
    )
    options = parser.parse_args(arguments)
    path = site_file(options.site)
    commands = build_commands(
        read_site_yaml(path.read_text(), str(path)), str(path)
    )
    for command in commands:
        print(shlex.join(command), flush=True)
        if not options.dry_run:
            # build.sbatch builds from its submit directory, the checkout.
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
