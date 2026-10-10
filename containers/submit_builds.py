#!/usr/bin/env python3
"""
Submit containers/build.sbatch once per image a site profile names.

    python3 containers/submit_builds.py arrhenius [--dry-run]

Each allocation of the site file (`execution_site.<cpu|gpu>`) that names an
image is built on that allocation, with its account, partition and GPU
request, so they are written down once, in the site profile. The argument is
a site profile under workflow/profiles/site/, whose site.yaml is read, or the
path of a site file.

This runs on the login node before any environment exists, since the
orchestration environment is what the build creates. It therefore runs on
the host's python3 (3.9 on Arrhenius) with the standard library only:
argparse instead of typer, and a reader for the subset of YAML that site
files use instead of PyYAML.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SITES = REPO_ROOT / "workflow" / "profiles" / "site"
# build.sbatch writes afabench-<arch>.sif, <arch> being the node's `uname -m`
IMAGE_NAME = re.compile(r"afabench-[A-Za-z0-9_]+\.sif")


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


def build_commands(site: dict[str, object], label: str) -> list[list[str]]:
    """Return one sbatch command per distinct image the site names."""
    allocations = site.get("execution_site")
    if not isinstance(allocations, dict):
        message = f"{label} has no execution_site mapping"
        raise TypeError(message)
    commands: list[list[str]] = []
    images: set[Path] = set()
    for hardware in ["cpu", "gpu"]:
        allocation = allocations.get(hardware)
        if not isinstance(allocation, dict) or "image" not in allocation:
            continue
        image = REPO_ROOT / str(allocation["image"])
        if not IMAGE_NAME.fullmatch(image.name):
            message = f"execution_site.{hardware}.image {allocation['image']!r} in {label} is not named afabench-<arch>.sif, the image containers/build.sbatch writes"
            raise ValueError(message)
        if image in images:
            # Both allocations run on one architecture: one build serves both.
            continue
        images.add(image)
        commands.append(_sbatch_command(allocation, image.parent))
    if not commands:
        message = f"{label} names no image in execution_site.cpu or execution_site.gpu"
        raise ValueError(message)
    return commands


def _sbatch_command(
    allocation: dict[str, object], image_dir: Path
) -> list[str]:
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
        f"--output={image_dir}/build-%j.log",
        "containers/build.sbatch",
        str(image_dir),
    ]


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Submit containers/build.sbatch once per image a site profile names."
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
