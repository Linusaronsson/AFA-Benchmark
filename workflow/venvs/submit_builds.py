"""
Submit workflow/venvs/build.sbatch once per venv a site profile names.

    uv run workflow/venvs/submit_builds.py arrhenius [--dry-run]

Each allocation of the site file (`execution_site.<cpu|gpu>`) that names a
venv is built on that allocation, with its account, partition and GPU
request, so they are written down once, in the site profile: a cluster
whose nodes differ in architecture needs a venv built on each (ADR 0015).
The argument is a site profile under workflow/profiles/site/, whose
site.yaml is read, or the path of a site file. The pipeline launcher
imports it to check the venvs.
"""

import re
import shlex
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import typer
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SITES = REPO_ROOT / "workflow" / "profiles" / "site"
# build.sbatch builds only .venv-<arch>, <arch> being the node's `uname -m`,
# the venv workflow/venvs/bin/python runs.
VENV_NAME = re.compile(r"\.venv-[A-Za-z0-9_]+")
BUILD_SCRIPT = "workflow/venvs/build.sbatch"
# Run as a script, this directory is on sys.path, not the checkout root.
sys.path.insert(0, str(REPO_ROOT))
from workflow.src.site_file import read_site_file  # noqa: E402


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


def site_venvs(site: Mapping[str, object], label: str) -> dict[str, str]:
    """Return the venv each allocation of `site` names, by its hardware."""
    return {
        hardware: str(allocation["venv"])
        for hardware, allocation in _allocations(site, label).items()
        if "venv" in allocation
    }


def _allocations(
    site: Mapping[str, object], label: str
) -> dict[str, Mapping[str, object]]:
    """Return the allocations of `site` by their hardware, cpu then gpu."""
    allocations = site.get("execution_site")
    if not isinstance(allocations, Mapping):
        message = f"{label} has no execution_site mapping"
        raise TypeError(message)
    found: dict[str, Mapping[str, object]] = {}
    for hardware in ["cpu", "gpu"]:
        allocation = allocations.get(hardware)
        if allocation is None:
            continue
        if not isinstance(allocation, Mapping):
            message = f"execution_site.{hardware} in {label} is no mapping"
            raise TypeError(message)
        found[hardware] = allocation
    return found


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


def build_commands(site: Mapping[str, object], label: str) -> list[list[str]]:
    """Return one sbatch command per distinct venv the site names."""
    commands: list[list[str]] = []
    built: set[str] = set()
    for hardware, allocation in _allocations(site, label).items():
        if "venv" not in allocation:
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


def _sbatch_command(allocation: Mapping[str, object], venv: str) -> list[str]:
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


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command()
def main(
    site: Annotated[
        str,
        typer.Argument(
            help="Site profile under workflow/profiles/site/, or a site file"
        ),
    ],
    *,
    dry_run: Annotated[
        bool,
        typer.Option(
            "-n",
            "--dry-run",
            help="Print the sbatch commands without submitting them",
        ),
    ] = False,
) -> None:
    """Submit workflow/venvs/build.sbatch once per venv a site profile names."""
    try:
        path = site_file(site)
        commands = build_commands(read_site_file(path), str(path))
    except (FileNotFoundError, TypeError, ValueError, yaml.YAMLError) as error:
        raise typer.BadParameter(str(error), param_hint="SITE") from error
    for command in commands:
        typer.echo(shlex.join(command))
        if not dry_run:
            # build.sbatch builds from its submit directory, the checkout.
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    app(prog_name="uv run workflow/venvs/submit_builds.py")
