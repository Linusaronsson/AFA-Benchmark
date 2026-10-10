#!/usr/bin/env python3
"""
Pipeline launcher: run the pipeline from its profiles and overrides.

A pipeline profile, an optional site profile and pipeline profile overrides:

    workflow/run.py [<pipeline>] [--site <site>] [-n] [--target <rule>]
        [--jobs <n>] [key=value ...] [-- <Snakemake arguments>]

It builds the one Snakemake command for them, adding what follows from the
site profile: its site file, which Snakemake would otherwise drop from the
site profile whenever overrides are passed, and the interpreter of a site
whose jobs run in an image. See docs/adr/0009.

It runs on a login node before any environment exists, like
containers/submit_builds.py, so on the host's python3 (3.9 on Arrhenius)
with the standard library only: argparse instead of typer.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, this directory is on sys.path, not the checkout root.
sys.path.insert(0, str(REPO_ROOT))
from containers.submit_builds import read_site_yaml  # noqa: E402

PIPELINE_PROFILES = "workflow/profiles/pipeline"
SITE_PROFILES = "workflow/profiles/site"


class LauncherError(ValueError):
    """Launcher arguments that name no profile or contradict each other."""


def snakemake_command(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the Snakemake command the launcher runs for `arguments`."""
    interpreter, snakemake = _invocation(arguments, checkout)
    return [*interpreter, *snakemake]


def snakemake_arguments(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the arguments of that command after `snakemake`."""
    return _invocation(arguments, checkout)[1]


def _invocation(
    arguments: list[str], checkout: Path
) -> tuple[list[str], list[str]]:
    """Return the interpreter running Snakemake, and Snakemake's arguments."""
    extra: list[str] = []
    if "--" in arguments:
        split = arguments.index("--")
        arguments, extra = arguments[:split], arguments[split + 1 :]
    if any(argument.startswith("--config") for argument in extra):
        # A second --config would replace the one carrying the site file.
        message = "--config after --: give pipeline profile overrides as key=value before --"
        raise LauncherError(message)
    options = _parser().parse_intermixed_args(arguments)
    pipeline, config = _pipeline_and_overrides(options.profile_and_overrides)
    interpreter = ["uv", "run", "snakemake"]
    profiles = ["--profile", _profile(checkout, PIPELINE_PROFILES, pipeline)]
    if options.site is not None:
        if options.jobs is not None:
            message = f"--jobs {options.jobs}: a cluster run submits as many jobs as the site profile {options.site!r} sets"
            raise LauncherError(message)
        site = _profile(checkout, SITE_PROFILES, options.site)
        site_file = f"{site}/site.yaml"
        if _names_an_image(checkout, site_file):
            # Jobs run in an image, and the host has only the orchestration
            # environment containers/build.sbatch built (ADR 0008).
            interpreter = ["containers/bin/python", "-m", "snakemake"]
        profiles += ["--workflow-profile", site]
        config.append(f"execution_site_file={site_file}")
    # The target first: a multi-value option among the extra arguments
    # would take it as one of its values.
    command = [*profiles, options.target]
    if options.jobs is not None:
        command += ["--jobs", str(options.jobs)]
    if options.dry_run:
        command += ["--dry-run", "--printshellcmds"]
    command += extra
    if config:
        command += ["--config", *config]
    return interpreter, command


def _pipeline_and_overrides(words: list[str]) -> tuple[str, list[str]]:
    pipeline = "all"
    if words and "=" not in words[0]:
        pipeline, words = words[0], words[1:]
    for word in words:
        if "=" not in word:
            message = f"{word!r} is no pipeline profile override (key=value); the pipeline profile, {pipeline!r}, comes first"
            raise LauncherError(message)
    return pipeline, list(words)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="workflow/run.py",
        description="Run the pipeline.",
        epilog="Arguments after -- go to Snakemake unchanged.",
    )
    parser.add_argument(
        "profile_and_overrides",
        nargs="*",
        metavar="[pipeline] [key=value ...]",
        help=f"A pipeline profile under {PIPELINE_PROFILES}/ (default: all), then pipeline profile overrides",
    )
    parser.add_argument(
        "--site",
        help=f"A site profile under {SITE_PROFILES}/; without one, jobs run locally",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Plan the run and print its commands without running jobs",
    )
    parser.add_argument(
        "--target", default="all", help="The rule to run (default: all)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="How many jobs run at once locally; a site profile sets it for its cluster",
    )
    return parser


def _profile(checkout: Path, profiles: str, name: str) -> str:
    """Return the path of the profile `name` under `profiles`."""
    path = f"{profiles}/{name}"
    if not (checkout / path / "config.yaml").is_file():
        shipped = sorted(
            str(config.parent.relative_to(checkout / profiles))
            for config in (checkout / profiles).rglob("config.yaml")
        )
        message = f"No profile {name!r} under {profiles}; there are: {', '.join(shipped)}"
        raise LauncherError(message)
    return path


def _names_an_image(checkout: Path, site_file: str) -> bool:
    if not (checkout / site_file).is_file():
        message = f"The site profile has no site file {site_file}"
        raise LauncherError(message)
    site = read_site_yaml((checkout / site_file).read_text(), site_file)
    allocations = site.get("execution_site")
    if not isinstance(allocations, dict):
        message = f"{site_file} has no execution_site mapping"
        raise LauncherError(message)
    return any(
        isinstance(allocation, dict) and "image" in allocation
        for allocation in allocations.values()
    )


def main(arguments: list[str]) -> int:
    try:
        command = snakemake_command(arguments)
    except LauncherError as error:
        print(f"workflow/run.py: error: {error}", file=sys.stderr)
        return 2
    print(shlex.join(command), file=sys.stderr, flush=True)
    # Site profiles expand $PWD when Snakemake reads them, as the checkout
    # root, which a changed working directory alone does not set.
    environment = {**os.environ, "PWD": str(REPO_ROOT)}
    return subprocess.run(
        command, cwd=REPO_ROOT, env=environment, check=False
    ).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
