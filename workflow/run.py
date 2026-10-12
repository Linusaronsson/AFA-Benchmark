#!/usr/bin/env python3
"""
Pipeline launcher: run the pipeline from its profiles and overrides.

A pipeline profile, an optional site profile and pipeline profile overrides:

    workflow/run.py [<pipeline>] [--site <site>] [-n | -e] [--no-soft-budget]
        [--wandb] [--target <rule>] [--jobs <n>] [key=value ...]
        [-- <Snakemake arguments>]

It builds the one Snakemake command for them, adding what follows from the
site profile: its site file, which Snakemake would otherwise drop from the
site profile whenever overrides are passed, and the interpreter of a site
whose jobs run in an image. See docs/adr/0009. --no-soft-budget and --wandb
stand for the overrides soft_budget_setting=false and use_wandb=true.

With -e it runs the compute estimate of that invocation instead
(scripts/compute_estimate/estimate_compute.py), which takes the same
arguments and, with -e only, the estimate's options. On a site whose jobs
run in an image, the estimate runs in the image: it needs both Snakemake and
the libraries the orchestration environment lacks.

It runs on a login node before any environment exists, like
containers/submit_builds.py, so on the host's python3 (3.9 on Arrhenius)
with the standard library only: argparse instead of typer.
"""

from __future__ import annotations

import argparse
import os
import re
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
ESTIMATE_SCRIPT = "scripts/compute_estimate/estimate_compute.py"
ESTIMATE_FLAGS = ("-e", "--estimate")
CATALOGS = {
    "methods": "workflow/conf/methods/all.yaml",
    "datasets": "workflow/conf/datasets/all.yaml",
    "unmaskers": "conf/components/unmasker",
    "initializers": "conf/components/initializer",
}
# The estimate's own options, taken with -e only.
ESTIMATE_OPTIONS = ("job_durations", "by", "output", "strict")


class LauncherError(ValueError):
    """Launcher arguments that name no profile or contradict each other."""


def launcher_command(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the command the launcher runs for `arguments`."""
    options = _parse(arguments)
    if not options.estimate:
        return snakemake_command(arguments, checkout)
    if options.dry_run:
        message = (
            "-n and -e: an estimate runs no jobs either; give one of them"
        )
        raise LauncherError(message)
    # Check the run's arguments, as the estimate will.
    snakemake_arguments(arguments, checkout)
    interpreter = ["uv", "run", "python"]
    if options.site is not None and _names_an_image(
        checkout,
        f"{_profile(checkout, SITE_PROFILES, options.site)}/site.yaml",
    ):
        interpreter = ["containers/run.sh", "python"]
    return [*interpreter, ESTIMATE_SCRIPT, *_without_estimate_flag(arguments)]


def _without_estimate_flag(arguments: list[str]) -> list[str]:
    split = arguments.index("--") if "--" in arguments else len(arguments)
    return [
        argument
        for argument in arguments[:split]
        if argument not in ESTIMATE_FLAGS
    ] + arguments[split:]


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
    options = _parse(arguments)
    pipeline, config = _pipeline_and_overrides(options.profile_and_overrides)
    config += _flag_overrides(options, config)
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


def _flag_overrides(
    options: argparse.Namespace, overrides: list[str]
) -> list[str]:
    """Return the overrides the launcher's flags stand for."""
    added = []
    if options.no_soft_budget:
        added.append(_soft_budget_setting_off(overrides))
    if options.wandb:
        added.append(_wandb_logging_on(overrides))
    return added


def _soft_budget_setting_off(overrides: list[str]) -> str:
    """Return the override --no-soft-budget stands for."""
    if any(word.startswith("soft_budget_setting=") for word in overrides):
        message = "--no-soft-budget and soft_budget_setting=: give one of them"
        raise LauncherError(message)
    return "soft_budget_setting=false"


def _wandb_logging_on(overrides: list[str]) -> str:
    """Return the override --wandb stands for."""
    if any(word.startswith("use_wandb=") for word in overrides):
        message = "--wandb and use_wandb=: give one of them"
        raise LauncherError(message)
    return "use_wandb=true"


def _pipeline_and_overrides(words: list[str]) -> tuple[str, list[str]]:
    pipeline = "all"
    if words and "=" not in words[0]:
        pipeline, words = words[0], words[1:]
    for word in words:
        if "=" not in word:
            message = f"{word!r} is no pipeline profile override (key=value); the pipeline profile, {pipeline!r}, comes first"
            raise LauncherError(message)
    return pipeline, list(words)


def _parse(arguments: list[str]) -> argparse.Namespace:
    """Parse the launcher's arguments, before any --."""
    if "--" in arguments:
        arguments = arguments[: arguments.index("--")]
    parser = _parser()
    options = parser.parse_intermixed_args(arguments)
    requested = [name for name in CATALOGS if getattr(options, "list_" + name)]
    if requested:
        for name in requested:
            print(f"Available {name} ({CATALOGS[name]}):")
            print("\n".join(available_names(name)))
        parser.exit()
    if not options.estimate:
        for name in ESTIMATE_OPTIONS:
            if getattr(options, name) not in (None, False):
                flag = "--" + name.replace("_", "-")
                message = (
                    f"{flag} is an option of the compute estimate: add -e"
                )
                raise LauncherError(message)
    return options


def available_names(kind: str, checkout: Path = REPO_ROOT) -> list[str]:
    """List shipped names using only the login node's standard library."""
    if kind not in CATALOGS:
        msg = f"Unknown catalog {kind!r}; choose from {', '.join(CATALOGS)}"
        raise LauncherError(msg)
    path = checkout / CATALOGS[kind]
    if kind in {"unmaskers", "initializers"}:
        names = sorted(config.stem for config in path.glob("*.yaml"))
        if not names:
            msg = f"{path}: no {kind} configs"
            raise LauncherError(msg)
        return names
    # These two catalogs use a flat list of unquoted names, not general YAML.
    names: list[str] = []
    header_seen = False
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not header_seen and line == f"{kind}:":
            header_seen = True
            continue
        entry = re.fullmatch(r" +\- ([a-z][a-z0-9_]*)", line)
        if not header_seen or entry is None:
            message = f"{path}:{number}: expected {kind}: followed by a flat list of names, got {raw!r}"
            raise LauncherError(message)
        names.append(entry.group(1))
    if not names:
        msg = f"{path}: no {kind} names"
        raise LauncherError(msg)
    return sorted(set(names))


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
    for kind in CATALOGS:
        parser.add_argument(
            f"--list-{kind}",
            action="store_true",
            help=f"List all shipped {kind} and exit, without running jobs",
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
        "-e",
        "--estimate",
        action="store_true",
        help="Print the compute estimate of the run without running jobs",
    )
    parser.add_argument(
        "--no-soft-budget",
        action="store_true",
        help="Run only the hard-budget setting (soft_budget_setting=false)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B and link each job's W&B run in the terminal (use_wandb=true); needs WANDB_ENTITY and WANDB_PROJECT",
    )
    parser.add_argument(
        "--target", default="all", help="The rule to run (default: all)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="How many jobs run at once locally; a site profile sets it for its cluster",
    )
    estimate = parser.add_argument_group(
        "compute estimate options, with -e (docs/reference/estimate_compute_command.md)"
    )
    estimate.add_argument(
        "--job-durations",
        help="A job duration table, or an output root of job records (default: output/production)",
    )
    estimate.add_argument(
        "--by",
        action="append",
        help="Group the totals by this job column; repeat for several",
    )
    estimate.add_argument(
        "--output", help="Also write each planned job's estimate here as CSV"
    )
    estimate.add_argument(
        "--strict",
        action="store_true",
        help="Exit with 1 when a planned job is unestimated",
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
        command = launcher_command(arguments)
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
