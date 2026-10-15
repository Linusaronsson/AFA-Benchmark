#!/usr/bin/env python3
"""
Pipeline launcher: run the pipeline from its profiles and overrides.

A pipeline profile, an optional site profile and pipeline profile overrides:

    workflow/run.py [<pipeline>] [--site <site>] [-n | -e] [--no-soft-budget]
        [--per-dataset-plots] [--wandb] [--wandb-group <name>]
        [--target <rule>] [--jobs <n>]
        [key=value ...] [-- <Snakemake arguments>]

It builds the one Snakemake command for them, adding what follows from the
site profile: its site file, which Snakemake would otherwise drop from the
site profile whenever overrides are passed, and the interpreter of a site
whose allocations name venvs: workflow/venvs/bin/python, the venv of the
login node's architecture, which it refuses to run while any of the site's
venvs is missing or built from another uv.lock. See docs/adr/0009 and docs/adr/0015. --no-soft-budget and --wandb
stand for the overrides soft_budget_setting=false and use_wandb=true,
--per-dataset-plots for per_dataset_plots=true, and
--wandb-group <name> for use_wandb=true wandb_group=<name>. With W&B logging
and no group, the launcher names a new W&B group for the invocation, from
the pipeline profile, the UTC time and random characters.

With -e it runs the compute estimate of that invocation instead
(scripts/compute_estimate/estimate_compute.py), which takes the same
arguments and, with -e only, the estimate's options, with the same
interpreter.

It runs on a login node before any venv exists, like
workflow/venvs/submit_builds.py, so on the host's python3 (3.9 on Arrhenius)
with the standard library only: argparse instead of typer.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import shlex
import string
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, this directory is on sys.path, not the checkout root.
sys.path.insert(0, str(REPO_ROOT))
from workflow.venvs.submit_builds import (  # noqa: E402
    read_site_yaml,
    site_venvs,
    venv_problem,
)

PIPELINE_PROFILES = "workflow/profiles/pipeline"
SITE_PROFILES = "workflow/profiles/site"
ESTIMATE_SCRIPT = "scripts/compute_estimate/estimate_compute.py"
# The Python of the node's venv, on a site whose allocations name venvs
VENV_PYTHON = "workflow/venvs/bin/python"
ESTIMATE_FLAGS = ("-e", "--estimate")
CATALOGS = {
    "methods": "workflow/conf/methods/all.yaml",
    "datasets": "workflow/conf/datasets/all.yaml",
    "unmaskers": "conf/components/unmasker",
    "initializers": "conf/components/initializer",
}
# The estimate's own options, taken with -e only.
ESTIMATE_OPTIONS = ("job_durations", "by", "output", "strict")
WANDB_GROUP_SUFFIX_LENGTH = 4


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
    if options.site is not None and _site_venvs(checkout, options.site):
        interpreter = [VENV_PYTHON]
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
    if _override_is_true(config, "use_wandb") and not _overrides(
        config, "wandb_group"
    ):
        smoke_test = _override_is_true(config, "smoke_test")
        config.append(
            f"wandb_group={new_wandb_group(pipeline, smoke_test=smoke_test)}"
        )
    interpreter = ["uv", "run", "snakemake"]
    profiles = ["--profile", _profile(checkout, PIPELINE_PROFILES, pipeline)]
    if options.site is not None:
        if options.jobs is not None:
            message = f"--jobs {options.jobs}: a cluster run submits as many jobs as the site profile {options.site!r} sets"
            raise LauncherError(message)
        site = _profile(checkout, SITE_PROFILES, options.site)
        if _site_venvs(checkout, options.site):
            interpreter = [VENV_PYTHON, "-m", "snakemake"]
        profiles += ["--workflow-profile", site]
        config.append(f"execution_site_file={site}/site.yaml")
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
    if options.per_dataset_plots:
        if _overrides(overrides, "per_dataset_plots"):
            message = (
                "--per-dataset-plots and per_dataset_plots=: give one of them"
            )
            raise LauncherError(message)
        added.append("per_dataset_plots=true")
    if options.wandb or options.wandb_group is not None:
        added.append(_wandb_logging_on(overrides))
    if options.wandb_group is not None:
        added.append(_named_wandb_group(options.wandb_group, overrides))
    return added


def _soft_budget_setting_off(overrides: list[str]) -> str:
    """Return the override --no-soft-budget stands for."""
    if any(word.startswith("soft_budget_setting=") for word in overrides):
        message = "--no-soft-budget and soft_budget_setting=: give one of them"
        raise LauncherError(message)
    return "soft_budget_setting=false"


def _wandb_logging_on(overrides: list[str]) -> str:
    """Return the override --wandb stands for."""
    if _overrides(overrides, "use_wandb"):
        message = "--wandb or --wandb-group, and use_wandb=: give one of them"
        raise LauncherError(message)
    return "use_wandb=true"


def _named_wandb_group(name: str, overrides: list[str]) -> str:
    """Return the override --wandb-group stands for."""
    if not name:
        message = "--wandb-group: give the W&B group a name"
        raise LauncherError(message)
    if _overrides(overrides, "wandb_group"):
        message = "--wandb-group and wandb_group=: give one of them"
        raise LauncherError(message)
    return f"wandb_group={name}"


def new_wandb_group(pipeline: str, *, smoke_test: bool) -> str:
    """Name a new W&B group for an invocation of `pipeline`."""
    time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(
        random.choices(
            string.ascii_lowercase + string.digits,
            k=WANDB_GROUP_SUFFIX_LENGTH,
        )
    )
    return f"{pipeline}-{time}-{suffix}" + ("-smoke" if smoke_test else "")


def _overrides(overrides: list[str], key: str) -> list[str]:
    """Return the values the overrides give `key`."""
    return [
        word.split("=", 1)[1]
        for word in overrides
        if word.startswith(f"{key}=")
    ]


def _override_is_true(overrides: list[str], key: str) -> bool:
    values = _overrides(overrides, key)
    return bool(values) and values[-1].lower() == "true"


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
        "--per-dataset-plots",
        action="store_true",
        help="Add per-dataset Accuracy, Macro F1 and AUROC figures for both classifiers (per_dataset_plots=true)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log to W&B and link each job's W&B run in the terminal (use_wandb=true), every run in a new W&B group; needs WANDB_ENTITY and WANDB_PROJECT",
    )
    parser.add_argument(
        "--wandb-group",
        metavar="NAME",
        help="Log to W&B as --wandb does, every run in the W&B group NAME, such as an earlier invocation's to continue it (use_wandb=true wandb_group=NAME)",
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


def _site_venvs(checkout: Path, site: str) -> dict[str, str]:
    """Return the venv each allocation of the site profile `site` names."""
    site_file = f"{_profile(checkout, SITE_PROFILES, site)}/site.yaml"
    if not (checkout / site_file).is_file():
        message = f"The site profile has no site file {site_file}"
        raise LauncherError(message)
    try:
        return site_venvs(
            read_site_yaml((checkout / site_file).read_text(), site_file),
            site_file,
        )
    except TypeError as error:
        raise LauncherError(str(error)) from error


def check_site_venvs(arguments: list[str], checkout: Path = REPO_ROOT) -> None:
    """Fail unless every venv the run's site names is built from uv.lock."""
    site = _parse(arguments).site
    if site is None:
        return
    problems = [
        problem
        for venv in sorted(set(_site_venvs(checkout, site).values()))
        if (problem := venv_problem(checkout, venv)) is not None
    ]
    if problems:
        message = f"{'; '.join(problems)}. Build the site's venvs with `just build-venvs {site}` (docs/how-to/slurm_integration.md), and run this again once those jobs finished"
        raise LauncherError(message)


def main(arguments: list[str]) -> int:
    try:
        command = launcher_command(arguments)
        check_site_venvs(arguments)
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
