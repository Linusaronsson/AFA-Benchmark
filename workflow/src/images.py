"""
The image command prefix every rule runs its script command with.

A site profile may name an image per allocation (`execution_site.<cpu|gpu>.
image`). A job whose allocation names one runs its command, job record
wrapper included, inside it through `apptainer exec`; otherwise the prefix is
empty and the command is unchanged. Snakemake itself runs on the host, so it
can submit jobs and call `srun` (docs/adr/0008).

The image holds the locked environment without the project code, which is
read from the checkout: the prefix binds the checkout, its git directory, the
working directory and the output root at their physical paths, because a path
reached through a symlink is not visible in the image. An image built from another `uv.lock`
fails the job before its script runs.
"""

import functools
import shlex
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from execution import REPO_ROOT, ExecutionPolicy, Stage


class Wildcards(Protocol):
    def items(self) -> object: ...


def image_lock(image: Path) -> Path:
    """Return where containers/build.sbatch records the lock `image` was built from."""
    return image.with_name(f"{image.name}.uv.lock")


@functools.cache
def check_image_lock(image: Path, lock: Path) -> None:
    """
    Fail unless `image` was built from the dependencies `lock` pins.

    The image holds its lock too, but reading it would run the image, which
    a node of another architecture cannot: the login node is x86_64 and a GPU
    image aarch64.
    """
    built_from = image_lock(image)
    if not built_from.is_file():
        message = f"No record of the uv.lock image {image} was built from at {built_from}; rebuild it with containers/build.sbatch"
        raise RuntimeError(message)
    if built_from.read_bytes() != lock.read_bytes():
        message = f"Image {image} was built from another uv.lock than {lock}; rebuild it with containers/build.sbatch"
        raise RuntimeError(message)


@functools.cache
def git_directory(checkout: Path) -> Path | None:
    """
    Return the checkout's git directory, or None outside a repository.

    A git worktree keeps it outside the checkout, so it must be bound too, or
    the provenance in the image records no commit.
    """
    try:
        completed = subprocess.run(
            [  # noqa: S607
                "git",
                "-C",
                str(checkout),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return Path(completed.stdout.strip()).resolve()


def _outermost(paths: list[Path]) -> list[Path]:
    """Return `paths` without duplicates and paths inside another of them."""
    kept: list[Path] = []
    for path in sorted(set(paths), key=lambda path: len(path.parts)):
        if not any(path.is_relative_to(parent) for parent in kept):
            kept.append(path)
    return kept


class ImageCommands:
    """Render the image prefix param of each rule."""

    def __init__(
        self, execution: ExecutionPolicy, *, output_root: str
    ) -> None:
        self.execution: ExecutionPolicy = execution
        self.output_root: str = output_root

    def param(
        self, stage: Stage, identity: Callable[[Wildcards], str | None]
    ) -> Callable[[Wildcards, Mapping[str, object]], str]:
        """
        Return a rule's `image` param, rendered per job.

        `identity` is the execution policy identity the rule's allocation
        resources use. The param is empty or ends in a space, so a rule's
        shell command reads `{params.image}python ...`. It takes the
        resources only because Snakemake does not track params that do:
        adding, changing or rebuilding an image reruns no job.
        """

        def image(
            wildcards: Wildcards,
            resources: Mapping[str, object],  # noqa: ARG001
        ) -> str:
            return self.prefix(stage, identity(wildcards))

        return image

    def prefix(self, stage: Stage, identity: str | None) -> str:
        """Return the command prefix of one job, or "" without an image."""
        image = self.execution.image(stage, identity)
        if image is None:
            return ""
        checkout = REPO_ROOT.resolve()
        check_image_lock(image, checkout / "uv.lock")
        working_directory = Path.cwd().resolve()
        arguments = ["apptainer", "exec"]
        if self.execution.hardware(stage, identity) == "gpu":
            arguments.append("--nv")
        binds = [checkout, working_directory, Path(self.output_root).resolve()]
        git = git_directory(checkout)
        if git is not None:
            binds.append(git)
        for path in _outermost(binds):
            arguments += ["--bind", str(path)]
        arguments += [
            "--pwd",
            str(working_directory),
            "--env",
            f"PYTHONPATH={checkout}",
            str(image),
        ]
        return shlex.join(arguments) + " "
