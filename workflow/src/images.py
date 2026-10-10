"""
The image command prefix every rule runs its script command with.

A site profile may name an image per allocation (`execution_site.<cpu|gpu>.
image`). A job whose allocation names one runs its command, job record
wrapper included, inside it through `apptainer exec`; otherwise the prefix is
empty and the command is unchanged. Snakemake itself runs on the host, so it
can submit jobs and call `srun` (docs/adr/0008).

The image holds the locked environment without the project code, which is
read from the checkout: the prefix binds the checkout, the working directory
and the output root at their physical paths, because a path reached through a
symlink is not visible in the image. An image built from another `uv.lock`
fails the job before its script runs.
"""

import functools
import shlex
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from execution import REPO_ROOT, ExecutionPolicy, Stage

# Where containers/afabench.def copies the lock the image was built from
IMAGE_LOCK = "/opt/afabench/uv.lock"


class Wildcards(Protocol):
    def items(self) -> object: ...


@functools.cache
def check_image_lock(image: Path, lock: Path) -> None:
    """Fail unless `image` was built from the dependencies `lock` pins."""
    result = subprocess.run(
        ["apptainer", "exec", "--pwd", "/", str(image), "cat", IMAGE_LOCK],  # noqa: S607
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = f"Cannot read {IMAGE_LOCK} from image {image}: {result.stderr.decode(errors='replace').strip()}"
        raise RuntimeError(message)
    if result.stdout != lock.read_bytes():
        message = f"Image {image} was built from another uv.lock than {lock}; rebuild it with containers/build.sbatch"
        raise RuntimeError(message)


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
        for path in _outermost(
            [checkout, working_directory, Path(self.output_root).resolve()]
        ):
            arguments += ["--bind", str(path)]
        arguments += [
            "--pwd",
            str(working_directory),
            "--env",
            f"PYTHONPATH={checkout}",
            str(image),
        ]
        return shlex.join(arguments) + " "
