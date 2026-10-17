"""Real orchestration fixtures with only scripts and SLURM replaced at boundaries."""

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

from afabench.core.output_layout import (
    PRODUCTION_OUTPUT_ROOT,
    SMOKE_OUTPUT_ROOT,
)
from workflow.run import snakemake_arguments

REPO_ROOT = Path(__file__).parents[2]
# The pipeline profile of the fixture's config, written into its checkout
PIPELINE_PROFILE = "harness"
ESTIMATE_COMPUTE = REPO_ROOT / "scripts/compute_estimate/estimate_compute.py"
# Cluster submission needs a site map; this one mirrors profiles/site/examples/mixed-gres.
SITE = {
    "cpu": {"slurm_partition": "cpu-queue", "slurm_account": "cpu-account"},
    "gpu": {
        "slurm_partition": "gpu-queue",
        "slurm_account": "gpu-account",
        "gres": "gpu:T4:1",
    },
}


def job_counts(output: str) -> dict[str, int]:
    """Read the job counts of Snakemake's job stats in `output`."""
    stats = output.split("Job stats:", 1)[1].split("\n\n\n", 1)[0]
    return {
        rule: int(count)
        for rule, count in re.findall(
            r"^(\w+)\s+(\d+)$", stats, flags=re.MULTILINE
        )
    }


class WorkflowHarness:
    def __init__(self, root: Path, *, smoke_test: bool = True) -> None:
        self.root = root
        # The output root the run writes under, holding its fixtures
        self.output = root / (
            SMOKE_OUTPUT_ROOT if smoke_test else PRODUCTION_OUTPUT_ROOT
        )
        shutil.copytree(REPO_ROOT / "workflow", root / "workflow")
        # Cheap enough to run for real, unlike the stubbed stage scripts
        (root / "scripts/misc").mkdir(parents=True)
        shutil.copyfile(
            REPO_ROOT / "scripts/misc/collect_job_records.py",
            root / "scripts/misc/collect_job_records.py",
        )
        self.config: dict[str, object] = {
            "pretrain_mapping": {},
            "method_options": {
                name: {"train_script_name": name, "eval_batch_size": 1}
                for name in ["alpha", "beta"]
            },
            "methods": ["alpha", "beta"],
            "datasets": ["cube"],
            "dataset_realization_indices": [0],
            "unmaskers": {"default": "direct"},
            "eval_hard_budgets": {"default": [1]},
            "soft_budget_params": {
                name: {"default": []} for name in ["alpha", "beta"]
            },
            "classifier_names": {"default": "masked_mlp_classifier"},
            "use_wandb": False,
            "smoke_test": smoke_test,
            "execution_file": "workflow/profiles/execution/cpu.yaml",
            # Tests read a dry run's job blocks.
            "list_jobs": True,
        }
        for split in ["train", "val", "test"]:
            (self.output / f"datasets/cube/0/{split}.bundle").mkdir(
                parents=True
            )
        (
            self.output
            / "trained_classifiers/initializer-cold/dataset-cube+realization_index-0.bundle"
        ).mkdir(parents=True)
        self.capture = root / "submissions.jsonl"
        self.arguments = root / "arguments.jsonl"
        self.wandb_runs_path = root / "wandb_runs.jsonl"
        self.bin = root / "bin"
        self.bin.mkdir()
        for name in [
            "sbatch",
            "sacct",
            "sacctmgr",
            "srun",
            "scancel",
            "sinfo",
        ]:
            path = self.bin / name
            path.write_text(
                f"#!{sys.executable}\n"
                + (Path(__file__).parent / "fake_slurm.py").read_text()
            )
            path.chmod(0o755)
        for script in [
            "train_method/alpha.py",
            "train_method/beta.py",
            "eval/eval_afa_method.py",
        ]:
            path = root / "scripts" / script
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = dict(arg.split('=', 1) for arg in sys.argv[1:])\n"
                "with Path(os.environ['ARGUMENTS']).open('a') as f:\n"
                "    f.write(json.dumps([sys.argv[0], args]) + '\\n')\n"
                "if 'WANDB_RUN_ID' in os.environ:\n"
                "    with Path(os.environ['WANDB_RUNS']).open('a') as f:\n"
                "        f.write(json.dumps([os.environ['WANDB_RUN_ID'], os.environ['WANDB_NAME'], os.environ.get('WANDB_RUN_GROUP')]) + '\\n')\n"
                "out = Path(args['save_path'])\n"
                "out.parent.mkdir(parents=True, exist_ok=True)\n"
                "if out.suffix == '.bundle': out.mkdir(exist_ok=True)\n"
                "else: out.write_text('fixture')\n"
            )

    def run(
        self,
        *options: str,
        target: str = "all_eval_methods",
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        return self._snakemake(
            self._pipeline_invocation(), target, options, timeout
        )

    def run_invocation(
        self,
        invocation: list[str],
        *options: str,
        target: str = "all",
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        return self._snakemake(invocation, target, options, timeout)

    def run_launcher(
        self, arguments: list[str], timeout: int = 240
    ) -> subprocess.CompletedProcess[str]:
        """Run Snakemake as the pipeline launcher does for `arguments`."""
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "snakemake",
                *snakemake_arguments(arguments, checkout=self.root),
            ],
            cwd=self.root,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def plan(
        self,
        *options: str,
        target: str = "all_eval_methods",
        configuration_source: Sequence[str] = (PIPELINE_PROFILE,),
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        """
        Plan, with `estimate-compute`, the jobs `run` would submit.

        `configuration_source` is the pipeline profile, or `--configfile`
        and the config the pipeline profile names (`config.yaml`).
        """
        self._write_pipeline_profile()
        return subprocess.run(
            [
                sys.executable,
                str(ESTIMATE_COMPUTE),
                "--output",
                str(self.planned_jobs_path),
                # The launcher's arguments for the same Snakemake arguments
                *configuration_source,
                "--target",
                target,
                "--",
                *self._snakemake_options(),
                *options,
            ],
            cwd=self.root,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def invocation_arguments(
        self, *options: str, target: str = "all_eval_methods"
    ) -> list[str]:
        """Return the Snakemake arguments the launcher resolves for a run."""
        self._write_pipeline_profile()
        return snakemake_arguments(
            [
                PIPELINE_PROFILE,
                "--target",
                target,
                "--",
                *self._snakemake_options(),
                *options,
            ],
            checkout=self.root,
        )

    @property
    def planned_jobs_path(self) -> Path:
        return self.root / "planned_jobs.csv"

    def planned_jobs(self) -> list[dict[str, str]]:
        with self.planned_jobs_path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    def submit_first_wave(
        self,
        count: int,
        *options: str,
        target: str,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        """
        Kill Snakemake once `count` jobs were submitted and have run.

        The pinned SLURM plugin waits 40 seconds before its first status
        check, and SIGTERM waits for it too, so only killing keeps this fast.
        """
        process = subprocess.Popen(
            self._command(self._pipeline_invocation(), target, options),
            cwd=self.root,
            env=self.environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        while process.poll() is None and time.monotonic() < deadline:
            if len(list(self.root.glob("job-*.status"))) >= count:
                break
            time.sleep(0.2)
        if process.poll() is None:
            process.kill()
        stdout, _ = process.communicate(timeout=60)
        return subprocess.CompletedProcess(
            process.args, process.returncode, stdout, ""
        )

    def _pipeline_invocation(self) -> list[str]:
        return ["--profile", self._write_pipeline_profile()]

    def _write_pipeline_profile(self) -> str:
        """Write the config and its pipeline profile; return the profile."""
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config))
        profile = f"workflow/profiles/pipeline/{PIPELINE_PROFILE}"
        (self.root / profile).mkdir(parents=True, exist_ok=True)
        (self.root / profile / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "snakefile": "workflow/snakefiles/orchestration/pipeline.smk",
                    "configfile": [str(config_path)],
                }
            )
        )
        return profile

    def _snakemake(
        self,
        invocation: list[str],
        target: str,
        options: tuple[str, ...],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self._command(invocation, target, options),
            cwd=self.root,
            env=self.environment(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def _command(
        self, invocation: list[str], target: str, options: tuple[str, ...]
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "snakemake",
            *self._arguments(invocation, target, options),
        ]

    def _arguments(
        self, invocation: list[str], target: str, options: tuple[str, ...]
    ) -> list[str]:
        """Return the Snakemake arguments of a run."""
        return [*invocation, *self._snakemake_options(), target, *options]

    def _snakemake_options(self) -> list[str]:
        return ["--cores", "2", "--jobs", "4", "--printshellcmds"]

    def environment(self) -> dict[str, str]:
        env = {
            **os.environ,
            # As a shell started in the root sets it; profiles may expand it.
            "PWD": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
            "PATH": f"{self.bin}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "CAPTURE": str(self.capture),
            "ARGUMENTS": str(self.arguments),
            "WANDB_RUNS": str(self.wandb_runs_path),
        }
        env.pop("SNAKEMAKE_PROFILE", None)
        return env

    def submissions(self) -> list[list[str]]:
        if not self.capture.exists():
            return []
        return [
            json.loads(line) for line in self.capture.read_text().splitlines()
        ]

    def wandb_runs(self) -> list[tuple[str, str, str | None]]:
        """Return the W&B run ID, name and group each script was given."""
        if not self.wandb_runs_path.exists():
            return []
        return [
            tuple(json.loads(line))
            for line in self.wandb_runs_path.read_text().splitlines()
        ]

    def script_arguments(self) -> list[tuple[str, dict[str, str]]]:
        return [
            json.loads(line)
            for line in self.arguments.read_text().splitlines()
        ]
