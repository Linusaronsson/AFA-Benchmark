"""Real orchestration fixtures with only scripts and SLURM replaced at boundaries."""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from afabench.core.output_layout import (
    PRODUCTION_OUTPUT_ROOT,
    SMOKE_OUTPUT_ROOT,
)

REPO_ROOT = Path(__file__).parents[2]
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
        self.bin = root / "bin"
        self.bin.mkdir()
        for name in [
            "apptainer",
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

    def plan(
        self,
        *options: str,
        target: str = "all_eval_methods",
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        """Plan, with `estimate-compute`, the jobs `run` would submit."""
        return subprocess.run(
            [
                sys.executable,
                str(ESTIMATE_COMPUTE),
                "--output",
                str(self.planned_jobs_path),
                *self._arguments(self._pipeline_invocation(), target, options),
            ],
            cwd=self.root,
            env=self._environment(),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
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
            env=self._environment(),
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
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config))
        return [
            "--profile",
            "none",
            "--snakefile",
            "workflow/snakefiles/orchestration/pipeline.smk",
            "--configfile",
            str(config_path),
        ]

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
            env=self._environment(),
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
        return [
            *invocation,
            "--cores",
            "2",
            "--jobs",
            "4",
            "--printshellcmds",
            target,
            *options,
        ]

    def _environment(self) -> dict[str, str]:
        env = {
            **os.environ,
            # As a shell started in the root sets it; profiles may expand it.
            "PWD": str(self.root),
            "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
            "PATH": f"{self.bin}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "CAPTURE": str(self.capture),
            "ARGUMENTS": str(self.arguments),
        }
        env.pop("SNAKEMAKE_PROFILE", None)
        return env

    def submissions(self) -> list[list[str]]:
        if not self.capture.exists():
            return []
        return [
            json.loads(line) for line in self.capture.read_text().splitlines()
        ]

    def script_arguments(self) -> list[tuple[str, dict[str, str]]]:
        return [
            json.loads(line)
            for line in self.arguments.read_text().splitlines()
        ]
