"""Real orchestration fixtures with only scripts and SLURM replaced at boundaries."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parents[2]


class WorkflowHarness:
    def __init__(self, root: Path) -> None:
        self.root = root
        shutil.copytree(REPO_ROOT / "extra/workflow", root / "extra/workflow")
        self.config: dict[str, object] = {
            "pretrain_mapping": {},
            "method_options": {
                name: {"train_script_name": name, "eval_batch_size": 1}
                for name in ["alpha", "beta"]
            },
            "methods": ["alpha", "beta"],
            "datasets": ["cube"],
            "dataset_instance_indices": [0],
            "unmaskers": {"default": "direct"},
            "eval_hard_budgets": {"default": [1]},
            "soft_budget_params": {
                name: {"default": []} for name in ["alpha", "beta"]
            },
            "classifier_names": {"default": "masked_mlp_classifier"},
            "use_wandb": False,
            "smoke_test": True,
        }
        for split in ["train", "val", "test"]:
            (root / f"extra/output/datasets/cube/0/{split}.bundle").mkdir(
                parents=True
            )
        (
            root
            / "extra/output/trained_classifiers/initializer-cold/dataset-cube.bundle"
        ).mkdir(parents=True)
        self.capture = root / "submissions.jsonl"
        self.arguments = root / "arguments.jsonl"
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
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(self.config))
        return self._snakemake(
            [
                "--profile",
                "none",
                "--snakefile",
                "extra/workflow/snakefiles/orchestration/pipeline.smk",
                "--configfile",
                str(config_path),
            ],
            target,
            options,
            timeout,
        )

    def run_invocation(
        self,
        invocation: list[str],
        *options: str,
        target: str = "all",
        timeout: int = 240,
    ) -> subprocess.CompletedProcess[str]:
        return self._snakemake(invocation, target, options, timeout)

    def _snakemake(
        self,
        profile_arguments: list[str],
        target: str,
        options: tuple[str, ...],
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "XDG_CONFIG_HOME": str(self.root / "xdg-config"),
            "PATH": f"{self.bin}:{Path(sys.executable).parent}:{os.environ['PATH']}",
            "CAPTURE": str(self.capture),
            "ARGUMENTS": str(self.arguments),
        }
        env.pop("SNAKEMAKE_PROFILE", None)
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "snakemake",
                *profile_arguments,
                "--cores",
                "2",
                "--jobs",
                "4",
                "--printshellcmds",
                target,
                *options,
            ],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

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
