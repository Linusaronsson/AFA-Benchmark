"""
Pin the release manifest's table enumeration to the workflow's targets.

`afabench.release.manifest` enumerates evaluation tables from the resolved
workflow config instead of parsing paths. A Snakemake dry run asked for
exactly those paths, on top of `all_eval_methods`, must schedule one
evaluation and one transform per manifest entry: a path the workflow cannot
produce fails the dry run, and a target the manifest misses adds a job.
"""

import re
from pathlib import Path

import yaml

from afabench.release.manifest import ReleaseScope, build_release_manifest
from afabench.release.workflow_config import resolve_workflow_config
from test.workflow.submission_harness import WorkflowHarness


def test_release_manifest_tables_match_workflow_targets(
    tmp_path: Path,
) -> None:
    harness = WorkflowHarness(tmp_path / "workflow")
    harness.config["pretrain_mapping"] = {
        "shared": {"pretrain_script_name": "beta"}
    }
    harness.config["method_options"]["beta"] |= {
        "pretrained_model_name": "shared",
        "classifier": {"script_name": "beta_classifier"},
    }
    harness.config["dataset_instance_indices"] = [0, 1]
    harness.config["eval_hard_budgets"] = {"default": [1, 2]}
    harness.config["soft_budget_params"]["alpha"] = {
        "default": [[0.5, None], [0.1, 0.2]]
    }
    configfile = tmp_path / "workflow.yaml"
    configfile.write_text(yaml.safe_dump(harness.config))
    manifest = build_release_manifest(
        release_id="pin",
        scope=ReleaseScope.TEST_ONLY,
        workflow_config=resolve_workflow_config(
            profile=None, configfiles=[configfile], overrides=[]
        ),
        output_root=tmp_path / "workflow/extra/output",
        checkout=tmp_path,
    )
    tables = manifest.evaluation_tables
    targets = [
        f"extra/output/{path}"
        for table in tables
        for path in [
            table.raw_path,
            table.transformed_path,
            table.classifier_bundle_path,
        ]
    ]

    # Snakemake needs its positional targets contiguous.
    result = harness.run(*targets, "--dry-run", target="all_eval_methods")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # alpha: 2 instances x (2 hard + 2 soft); beta: 2 instances x 2 hard.
    assert len(tables) == 12
    assert _job_count(output, "eval_method") == len(tables)
    assert _job_count(output, "transform_eval_data") == len(tables)


def _job_count(output: str, rule: str) -> int:
    match = re.search(rf"^{rule}\s+(\d+)$", output, flags=re.MULTILINE)
    assert match is not None, output
    return int(match.group(1))
