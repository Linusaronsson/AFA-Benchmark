"""
Repository adopter and results-only user workflows on a real smoke run (#41).

A published workspace runs the real pipeline on CUBE with `random_dummy`
and `gdfs` (see `test_native_bundle_restore`), saves a smoke release
and publishes it to a fake release host. A separate fork downloads the
baselines' plotting-ready tables and the shared prerequisites, adds
`gdfs_adopter`, a configured variant of GDFS, and runs the ordinary `all`
target with the baselines as reference methods. The comparison is a
workflow demonstration on smoke runs, not a scientific result, and its
plots say so.
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from afabench.release.manifest import ExecutionMode, read_release_manifest
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport
from test.workflow.test_native_bundle_restore import (
    PROFILE_CONFIGFILES,
    REPO_ROOT,
    SMOKE_SELECTION,
    workspace,
)

RELEASE_ID = "smoke-adopter-journey"
REPO_ID = "afabench-test/releases"
TAG = "initializer-cold"
CAPTION = "Workflow demonstration from smoke runs, not scientific results"
# Layered over the release's own workflow configuration, so the reference
# tables are expected exactly where the release has them.
ADOPTER_CONFIG: dict[str, Any] = {
    "methods": ["gdfs_adopter"],
    "reference_methods": ["random_dummy", "gdfs"],
    "method_options": {
        "gdfs_adopter": {
            "pretrained_model_name": "gdfs",
            "train_script_name": "gdfs",
            "method_specific_params": ["lr=0.0005"],
            "use_max_hard_budget_when_training_soft_budget": True,
            "eval_batch_size": {"default": 32},
        }
    },
    "method_sets": {
        "adopter_comparison": ["random_dummy", "gdfs", "gdfs_adopter"]
    },
    "soft_budget_params": {"gdfs_adopter": {"cube": []}},
}
# The shared prerequisites a new method on CUBE needs, as the fork has them
# after downloading: the dataset realization's splits, the external classifier
# and the pretrained model GDFS variants train from.
SHARED_PREREQUISITES = [
    "datasets/cube/0/train.bundle",
    "datasets/cube/0/val.bundle",
    "datasets/cube/0/test.bundle",
    f"trained_classifiers/{TAG}/dataset-cube+realization_index-0.bundle",
    f"pretrained_models/{TAG}/gdfs/dataset-cube+realization_index-0/"
    "pretrain_seed-0/model.bundle",
]
EVAL_PERF = f"eval_split-test/{TAG}/eval_perf"
COMPARISON = "method_set-adopter_comparison"
# Only the variant's training, evaluation and transformation, the job
# duration table, and the comparison's aggregation and plots.
ADOPTER_JOBS = {
    "all": 1,
    "eval_method": 1,
    "merge_eval_perf": 1,
    "collect_job_records": 1,
    "plot_eval_perf": 2,
    "plot_time": 1,
    "split_by_classifier_type": 1,
    "train_method": 1,
    "transform_eval_data": 1,
    "total": 10,
}


def fork(root: Path) -> Path:
    """
    Make a checkout-shaped fork that owns its config, as a fork does.

    The display configuration names the variant, and the comparison's
    plots carry the workflow-demonstration caption.
    """
    workspace(root)
    (root / "extra/conf").unlink()
    shutil.copytree(REPO_ROOT / "extra/conf", root / "extra/conf")
    display_path = root / "extra/conf/scripts/plotting/common/default.yaml"
    display = yaml.safe_load(display_path.read_text())
    display["method_name_mapping"]["gdfs_adopter"] = "GDFS (adopter)"
    display["method_policy_family_mapping"]["gdfs_adopter"] = "gdfs"
    display["caption"] = CAPTION
    display_path.write_text(yaml.safe_dump(display))
    return root


def snakemake(
    root: Path, configfiles: list[Path], *arguments: str
) -> subprocess.CompletedProcess[str]:
    # No Hugging Face token, and no service: the pipeline reads local files.
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("HF_") and name != "SNAKEMAKE_PROFILE"
    }
    environment |= {
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": str(root / "no-hf-home"),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            "extra/workflow/snakefiles/orchestration/pipeline.smk",
            "--configfile",
            *(str(path) for path in configfiles),
            "--cores",
            "4",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )


def job_counts(output: str) -> dict[str, int]:
    stats = output.split("Job stats:", 1)[1].split("\n\n\n", 1)[0]
    return {
        rule: int(count)
        for rule, count in re.findall(
            r"^(\w+)\s+(\d+)$", stats, flags=re.MULTILINE
        )
    }


def invoke(transport: FakeReleaseTransport, *args: str) -> Result:
    return CliRunner().invoke(
        app, [*args, "--repo-id", REPO_ID], obj=lambda _repo_id: transport
    )


def files_and_mtimes(root: Path) -> dict[str, float]:
    return {
        path.relative_to(root).as_posix(): path.stat().st_mtime
        for path in root.rglob("*")
        if path.is_file()
    }


class Journey:
    published: Path
    forked: Path
    transport: FakeReleaseTransport
    plan: subprocess.CompletedProcess[str]
    run: subprocess.CompletedProcess[str]
    downloaded: dict[str, float]


@pytest.fixture(scope="module")
def journey(tmp_path_factory: pytest.TempPathFactory) -> Journey:
    tmp_path = tmp_path_factory.mktemp("adopter-journey")
    result = Journey()

    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(yaml.safe_dump(SMOKE_SELECTION))
    baseline_configfiles = [*PROFILE_CONFIGFILES, baseline]
    result.published = workspace(tmp_path / "published")
    run = snakemake(result.published, baseline_configfiles, "all")
    assert run.returncode == 0, run.stdout + run.stderr

    package = tmp_path / "package"
    saved = CliRunner().invoke(
        app,
        [
            "save",
            str(package),
            "--source-root",
            str(result.published / "extra/output"),
            *(
                argument
                for path in baseline_configfiles
                for argument in ["--configfile", str(path)]
            ),
            "--release-id",
            RELEASE_ID,
            "--scope",
            "smoke",
            "--checkout",
            str(REPO_ROOT),
        ],
    )
    assert saved.exit_code == 0, saved.output
    result.transport = FakeReleaseTransport()
    published = invoke(
        result.transport, "publish", str(package), "--smoke-release"
    )
    assert published.exit_code == 0, published.output

    result.forked = fork(tmp_path / "fork")
    downloaded = invoke(
        result.transport,
        "download",
        RELEASE_ID,
        "--smoke-release",
        "--destination-root",
        str(result.forked / "extra/output"),
        "--payload-category",
        "transformed_evaluation_table",
        "--payload-category",
        "dataset_bundle",
        "--payload-category",
        "classifier_bundle",
        "--payload-category",
        "pretrained_model_bundle",
    )
    assert downloaded.exit_code == 0, downloaded.output
    result.downloaded = files_and_mtimes(result.forked / "extra/output")

    release_config = tmp_path / "release_config.json"
    release_config.write_text(
        json.dumps(
            json.loads(
                (result.forked / "extra/release_manifest.json").read_text()
            )["workflow_config"]["merged"]
        )
    )
    adopter = tmp_path / "adopter.yaml"
    adopter.write_text(yaml.safe_dump(ADOPTER_CONFIG))
    adopter_configfiles = [release_config, adopter]
    result.plan = snakemake(
        result.forked, adopter_configfiles, "--dry-run", "all"
    )
    result.run = snakemake(result.forked, adopter_configfiles, "all")
    return result


@pytest.mark.pipeline
def test_fork_downloads_shared_prerequisites_but_no_baseline_bundle(
    journey: Journey,
) -> None:
    downloaded = set(journey.downloaded)

    for prerequisite in SHARED_PREREQUISITES:
        assert any(path.startswith(f"{prerequisite}/") for path in downloaded)
    assert not [path for path in downloaded if "trained_methods/" in path]
    assert not [
        path for path in downloaded if path.startswith("eval_results/")
    ]
    manifest = read_release_manifest(
        journey.forked / "extra/release_manifest.json"
    )
    assert manifest.release_id == RELEASE_ID
    assert manifest.execution_mode is ExecutionMode.SMOKE


@pytest.mark.pipeline
def test_adding_a_method_runs_only_its_missing_work(journey: Journey) -> None:
    planned = journey.plan.stdout + journey.plan.stderr
    executed = journey.run.stdout + journey.run.stderr

    assert journey.plan.returncode == 0, planned
    assert job_counts(planned) == ADOPTER_JOBS, planned
    assert journey.run.returncode == 0, executed
    assert job_counts(executed) == ADOPTER_JOBS, executed
    finished = Counter(
        re.findall(r"Finished jobid: \d+ \(Rule: (\w+)\)", executed)
    )
    assert finished == {
        rule: count for rule, count in ADOPTER_JOBS.items() if rule != "total"
    }, executed
    output = journey.forked / "extra/output"
    for method in ["random_dummy", "gdfs"]:
        assert not (output / f"trained_methods/{TAG}/{method}").exists()
        assert not (
            output / f"eval_results/eval_split-test/{TAG}/{method}"
        ).exists()
    assert (output / f"trained_methods/{TAG}/gdfs_adopter").is_dir()


@pytest.mark.pipeline
def test_adding_a_method_reuses_the_downloaded_shared_prerequisites(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    after = files_and_mtimes(journey.forked / "extra/output")

    # Every downloaded file is still there, untouched.
    assert {path: after[path] for path in journey.downloaded} == (
        journey.downloaded
    )
    trained = next(
        (journey.forked / f"extra/output/trained_methods/{TAG}").glob(
            "gdfs_adopter/**/method.bundle"
        )
    )
    contract = json.loads((trained / "manifest.json").read_text())["metadata"][
        "contract"
    ]
    inputs = json.dumps(contract)
    for prerequisite in SHARED_PREREQUISITES:
        if prerequisite.endswith("test.bundle"):
            continue
        assert f"extra/output/{prerequisite}" in inputs, prerequisite


@pytest.mark.pipeline
def test_comparison_holds_each_baseline_row_once_and_keeps_budgets_apart(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    output = journey.forked / "extra/output"
    merged = pd.read_parquet(
        output / f"merged_results/{EVAL_PERF}/{COMPARISON}+all.parquet"
    )

    assert set(merged["afa_method"]) == {
        "random_dummy",
        "gdfs",
        "gdfs_adopter",
    }
    published = journey.published / "extra/output"
    for method in ["random_dummy", "gdfs"]:
        restored = pd.concat(
            pd.read_parquet(path)
            for path in sorted(
                (
                    published / f"eval_results_transformed/eval_split-test/"
                    f"{TAG}/{method}"
                ).rglob("eval_data.parquet")
            )
        )
        assert (merged["afa_method"] == method).sum() == len(restored)
    dummy = merged.loc[merged["afa_method"] == "random_dummy"]
    hard_budget = dummy["eval_hard_budget"].notna()
    soft_budget = dummy["train_soft_budget_param"].notna()
    assert hard_budget.any()
    assert soft_budget.any()
    assert not (hard_budget & soft_budget).any()
    assert set(dummy.loc[hard_budget, "eval_hard_budget"]) == {2.0}
    assert set(dummy.loc[soft_budget, "train_soft_budget_param"]) == {0.3}
    for classifier in ["builtin", "external"]:
        split = pd.read_parquet(
            output / f"merged_results/{EVAL_PERF}/"
            f"{COMPARISON}+classifier_type-{classifier}.parquet"
        )
        # Built-in and external predictions are plotted apart.
        assert len(split) == (merged["classifier"] == classifier).sum()


@pytest.mark.pipeline
def test_comparison_plots_show_both_and_are_labelled_demonstrations(
    journey: Journey,
) -> None:
    assert journey.run.returncode == 0
    plot = (
        journey.forked / f"extra/output/plot_results/{EVAL_PERF}/"
        f"{COMPARISON}+classifier_type-external/all/hard_budget_normal.svg"
    ).read_text()

    # Matplotlib keeps each text of an SVG as a comment beside its glyphs.
    for text in ["Random", "GDFS", "GDFS (adopter)", CAPTION]:
        assert f"<!-- {text} -->" in plot, text


@pytest.mark.pipeline
def test_results_only_journey_needs_no_afabench_loading(
    journey: Journey, tmp_path: Path
) -> None:
    destination = tmp_path / "results"
    downloaded = invoke(
        journey.transport,
        "download",
        RELEASE_ID,
        "--smoke-release",
        "--destination-root",
        str(destination),
        "--payload-category",
        "transformed_evaluation_table",
        "--payload-category",
        "raw_evaluation_table",
        "--output-category",
        "plot_results",
    )

    assert downloaded.exit_code == 0, downloaded.output
    manifest = json.loads((tmp_path / "release_manifest.json").read_text())
    for table in manifest["evaluations"]:
        for key in ["raw_path", "transformed_path"]:
            # The host stores each table as an ordinary file, so a plain
            # download of its URL reads the same as the restored copy.
            hosted = journey.transport.files[
                f"smoke_releases/{RELEASE_ID}/output/{table[key]}"
            ]
            pd.testing.assert_frame_equal(
                pd.read_parquet(destination / table[key]),
                pd.read_parquet(io.BytesIO(hosted)),
            )
    # Plots open in any viewer: plain PDF and SVG files.
    for suffix, header in [(".pdf", b"%PDF"), (".svg", b"<?xml")]:
        plots = sorted((destination / "plot_results").rglob(f"*{suffix}"))
        assert plots, suffix
        for plot in plots:
            assert plot.read_bytes().startswith(header), plot
