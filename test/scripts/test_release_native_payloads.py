"""
Native bundles and raw evaluation tables through the snapshot commands.

The snapshot copies the output root verbatim; these tests pin what the
release manifest says about each payload category, and that the payloads
survive `save` and `restore` unchanged.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from afabench.evaluation.provenance import PROVENANCE_METADATA_KEY
from afabench.release.manifest import read_release_manifest
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport
from test.scripts.release_artifacts import (
    ALPHA,
    BETA,
    HARD_3,
    Catalog,
    Evaluation,
    dataset_bundle,
    evaluation_table,
    method_bundle,
    provenance,
    raw_table,
    write_bundle,
    write_catalog,
    write_job_record,
)

runner = CliRunner()

TRAIN = dataset_bundle("cube", 0, "train")
ALPHA_METHOD = method_bundle(ALPHA, "cube", 0, HARD_3)
ALPHA_RAW_TABLE = f"eval_results/{evaluation_table(ALPHA, 'cube', 0, HARD_3)}"
SMOKE = Catalog(methods=[ALPHA, BETA], smoke_test=True)


def save_smoke(
    tmp_path: Path, source_root: Path, *extra: str
) -> dict[str, Any]:
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump({"smoke_test": True}))
    snapshot_dir = tmp_path / "snapshot"
    result = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(source_root),
            "--configfile",
            str(configfile),
            "--release-id",
            "smoke-native",
            "--scope",
            "smoke",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads((snapshot_dir / "release_manifest.json").read_text())


def tree_size(path: Path) -> int:
    return sum(
        file.stat().st_size for file in path.rglob("*") if file.is_file()
    )


def test_coverage_inventories_each_payload_category(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    write_job_record(
        source_root,
        ALPHA_METHOD.replace("method.bundle", "method.job_record.json"),
        smoke_test=True,
    )

    manifest = save_smoke(tmp_path, source_root)
    payloads = manifest["coverage"]["payloads"]

    def size(pattern: str) -> int:
        return sum(
            tree_size(path) if path.is_dir() else path.stat().st_size
            for path in source_root.glob(pattern)
        )

    def summary(
        category: str, count: int, size_bytes: int, class_names: list[str]
    ) -> dict[str, Any]:
        return {
            "category": category,
            "count": count,
            "size_bytes": size_bytes,
            "class_names": class_names,
        }

    assert payloads == [
        summary(
            "raw_evaluation_table",
            3,
            size("eval_results/**/*.parquet"),
            [],
        ),
        summary(
            "transformed_evaluation_table",
            3,
            size("eval_results_transformed/**/*.parquet"),
            [],
        ),
        summary("dataset_bundle", 3, size("datasets/*/*/*.bundle"), ["Fake"]),
        summary(
            "classifier_bundle",
            2,
            size("trained_classifiers/**/*.bundle"),
            ["Fake"],
        ),
        summary(
            "pretrained_model_bundle",
            1,
            size("pretrained_models/**/*.bundle"),
            ["Fake"],
        ),
        summary(
            "afa_method_bundle",
            3,
            size("trained_methods/**/*.bundle"),
            ["Fake"],
        ),
        summary(
            "job_duration_table",
            1,
            (tmp_path / "snapshot/release_job_duration_table.parquet")
            .stat()
            .st_size,
            [],
        ),
    ]
    # Marked smoke, as the compute estimate refuses smoke durations.
    assert manifest["job_duration_table"]["smoke_test"] is True


def restore(snapshot_dir: Path, destination_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(destination_root),
        ],
    )
    assert result.exit_code == 0, result.output


def test_raw_tables_round_trip_values_nulls_schema_and_histories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, Catalog(methods=[ALPHA], smoke_test=True))
    raw_path = source_root / ALPHA_RAW_TABLE
    record = pq.read_schema(raw_path).metadata[PROVENANCE_METADATA_KEY]
    table = pa.Table.from_pandas(
        raw_table(
            Evaluation(
                method="alpha",
                dataset="cube",
                realization=0,
                train_hard_budget=3,
                train_soft_budget_param=None,
                eval_hard_budget=3,
                eval_soft_budget_param=None,
            )
        ),
        preserve_index=False,
    )
    # A null soft-budget parameter and a NaN cost must stay distinct, and
    # the provenance record in the schema metadata must survive.
    table = table.set_column(
        table.schema.get_field_index("accumulated_cost"),
        "accumulated_cost",
        pa.array([1.0, float("nan")]),
    ).replace_schema_metadata(
        {**table.schema.metadata, PROVENANCE_METADATA_KEY: record}
    )
    pq.write_table(table, raw_path)
    save_smoke(tmp_path, source_root)

    destination_root = tmp_path / "checkout/extra/output"
    restore(tmp_path / "snapshot", destination_root)

    restored = pq.read_table(destination_root / ALPHA_RAW_TABLE)
    assert restored.schema.equals(table.schema, check_metadata=True)
    # NaN never compares equal, so the cost column is checked below.
    assert restored.drop_columns(["accumulated_cost"]).equals(
        table.drop_columns(["accumulated_cost"])
    )
    assert restored.schema.metadata[PROVENANCE_METADATA_KEY] == record
    assert restored.column("eval_soft_budget_param").null_count == 2
    assert restored.column("builtin_predicted_class").null_count == 2
    costs = restored.column("accumulated_cost").to_pylist()
    assert costs[0] == 1.0
    assert math.isnan(costs[1])
    # Selection histories, read with plain pandas: action 0 is stop and
    # action i > 0 is selection i - 1.
    read = pd.read_parquet(destination_root / ALPHA_RAW_TABLE)
    histories = {
        int(episode): [
            int(action) - 1
            for action in steps.sort_values("step")["action_performed"]
            if action != 0
        ]
        for episode, steps in read.groupby("episode_id")
    }
    assert histories == {0: [2]}


UNREVIEWED = {
    "status": "unreviewed",
    "license": None,
    "source": None,
    "reviewed_by": None,
    "notes": None,
}


def test_datasets_without_a_review_are_recorded_and_reported_unreviewed(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump({"smoke_test": True}))

    result = runner.invoke(
        app,
        [
            "save",
            str(tmp_path / "snapshot"),
            "--source-root",
            str(source_root),
            "--configfile",
            str(configfile),
            "--release-id",
            "smoke-native",
            "--scope",
            "smoke",
            "--checkout",
            str(checkout),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (tmp_path / "snapshot/release_manifest.json").read_text()
    )
    assert manifest["dataset_redistribution"] == {"cube": UNREVIEWED}
    assert "Unreviewed dataset redistribution: cube" in result.output


def test_reviewed_datasets_record_the_review_from_the_checkout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    checkout = tmp_path / "checkout"
    review = {
        "status": "permitted",
        "license": "MIT",
        "source": "generated by afabench",
        "reviewed_by": "maintainer",
        "notes": None,
    }
    review_file = checkout / "extra/conf/release/dataset_redistribution.yaml"
    review_file.parent.mkdir(parents=True)
    review_file.write_text(yaml.safe_dump({"datasets": {"cube": review}}))

    manifest = save_smoke(tmp_path, source_root, "--checkout", str(checkout))

    assert manifest["dataset_redistribution"] == {"cube": review}


def test_the_checked_in_review_grants_no_dataset_redistribution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    repository = Path(__file__).parents[2]

    manifest = save_smoke(tmp_path, source_root, "--checkout", str(repository))

    assert manifest["dataset_redistribution"] == {"cube": UNREVIEWED}


def test_every_dataset_a_record_names_is_reviewed(tmp_path: Path) -> None:
    # Evaluation tables hold true class labels, so a dataset with tables
    # but no bundles in the release still needs a review.
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    write_catalog(
        source_root,
        Catalog(methods=[ALPHA], datasets=["mnist"], smoke_test=True),
    )
    for bundle in list(source_root.rglob("*.bundle")):
        if "mnist" in bundle.as_posix():
            shutil.rmtree(bundle)

    manifest = save_smoke(tmp_path, source_root)

    assert set(manifest["dataset_redistribution"]) == {"cube", "mnist"}


def test_inventory_reports_payload_categories_without_copying(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_bundle(
        source_root,
        TRAIN,
        provenance("dataset_generation", smoke_test=True),
        class_name="CubeDataset",
    )
    legacy = dataset_bundle("cube", 0, "val")
    write_bundle(source_root, legacy, None)
    before = sorted(tmp_path.rglob("*"))

    result = runner.invoke(
        app, ["inventory", "--source-root", str(source_root)]
    )

    assert result.exit_code == 0, result.output
    assert sorted(tmp_path.rglob("*")) == before
    lines = result.output.splitlines()
    assert "Execution: smoke" in lines
    assert (
        f"dataset_bundle: 1 present, {tree_size(source_root / TRAIN)} bytes, "
        "classes: CubeDataset"
    ) in lines
    assert "afa_method_bundle: 0 present, 0 bytes, classes: none" in lines
    assert "raw_evaluation_table: 0 present, 0 bytes, classes: none" in lines
    assert "Without a provenance record:" in lines
    assert f"  {legacy}" in lines


def test_release_transport_carries_bundles_and_redistribution_warnings(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_catalog(source_root, SMOKE)
    save_smoke(tmp_path, source_root)
    transport = FakeReleaseTransport()
    destination_root = tmp_path / "checkout/extra/output"

    published = runner.invoke(
        app,
        ["publish", str(tmp_path / "snapshot"), "--repo-id", "fake/repo"]
        + ["--smoke-release"],
        obj=lambda _repo_id: transport,
    )
    downloaded = runner.invoke(
        app,
        ["download", "smoke-native", "--repo-id", "fake/repo"]
        + ["--destination-root", str(destination_root), "--smoke-release"]
        + ["--all"],
        obj=lambda _repo_id: transport,
    )

    assert published.exit_code == 0, published.output
    assert downloaded.exit_code == 0, downloaded.output
    for result in [published, downloaded]:
        assert "Unreviewed dataset redistribution: cube" in result.output
    for path in [TRAIN, ALPHA_METHOD]:
        for file in (source_root / path).rglob("*"):
            restored = destination_root / file.relative_to(source_root)
            assert restored.is_dir() == file.is_dir()
            if file.is_file():
                assert restored.read_bytes() == file.read_bytes()
    manifest = read_release_manifest(
        tmp_path / "checkout/extra/release_manifest.json"
    )
    assert {TRAIN, ALPHA_METHOD} <= {
        bundle.path for bundle in manifest.bundles
    }
