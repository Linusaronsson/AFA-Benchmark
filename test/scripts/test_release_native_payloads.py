"""
Native bundles and raw evaluation tables through the snapshot commands.

The snapshot copies the output root verbatim; these tests pin what the
release manifest says about each payload, and that the payloads survive
`save` and `restore` unchanged.
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from scripts.release.snapshot import app

runner = CliRunner()

TAG = "initializer-cold"
TRAIN = "datasets/cube/0/train.bundle"
VAL = "datasets/cube/0/val.bundle"
TEST = "datasets/cube/0/test.bundle"
EXTERNAL_CLASSIFIER = f"trained_classifiers/{TAG}/dataset-cube.bundle"
BETA_CLASSIFIER = f"trained_classifiers/{TAG}/method-beta+dataset-cube.bundle"
PRETRAINED_MODEL = (
    f"pretrained_models/{TAG}/shared/dataset-cube+instance_idx-0/"
    "pretrain_seed-0/model.bundle"
)
ALPHA_METHOD = (
    f"trained_methods/{TAG}/alpha/dataset-cube+instance_idx-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
    "method.bundle"
)
BETA_METHOD = (
    f"trained_methods/{TAG}/beta/dataset-cube+instance_idx-0/"
    "pretrain_seed-0/"
    "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
    "method.bundle"
)


def workflow_config() -> dict[str, Any]:
    """Alpha has no pretraining stage; beta has one and its own classifier."""
    return {
        "pretrain_mapping": {"shared": {"pretrain_script_name": "shared"}},
        "method_options": {
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 4},
            "beta": {
                "train_script_name": "beta",
                "eval_batch_size": 4,
                "pretrained_model_name": "shared",
                "classifier": {"script_name": "special"},
            },
        },
        "methods": ["alpha", "beta"],
        "datasets": ["cube"],
        "dataset_instance_indices": [0],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [3]},
        "soft_budget_params": {
            "alpha": {"default": []},
            "beta": {"default": []},
        },
        "classifier_names": {"default": "masked_mlp_classifier"},
        "use_wandb": False,
        "smoke_test": True,
    }


def write_bundle(root: Path, path: str, class_name: str) -> int:
    """Lay out a bundle as `save_bundle` does; return its size in bytes."""
    bundle = root / path
    (bundle / "data").mkdir(parents=True)
    payload = b"\x00\x01\x02\x03"
    (bundle / "data/weights.bin").write_bytes(payload)
    manifest = json.dumps(
        {
            "bundle_version": "1.0.0",
            "class_name": class_name,
            "class_version": None,
            "metadata": {"seed": 0},
        }
    )
    (bundle / "manifest.json").write_text(manifest)
    return len(payload) + len(manifest)


def raw_episode_table() -> pd.DataFrame:
    """Two episodes as the evaluator writes them, external predictions only."""
    return pd.DataFrame(
        {
            "episode_id": [0, 0, 0, 1],
            "step": [0, 1, 2, 0],
            "action_performed": [3, 1, 0, 0],
            "builtin_predicted_class": pd.array(
                [None, None, None, None], dtype="Int64"
            ),
            "external_predicted_class": [2, 2, 1, 0],
            "true_class": [1, 1, 1, 0],
            "accumulated_cost": [1.0, 2.0, 2.0, 0.0],
            "forced_stop": [False, False, False, False],
            "eval_seed": [0, 0, 0, 0],
            "eval_hard_budget": [3.0, 3.0, 3.0, 3.0],
        }
    )


def save_test_only(
    tmp_path: Path, source_root: Path, *extra: str
) -> dict[str, Any]:
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump(workflow_config()))
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
            "test_only",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads((snapshot_dir / "release_manifest.json").read_text())


def test_manifest_lists_every_configured_bundle_with_category_and_inputs(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_bundle(source_root, TRAIN, "CubeDataset")

    bundles = {
        record["path"]: record
        for record in save_test_only(tmp_path, source_root)["bundles"]
    }

    assert set(bundles) == {
        TRAIN,
        VAL,
        TEST,
        EXTERNAL_CLASSIFIER,
        BETA_CLASSIFIER,
        PRETRAINED_MODEL,
        ALPHA_METHOD,
        BETA_METHOD,
    }
    assert {path: record["category"] for path, record in bundles.items()} == {
        TRAIN: "dataset_bundle",
        VAL: "dataset_bundle",
        TEST: "dataset_bundle",
        EXTERNAL_CLASSIFIER: "classifier_bundle",
        BETA_CLASSIFIER: "classifier_bundle",
        PRETRAINED_MODEL: "pretrained_model_bundle",
        ALPHA_METHOD: "afa_method_bundle",
        BETA_METHOD: "afa_method_bundle",
    }
    assert {path: record["inputs"] for path, record in bundles.items()} == {
        TRAIN: [],
        VAL: [],
        TEST: [],
        EXTERNAL_CLASSIFIER: [
            {"role": "train_dataset", "path": TRAIN},
            {"role": "val_dataset", "path": VAL},
        ],
        BETA_CLASSIFIER: [
            {"role": "train_dataset", "path": TRAIN},
            {"role": "val_dataset", "path": VAL},
        ],
        PRETRAINED_MODEL: [
            {"role": "train_dataset", "path": TRAIN},
            {"role": "val_dataset", "path": VAL},
            {"role": "classifier", "path": EXTERNAL_CLASSIFIER},
        ],
        ALPHA_METHOD: [
            {"role": "train_dataset", "path": TRAIN},
            {"role": "val_dataset", "path": VAL},
            {"role": "classifier", "path": EXTERNAL_CLASSIFIER},
        ],
        BETA_METHOD: [
            {"role": "train_dataset", "path": TRAIN},
            {"role": "val_dataset", "path": VAL},
            {"role": "classifier", "path": BETA_CLASSIFIER},
            {"role": "pretrained_model", "path": PRETRAINED_MODEL},
        ],
    }
    assert bundles[TRAIN]["present"] is True
    assert bundles[VAL]["present"] is False


IDENTITY_FIELDS = [
    "dataset_key",
    "dataset_instance_index",
    "split",
    "method_name",
    "pretrained_model_name",
    "seed",
    "train_hard_budget",
    "train_soft_budget_param",
]


def test_bundle_records_carry_identity_seeds_and_budgets(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_bundle(source_root, TRAIN, "CubeDataset")

    bundles = {
        record["path"]: {field: record[field] for field in IDENTITY_FIELDS}
        for record in save_test_only(tmp_path, source_root)["bundles"]
    }

    shared = {"method_name": None, "pretrained_model_name": None}
    untrained = {"train_hard_budget": None, "train_soft_budget_param": None}
    instance_0 = {"dataset_key": "cube", "dataset_instance_index": 0}
    assert bundles[TEST] == {
        **instance_0,
        **shared,
        **untrained,
        "split": "test",
        "seed": 0,
    }
    assert bundles[EXTERNAL_CLASSIFIER] == {
        **instance_0,
        **shared,
        **untrained,
        "split": None,
        "seed": 0,
    }
    assert bundles[BETA_CLASSIFIER]["method_name"] == "beta"
    assert bundles[PRETRAINED_MODEL] == {
        **instance_0,
        **untrained,
        "split": None,
        "method_name": None,
        "pretrained_model_name": "shared",
        "seed": 0,
    }
    assert bundles[ALPHA_METHOD] == {
        **instance_0,
        "split": None,
        "method_name": "alpha",
        "pretrained_model_name": None,
        "seed": 0,
        "train_hard_budget": 3,
        "train_soft_budget_param": None,
    }
    assert bundles[BETA_METHOD]["pretrained_model_name"] == "shared"


def test_present_bundles_record_size_and_their_own_manifest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    size_bytes = write_bundle(
        source_root, PRETRAINED_MODEL, "GreedyAFAClassifier"
    )

    bundles = {
        record["path"]: record
        for record in save_test_only(tmp_path, source_root)["bundles"]
    }

    assert bundles[PRETRAINED_MODEL]["size_bytes"] == size_bytes
    assert bundles[PRETRAINED_MODEL]["bundle_manifest"] == {
        "bundle_version": "1.0.0",
        "class_name": "GreedyAFAClassifier",
        "class_version": None,
        "metadata": {"seed": 0},
    }
    assert bundles[ALPHA_METHOD]["size_bytes"] is None
    assert bundles[ALPHA_METHOD]["bundle_manifest"] is None


ALPHA_RAW_TABLE = (
    f"eval_results/eval_split-test/{TAG}/alpha/dataset-cube+instance_idx-0/"
    "NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
    "eval_seed-0+eval_hard_budget-3+eval_soft_budget_param-null/"
    "eval_data.parquet"
)


def test_evaluation_tables_record_the_bundles_they_were_evaluated_from(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    raw_table = source_root / ALPHA_RAW_TABLE
    raw_table.parent.mkdir(parents=True)
    raw_episode_table().to_parquet(raw_table, index=False)

    tables = {
        record["raw_path"]: record
        for record in save_test_only(tmp_path, source_root)[
            "evaluation_tables"
        ]
    }

    alpha = tables[ALPHA_RAW_TABLE]
    assert alpha["inputs"] == [
        {"role": "eval_dataset", "path": TEST},
        {"role": "method", "path": ALPHA_METHOD},
        {"role": "classifier", "path": EXTERNAL_CLASSIFIER},
    ]
    assert alpha["raw_size_bytes"] == raw_table.stat().st_size
    assert alpha["transformed_size_bytes"] is None


def test_coverage_inventories_each_payload_category(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    dataset_bytes = write_bundle(source_root, TRAIN, "CubeDataset")
    dataset_bytes += write_bundle(source_root, TEST, "CubeDataset")
    model_bytes = write_bundle(
        source_root, PRETRAINED_MODEL, "GreedyAFAClassifier"
    )
    raw_table = source_root / ALPHA_RAW_TABLE
    raw_table.parent.mkdir(parents=True)
    raw_episode_table().to_parquet(raw_table, index=False)

    payloads = save_test_only(tmp_path, source_root)["coverage"]["payloads"]

    def summary(
        category: str,
        scheduled: int,
        present: int,
        size_bytes: int,
        class_names: list[str],
    ) -> dict[str, Any]:
        return {
            "category": category,
            "scheduled": scheduled,
            "present": present,
            "size_bytes": size_bytes,
            "class_names": class_names,
        }

    assert payloads == [
        summary("raw_evaluation_table", 2, 1, raw_table.stat().st_size, []),
        summary("transformed_evaluation_table", 2, 0, 0, []),
        summary("dataset_bundle", 3, 2, dataset_bytes, ["CubeDataset"]),
        summary("classifier_bundle", 2, 0, 0, []),
        summary(
            "pretrained_model_bundle",
            1,
            1,
            model_bytes,
            ["GreedyAFAClassifier"],
        ),
        summary("afa_method_bundle", 2, 0, 0, []),
    ]


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
    raw_table = source_root / ALPHA_RAW_TABLE
    raw_table.parent.mkdir(parents=True)
    table = pa.Table.from_pandas(raw_episode_table(), preserve_index=False)
    # A null soft-budget parameter and a NaN cost must stay distinct, and
    # schema metadata (ADR 0002 stores provenance there) must survive.
    table = table.append_column(
        "eval_soft_budget_param", pa.array([None] * 4, type=pa.float64())
    ).set_column(
        table.schema.get_field_index("accumulated_cost"),
        "accumulated_cost",
        pa.array([1.0, 2.0, float("nan"), 0.0]),
    )
    table = table.replace_schema_metadata(
        {**table.schema.metadata, b"afabench.provenance": b'{"seed": 0}'}
    )
    pq.write_table(table, raw_table)
    save_test_only(tmp_path, source_root)

    destination_root = tmp_path / "checkout/extra/output"
    restore(tmp_path / "snapshot", destination_root)

    restored = pq.read_table(destination_root / ALPHA_RAW_TABLE)
    assert restored.schema.equals(table.schema, check_metadata=True)
    # NaN never compares equal, so the cost column is checked below.
    assert restored.drop_columns(["accumulated_cost"]).equals(
        table.drop_columns(["accumulated_cost"])
    )
    assert restored.schema.metadata[b"afabench.provenance"] == b'{"seed": 0}'
    assert restored.column("eval_soft_budget_param").null_count == 4
    assert restored.column("builtin_predicted_class").null_count == 4
    costs = restored.column("accumulated_cost").to_pylist()
    assert costs[:2] == [1.0, 2.0]
    assert math.isnan(costs[2])
    # Selection histories, read with plain pandas: action 0 is stop and
    # action i > 0 is selection i - 1.
    frame = pd.read_parquet(destination_root / ALPHA_RAW_TABLE)
    histories = {
        int(episode): [
            int(action) - 1
            for action in steps.sort_values("step")["action_performed"]
            if action != 0
        ]
        for episode, steps in frame.groupby("episode_id")
    }
    assert histories == {0: [2, 0], 1: []}


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
    write_bundle(source_root, TRAIN, "CubeDataset")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump(workflow_config()))

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
            "test_only",
            "--checkout",
            str(checkout),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads(
        (tmp_path / "snapshot/release_manifest.json").read_text()
    )
    assert manifest["settings"]["dataset_redistribution"] == {
        "cube": UNREVIEWED
    }
    assert "Unreviewed dataset redistribution: cube" in result.output


def test_reviewed_datasets_record_the_review_from_the_checkout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_bundle(source_root, TRAIN, "CubeDataset")
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

    manifest = save_test_only(
        tmp_path, source_root, "--checkout", str(checkout)
    )

    assert manifest["settings"]["dataset_redistribution"] == {"cube": review}


def test_the_checked_in_review_grants_no_dataset_redistribution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    write_bundle(source_root, TRAIN, "CubeDataset")
    repository = Path(__file__).parents[2]

    manifest = save_test_only(
        tmp_path, source_root, "--checkout", str(repository)
    )

    assert manifest["settings"]["dataset_redistribution"] == {
        "cube": UNREVIEWED
    }


def test_inventory_reports_payload_categories_without_copying(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    dataset_bytes = write_bundle(source_root, TRAIN, "CubeDataset")
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump(workflow_config()))
    before = sorted(tmp_path.rglob("*"))

    result = runner.invoke(
        app,
        [
            "inventory",
            "--source-root",
            str(source_root),
            "--configfile",
            str(configfile),
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(tmp_path.rglob("*")) == before
    lines = result.output.splitlines()
    assert "Execution: smoke" in lines
    assert (
        f"dataset_bundle: 1/3 present, {dataset_bytes} bytes, "
        "classes: CubeDataset"
    ) in lines
    assert "afa_method_bundle: 0/2 present, 0 bytes, classes: none" in lines
    assert "raw_evaluation_table: 0/2 present, 0 bytes, classes: none" in lines
