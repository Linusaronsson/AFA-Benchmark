from pathlib import Path

from afabench.common.naming import (
    canonicalize_dataset_key,
    get_legacy_dataset_key,
    infer_dataset_key_from_class_name,
    resolve_existing_dataset_path,
)


def test_legacy_dataset_keys_canonicalize_to_cube_nm_variants() -> None:
    assert canonicalize_dataset_key("afa_context_v2") == "cube_nm"
    assert (
        canonicalize_dataset_key("afa_context_v2_without_noise")
        == "cube_nm_without_noise"
    )
    assert canonicalize_dataset_key("afa_context") == "cube_nm_3ctx"
    assert (
        canonicalize_dataset_key("afa_context_without_noise")
        == "cube_nm_3ctx_without_noise"
    )


def test_infer_dataset_key_handles_xor_dataset_name() -> None:
    assert (
        infer_dataset_key_from_class_name("XORNoisyShortcutDataset")
        == "xor_noisy_shortcut"
    )


def test_get_legacy_dataset_key_for_canonical_cube_nm() -> None:
    assert get_legacy_dataset_key("cube_nm") == "afa_context_v2"


def test_resolve_existing_dataset_path_uses_legacy_path(
    tmp_path: Path,
) -> None:
    canonical_path = (
        tmp_path / "dataset-cube_nm+instance_idx-0" / "eval_data.csv"
    )
    legacy_path = (
        tmp_path / "dataset-afa_context_v2+instance_idx-0" / "eval_data.csv"
    )
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("value\n1\n")

    assert resolve_existing_dataset_path(canonical_path, "cube_nm") == str(
        legacy_path
    )


def test_resolve_existing_dataset_path_prefers_canonical_path(
    tmp_path: Path,
) -> None:
    canonical_path = (
        tmp_path / "dataset-cube_nm+instance_idx-0" / "eval_data.csv"
    )
    legacy_path = (
        tmp_path / "dataset-afa_context_v2+instance_idx-0" / "eval_data.csv"
    )
    canonical_path.parent.mkdir(parents=True)
    canonical_path.write_text("canonical\n")
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text("legacy\n")

    assert resolve_existing_dataset_path(canonical_path, "cube_nm") == str(
        canonical_path
    )
