from afabench.common.naming import (
    canonicalize_dataset_key,
    infer_dataset_key_from_class_name,
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
