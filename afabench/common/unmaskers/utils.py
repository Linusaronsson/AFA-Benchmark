from afabench.common.config_classes import UnmaskerConfig
from afabench.common.custom_types import AFAUnmasker
from afabench.common.registry import get_class
from afabench.common.unmaskers.afa_context_unmasker import AFAContextUnmasker
from afabench.common.unmaskers.cube_nm_ar_unmasker import CubeNMARUnmasker
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker
from afabench.common.unmaskers.image_patch_unmasker import ImagePatchUnmasker


def get_afa_unmasker_from_config(
    unmasker_config: UnmaskerConfig,
) -> AFAUnmasker:
    """Get unmasker function by config."""
    if unmasker_config.class_name == "DirectUnmasker":
        assert not unmasker_config.kwargs

        cls = get_class(unmasker_config.class_name)
        assert cls is DirectUnmasker
        return cls()

    if unmasker_config.class_name == "ImagePatchUnmasker":
        cls = get_class(unmasker_config.class_name)
        assert cls is ImagePatchUnmasker
        return cls(**unmasker_config.kwargs)

    if unmasker_config.class_name == "AFAContextUnmasker":
        cls = get_class(unmasker_config.class_name)
        assert cls is AFAContextUnmasker
        return cls(**unmasker_config.kwargs)

    if unmasker_config.class_name == "CubeNMARUnmasker":
        cls = get_class(unmasker_config.class_name)
        assert cls is CubeNMARUnmasker
        return cls(**unmasker_config.kwargs)

    msg = f"Unknown unmasker: {unmasker_config.class_name}"
    raise ValueError(msg)
