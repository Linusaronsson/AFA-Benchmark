from afabench.common.config_classes import InitializerConfig
from afabench.common.custom_types import AFAInitializer
from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)
from afabench.common.initializers.cube_nm_ar_mar_initializer import (
    CubeNMARMARInitializer,
)
from afabench.common.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.common.initializers.least_informative_initializer import (
    LeastInformativeInitializer,
)
from afabench.common.initializers.manual_initializer import ManualInitializer
from afabench.common.initializers.missingness_initializer import (
    MissingnessInitializer,
)
from afabench.common.initializers.mutual_information_initializer import (
    MutualInformationInitializer,
)
from afabench.common.initializers.random_initializer import (
    RandomInitializer,
)
from afabench.common.initializers.xor_noisy_shortcut_initializer import (
    XORNoisyShortcutInitializer,
)
from afabench.common.initializers.zero_initializer import ZeroInitializer
from afabench.common.registry import get_class


def get_afa_initializer_from_config(  # noqa: C901, PLR0911
    initializer_config: InitializerConfig,
) -> AFAInitializer:
    """Get initializer from config."""
    if initializer_config.class_name == "ZeroInitializer":
        assert not initializer_config.kwargs

        cls = get_class(initializer_config.class_name)
        assert cls is ZeroInitializer
        return cls()

    if initializer_config.class_name == "FixedRandomInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is FixedRandomInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "ManualInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is ManualInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "MutualInformationInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is MutualInformationInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "LeastInformativeInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is LeastInformativeInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "RandomInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is RandomInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "MissingnessInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is MissingnessInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "CubeNMARInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is CubeNMARInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "CubeNMARMARInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is CubeNMARMARInitializer
        return cls(**initializer_config.kwargs)

    if initializer_config.class_name == "XORNoisyShortcutInitializer":
        cls = get_class(initializer_config.class_name)
        assert cls is XORNoisyShortcutInitializer
        return cls(**initializer_config.kwargs)

    msg = f"Unknown initializer: {initializer_config.class_name}"
    raise ValueError(msg)
