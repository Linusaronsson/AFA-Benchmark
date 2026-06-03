from afabench.components.initializers.fixed_random_initializer import (
    FixedRandomInitializer,
)
from afabench.components.initializers.least_informative_initializer import (
    LeastInformativeInitializer,
)
from afabench.components.initializers.manual_initializer import (
    ManualInitializer,
)
from afabench.components.initializers.mutual_information_initializer import (
    MutualInformationInitializer,
)
from afabench.components.initializers.random_initializer import (
    RandomInitializer,
)
from afabench.components.initializers.zero_initializer import (
    ZeroInitializer,
)
from afabench.core.config_classes import InitializerConfig
from afabench.core.registry import get_class
from afabench.core.types import AFAInitializer


def get_afa_initializer_from_config(
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

    msg = f"Unknown initializer: {initializer_config.class_name}"
    raise ValueError(msg)
