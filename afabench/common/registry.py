from afabench.common.afa_initializers.base import AFAInitializer
from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
    AFAUnmaskFn,
)

AFA_METHOD_TYPES = {
    "shim2018",
    "zannone2019",
    "kachuee2019",
    "covert2023",
    "gadgil2023",
    "ma2018",
    "aaco",
    "cae",
    "permutation",
    "sequential_dummy",
    "random_dummy",
    "optimalcube",
}


def get_afa_method_class(name: str) -> type[AFAMethod]:  # noqa: PLR0911
    """
    Return the appropriate AFAMethod for a given method type.

    Note that several method types can have the same AFAMethod class, like the RL methods. A dictionary is not used since it could lead to circular imports.
    """
    if name == "RLAFAMethod":
        from afabench.afa_rl.afa_methods import RLAFAMethod

        return RLAFAMethod
    if name == "Covert2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (
            Covert2023AFAMethod,
        )

        return Covert2023AFAMethod
    if name == "Gadgil2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (
            Gadgil2023AFAMethod,
        )

        return Gadgil2023AFAMethod
    if name == "Ma2018AFAMethod":
        from afabench.afa_generative.afa_methods import (
            Ma2018AFAMethod,
        )

        return Ma2018AFAMethod

    if name == "AACOAFAMethod":
        from afabench.afa_oracle.afa_methods import (
            AACOAFAMethod,
        )

        return AACOAFAMethod

    if name == "StaticBaseMethod":
        from afabench.static.static_methods import (
            StaticBaseMethod,
        )

        return StaticBaseMethod
    if name == "SequentialDummyAFAMethod":
        from afabench.common.afa_methods import (
            SequentialDummyAFAMethod,
        )

        return SequentialDummyAFAMethod
    if name == "RandomDummyAFAMethod":
        from afabench.common.afa_methods import (
            RandomDummyAFAMethod,
        )

        return RandomDummyAFAMethod
    if name == "OptimalCubeAFAMethod":
        from afabench.common.afa_methods import (
            OptimalCubeAFAMethod,
        )

        return OptimalCubeAFAMethod
    msg = f"Unknown AFA method: {name}"
    raise ValueError(msg)


AFA_DATASET_TYPES = {
    "cube",
    "cubeSimple",
    "cubeOnlyInformative",
    "shim2018cube",
    "AFAContext",
    "AFAContextRandomInsert",
    "ContextSelectiveXOR",
    "MNIST",
    "diabetes",
    "physionet",
    "miniboone",
    "FashionMNIST",
    "bank_marketing",
    "ckd",
    "actg",
}


def get_afa_dataset_class(name: str) -> type[AFADataset]:  # noqa: C901, PLR0911
    if name == "cube":
        from afabench.common.datasets.datasets import (
            CubeDataset,
        )

        return CubeDataset
    if name == "AFAContext":
        from afabench.common.datasets.datasets import (
            AFAContextDataset,
        )

        return AFAContextDataset
    if name == "MNIST":
        from afabench.common.datasets.datasets import (
            MNISTDataset,
        )

        return MNISTDataset
    if name == "diabetes":
        from afabench.common.datasets.datasets import (
            DiabetesDataset,
        )

        return DiabetesDataset
    if name == "physionet":
        from afabench.common.datasets.datasets import (
            PhysionetDataset,
        )

        return PhysionetDataset
    if name == "miniboone":
        from afabench.common.datasets.datasets import (
            MiniBooNEDataset,
        )

        return MiniBooNEDataset
    if name == "FashionMNIST":
        from afabench.common.datasets.datasets import (
            FashionMNISTDataset,
        )

        return FashionMNISTDataset
    if name == "bank_marketing":
        from afabench.common.datasets.datasets import BankMarketingDataset

        return BankMarketingDataset

    if name == "ckd":
        from afabench.common.datasets.datasets import CKDDataset

        return CKDDataset

    if name == "actg":
        from afabench.common.datasets.datasets import ACTG175Dataset

        return ACTG175Dataset

    if name == "imagenette":
        from afabench.common.datasets.datasets import ImagenetteDataset

        return ImagenetteDataset

    msg = f"Unknown AFA dataset: {name}"
    raise ValueError(msg)


AFA_CLASSIFIER_TYPES = {
    "randomdummy",
    "uniformdummy",
    "WrappedMaskedMLPClassifier",
    "WrappedMaskedVitClassifier",
    "Shim2018AFAClassifier",
}


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:  # noqa: PLR0911
    if name == "randomdummy":
        from afabench.common.classifiers import (
            RandomDummyAFAClassifier,
        )

        return RandomDummyAFAClassifier
    if name == "uniformdummy":
        from afabench.common.classifiers import (
            UniformDummyAFAClassifier,
        )

        return UniformDummyAFAClassifier
    if name == "WrappedMaskedMLPClassifier":
        from afabench.common.classifiers import (
            WrappedMaskedMLPClassifier,
        )

        return WrappedMaskedMLPClassifier

    if name == "WrappedMaskedViTClassifier":
        from afabench.common.classifiers import (
            WrappedMaskedViTClassifier,
        )

        return WrappedMaskedViTClassifier
    if name == "Shim2018AFAClassifier":
        from afabench.afa_rl.shim2018.models import (
            Shim2018AFAClassifier,
        )

        return Shim2018AFAClassifier
    if name == "Zannone2019AFAClassifier":
        from afabench.afa_rl.zannone2019.models import (
            Zannone2019AFAClassifier,
        )

        return Zannone2019AFAClassifier
    if name == "Kachuee2019AFAClassifier":
        from afabench.afa_rl.kachuee2019.models import (
            Kachuee2019AFAClassifier,
        )

        return Kachuee2019AFAClassifier
    msg = f"Unknown AFA classifier: {name}"
    raise ValueError(msg)


AFA_UNMASKER_TYPES = {"one_based_index", "image_patch"}


def get_afa_unmasker(name: str, **kwargs) -> AFAUnmaskFn:
    """
    Get unmasker function by name.

    Args:
        name: "one_based_index" or "image_patch"
        **kwargs: For image_patch: image_side_length, n_channels, patch_size
    """
    if name == "one_based_index":
        from afabench.common.afa_unmaskers import one_based_index_unmask_fn

        return one_based_index_unmask_fn

    if name == "image_patch":
        from afabench.common.afa_unmaskers import get_image_patch_unmask_fn

        return get_image_patch_unmask_fn(
            image_side_length=kwargs["image_side_length"],
            n_channels=kwargs["n_channels"],
            patch_size=kwargs["patch_size"],
        )

    raise ValueError(f"Unknown unmasker: {name}")


AFA_INITIALIZER_TYPES = {
    "zero",
    "fixed_random",
    "random_per_episode",
    "manual",
    "mutual_information",
    "least_informative",
    "aaco_default",
}


def get_afa_initializer(name: str, **kwargs) -> AFAInitializer:
    """
    Get initializer by name.

    Args:
        name: Initializer type
        **kwargs: seed (optional), feature_indices (for manual)
    """
    seed = kwargs.get("seed")

    if name == "zero":
        from afabench.common.afa_initializers import ZeroInitializer

        return ZeroInitializer(seed=seed)

    if name == "fixed_random":
        from afabench.common.afa_initializers import FixedRandomStrategy

        return FixedRandomStrategy(seed=seed)

    if name == "random_per_episode":
        from afabench.common.afa_initializers import RandomPerEpisodeStrategy

        return RandomPerEpisodeStrategy(seed=seed)

    if name == "manual":
        from afabench.common.afa_initializers import ManualStrategy

        return ManualStrategy(
            feature_indices=kwargs["feature_indices"], seed=seed
        )

    if name == "mutual_information":
        from afabench.common.afa_initializers import MutualInformationStrategy

        return MutualInformationStrategy(seed=seed)

    if name == "least_informative":
        from afabench.common.afa_initializers import LeastInformativeStrategy

        return LeastInformativeStrategy(seed=seed)

    if name == "aaco_default":
        from afabench.common.afa_initializers.strategies import (
            AACODefaultInitializer,
        )

        return AACODefaultInitializer(
            dataset_name=kwargs["dataset_artifact_path"],
            seed=kwargs.get("seed"),
        )

    raise ValueError(f"Unknown initializer: {name}")
