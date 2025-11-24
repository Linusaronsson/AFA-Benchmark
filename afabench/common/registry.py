from afabench.common.custom_types import (
    AFAClassifier,
    AFADataset,
    AFAMethod,
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
        from afabench.afa_rl.afa_methods import RLAFAMethod  # noqa: PLC0415

        return RLAFAMethod
    if name == "Covert2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (  # noqa: PLC0415
            Covert2023AFAMethod,
        )

        return Covert2023AFAMethod
    if name == "Gadgil2023AFAMethod":
        from afabench.afa_discriminative.afa_methods import (  # noqa: PLC0415
            Gadgil2023AFAMethod,
        )

        return Gadgil2023AFAMethod
    if name == "Ma2018AFAMethod":
        from afabench.afa_generative.afa_methods import Ma2018AFAMethod  # noqa: PLC0415

        return Ma2018AFAMethod

    if name == "AACOAFAMethod":
        from afabench.afa_oracle.afa_methods import AACOAFAMethod  # noqa: PLC0415

        return AACOAFAMethod

    if name == "StaticBaseMethod":
        from afabench.static.static_methods import StaticBaseMethod  # noqa: PLC0415

        return StaticBaseMethod
    if name == "SequentialDummyAFAMethod":
        from afabench.common.afa_methods import (  # noqa: PLC0415
            SequentialDummyAFAMethod,
        )

        return SequentialDummyAFAMethod
    if name == "RandomDummyAFAMethod":
        from afabench.common.afa_methods import RandomDummyAFAMethod  # noqa: PLC0415

        return RandomDummyAFAMethod
    if name == "OptimalCubeAFAMethod":
        from afabench.common.afa_methods import OptimalCubeAFAMethod  # noqa: PLC0415

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
        from afabench.common.datasets import CubeDataset  # noqa: PLC0415

        return CubeDataset
    if name == "cubeSimple":
        from afabench.common.datasets import CubeSimpleDataset  # noqa: PLC0415

        return CubeSimpleDataset
    if name == "cubeOnlyInformative":
        from afabench.common.datasets import CubeOnlyInformativeDataset  # noqa: PLC0415

        return CubeOnlyInformativeDataset
    if name == "shim2018cube":
        from common.datasets import Shim2018CubeDataset  # noqa: PLC0415

        return Shim2018CubeDataset
    if name == "AFAContext":
        from afabench.common.datasets import AFAContextDataset  # noqa: PLC0415

        return AFAContextDataset
    if name == "AFAContextRandomInsert":
        from afabench.common.datasets import (  # noqa: PLC0415
            AFAContextRandomInsertDataset,
        )

        return AFAContextRandomInsertDataset
    if name == "ContextSelectiveXOR":
        from afabench.common.datasets import ContextSelectiveXORDataset  # noqa: PLC0415

        return ContextSelectiveXORDataset
    if name == "MNIST":
        from afabench.common.datasets import MNISTDataset  # noqa: PLC0415

        return MNISTDataset
    if name == "diabetes":
        from afabench.common.datasets import DiabetesDataset  # noqa: PLC0415

        return DiabetesDataset
    if name == "physionet":
        from afabench.common.datasets import PhysionetDataset  # noqa: PLC0415

        return PhysionetDataset
    if name == "miniboone":
        from afabench.common.datasets import MiniBooNEDataset  # noqa: PLC0415

        return MiniBooNEDataset
    if name == "FashionMNIST":
        from afabench.common.datasets import FashionMNISTDataset  # noqa: PLC0415

        return FashionMNISTDataset
    if name == "bank_marketing":
        from afabench.common.datasets import BankMarketingDataset

        return BankMarketingDataset

    if name == "ckd":
        from afabench.common.datasets import CKDDataset

        return CKDDataset

    if name == "actg":
        from afabench.common.datasets import ACTG175Dataset

        return ACTG175Dataset

    if name == "imagenette":
        from afabench.common.datasets import ImagenetteDataset

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


def get_afa_classifier_class(name: str) -> type[AFAClassifier]:
    if name == "randomdummy":
        from afabench.common.classifiers import (  # noqa: PLC0415
            RandomDummyAFAClassifier,
        )

        return RandomDummyAFAClassifier
    if name == "uniformdummy":
        from afabench.common.classifiers import (  # noqa: PLC0415
            UniformDummyAFAClassifier,
        )

        return UniformDummyAFAClassifier
    if name == "WrappedMaskedMLPClassifier":
        from afabench.common.classifiers import (  # noqa: PLC0415
            WrappedMaskedMLPClassifier,
        )

        return WrappedMaskedMLPClassifier

    if name == "WrappedMaskedViTClassifier":
        from afabench.common.classifiers import (
            WrappedMaskedViTClassifier,
        )

        return WrappedMaskedViTClassifier
    if name == "Shim2018AFAClassifier":
        from afabench.afa_rl.shim2018.models import (  # noqa: PLC0415
            Shim2018AFAClassifier,
        )

        return Shim2018AFAClassifier
    if name == "Zannone2019AFAClassifier":
        from afabench.afa_rl.zannone2019.models import (  # noqa: PLC0415
            Zannone2019AFAClassifier,
        )

        return Zannone2019AFAClassifier
    if name == "Kachuee2019AFAClassifier":
        from afabench.afa_rl.kachuee2019.models import (  # noqa: PLC0415
            Kachuee2019AFAClassifier,
        )

        return Kachuee2019AFAClassifier
    msg = f"Unknown AFA classifier: {name}"
    raise ValueError(msg)
