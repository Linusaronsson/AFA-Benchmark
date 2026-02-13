# from afabench.common.custom_types import (
#     AFAClassifier,
#     AFADataset,
#     AFAInitializer,
#     AFAMethod,
#     AFAUnmasker,
# )

REGISTERED_CLASSES = {
    # AFA Method Classes
    "RLAFAMethod": "afabench.afa_rl.common.afa_methods.RLAFAMethod",
    "Covert2023AFAMethod": "afabench.afa_discriminative.afa_methods.Covert2023AFAMethod",
    "Gadgil2023AFAMethod": "afabench.afa_discriminative.afa_methods.Gadgil2023AFAMethod",
    "Ma2018AFAMethod": "afabench.afa_generative.afa_methods.Ma2018AFAMethod",
    "AACOAFAMethod": "afabench.afa_oracle.afa_methods.AACOAFAMethod",
    "AACONNAFAMethod": "afabench.afa_oracle.aaco_nn.AACONNAFAMethod",
    "StaticBaseMethod": "afabench.static.static_methods.StaticBaseMethod",
    "SequentialDummyAFAMethod": "afabench.common.afa_methods.SequentialDummyAFAMethod",
    "RandomDummyAFAMethod": "afabench.common.afa_methods.RandomDummyAFAMethod",
    "OptimalCubeAFAMethod": "afabench.common.afa_methods.OptimalCubeAFAMethod",
    # AFA Dataset Classes
    "CubeDataset": "afabench.common.datasets.datasets.CubeDataset",
    "CubeNonUniformCostsDataset": "afabench.common.datasets.datasets.CubeNonUniformCostsDataset",
    "AFAContextDataset": "afabench.common.datasets.datasets.AFAContextDataset",
    "MNISTDataset": "afabench.common.datasets.datasets.MNISTDataset",
    "SyntheticMNISTDataset": "afabench.common.datasets.datasets.SyntheticMNISTDataset",
    "DiabetesDataset": "afabench.common.datasets.datasets.DiabetesDataset",
    "PhysionetDataset": "afabench.common.datasets.datasets.PhysionetDataset",
    "MiniBooNEDataset": "afabench.common.datasets.datasets.MiniBooNEDataset",
    "FashionMNISTDataset": "afabench.common.datasets.datasets.FashionMNISTDataset",
    "BankMarketingDataset": "afabench.common.datasets.datasets.BankMarketingDataset",
    "CKDDataset": "afabench.common.datasets.datasets.CKDDataset",
    "ACTG175Dataset": "afabench.common.datasets.datasets.ACTG175Dataset",
    "ImagenetteDataset": "afabench.common.datasets.datasets.ImagenetteDataset",
    "DummyDataset": "afabench.common.datasets.datasets.DummyDataset",
    # AFA Classifier Classes
    "RandomDummyAFAClassifier": "afabench.common.classifiers.RandomDummyAFAClassifier",
    "UniformDummyAFAClassifier": "afabench.common.classifiers.UniformDummyAFAClassifier",
    "WrappedMaskedMLPClassifier": "afabench.common.classifiers.WrappedMaskedMLPClassifier",
    "WrappedMaskedViTClassifier": "afabench.common.classifiers.WrappedMaskedViTClassifier",
    "Shim2018AFAClassifier": "afabench.afa_rl.shim2018.models.Shim2018AFAClassifier",
    "Zannone2019AFAClassifier": "afabench.afa_rl.zannone2019.models.Zannone2019AFAClassifier",
    "Kachuee2019AFAClassifier": "afabench.afa_rl.kachuee2019.models.Kachuee2019AFAClassifier",
    "GreedyAFAClassifier": "afabench.afa_discriminative.models.GreedyAFAClassifier",
    # AFA Unmasker Classes
    "DirectUnmasker": "afabench.common.unmaskers.DirectUnmasker",
    "ImagePatchUnmasker": "afabench.common.unmaskers.ImagePatchUnmasker",
    "AFAContextUnmasker": "afabench.common.unmaskers.AFAContextUnmasker",
    # AFA Initializer Classes
    "ZeroInitializer": "afabench.common.initializers.ZeroInitializer",
    "FixedRandomInitializer": "afabench.common.initializers.FixedRandomInitializer",
    "DynamicRandomInitializer": "afabench.common.initializers.DynamicRandomInitializer",
    "ManualInitializer": "afabench.common.initializers.ManualInitializer",
    "MutualInformationInitializer": "afabench.common.initializers.MutualInformationInitializer",
    "LeastInformativeInitializer": "afabench.common.initializers.LeastInformativeInitializer",
    "RandomInitializer": "afabench.common.initializers.RandomInitializer",
    # General PyTorch Model Bundle
    "TorchModelBundle": "afabench.common.torch_bundle.TorchModelBundle",
}


def get_class(class_name: str) -> type:
    """Return a reference to a registered class, given the class name."""
    if class_name not in REGISTERED_CLASSES:
        msg = f"Unknown class: {class_name}"
        raise ValueError(msg)

    module_name, class_name_in_module = REGISTERED_CLASSES[class_name].rsplit(
        ".", 1
    )
    module = __import__(module_name, fromlist=[class_name_in_module])
    return getattr(module, class_name_in_module)
