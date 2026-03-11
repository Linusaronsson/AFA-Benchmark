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
    "StopBaselineMethod": "afabench.common.stop_baseline_method.StopBaselineMethod",
    "CubeNMAROracleMethod": "afabench.common.cube_nm_ar_oracle_method.CubeNMAROracleMethod",
    # AFA Dataset Classes
    "CubeDataset": "afabench.common.datasets.datasets.CubeDataset",
    "CubeNonUniformCostsDataset": "afabench.common.datasets.datasets.CubeNonUniformCostsDataset",
    "AFAContextDataset": "afabench.common.datasets.datasets.AFAContextDataset",
    "CubeNMARDataset": "afabench.common.datasets.datasets.CubeNMARDataset",
    "MNISTDataset": "afabench.common.datasets.datasets.MNISTDataset",
    "SyntheticMNISTDataset": "afabench.common.datasets.datasets.SyntheticMNISTDataset",
    "DiabetesDataset": "afabench.common.datasets.datasets.DiabetesDataset",
    "PhysionetDataset": "afabench.common.datasets.datasets.PhysionetDataset",
    "MiniBooNEDataset": "afabench.common.datasets.datasets.MiniBooNEDataset",
    "FashionMNISTDataset": "afabench.common.datasets.datasets.FashionMNISTDataset",
    "BankMarketingDataset": "afabench.common.datasets.datasets.BankMarketingDataset",
    "CKDDataset": "afabench.common.datasets.datasets.CKDDataset",
    "ACTG175Dataset": "afabench.common.datasets.datasets.ACTG175Dataset",
    "FICODataset": "afabench.common.datasets.datasets.FICODataset",
    "PharyngitisDataset": "afabench.common.datasets.datasets.PharyngitisDataset",
    "ImagenetteDataset": "afabench.common.datasets.datasets.ImagenetteDataset",
    # AFA Classifier Classes
    "RandomDummyAFAClassifier": "afabench.common.classifiers.RandomDummyAFAClassifier",
    "UniformDummyAFAClassifier": "afabench.common.classifiers.UniformDummyAFAClassifier",
    "WrappedMaskedMLPClassifier": "afabench.common.classifiers.WrappedMaskedMLPClassifier",
    "WrappedMaskedViTClassifier": "afabench.common.classifiers.WrappedMaskedViTClassifier",
    "WrappedMALearnClassifier": "afabench.common.classifiers.WrappedMALearnClassifier",
    "Shim2018AFAClassifier": "afabench.afa_rl.shim2018.models.Shim2018AFAClassifier",
    "Zannone2019AFAClassifier": "afabench.afa_rl.zannone2019.models.Zannone2019AFAClassifier",
    "Kachuee2019AFAClassifier": "afabench.afa_rl.kachuee2019.models.Kachuee2019AFAClassifier",
    "GreedyAFAClassifier": "afabench.afa_discriminative.models.GreedyAFAClassifier",
    # AFA Unmasker Classes
    "DirectUnmasker": "afabench.common.unmaskers.DirectUnmasker",
    "ImagePatchUnmasker": "afabench.common.unmaskers.ImagePatchUnmasker",
    "AFAContextUnmasker": "afabench.common.unmaskers.AFAContextUnmasker",
    "CubeNMARUnmasker": "afabench.common.unmaskers.CubeNMARUnmasker",
    # AFA Initializer Classes
    "ZeroInitializer": "afabench.common.initializers.ZeroInitializer",
    "FixedRandomInitializer": "afabench.common.initializers.FixedRandomInitializer",
    "DynamicRandomInitializer": "afabench.common.initializers.DynamicRandomInitializer",
    "ManualInitializer": "afabench.common.initializers.ManualInitializer",
    "MutualInformationInitializer": "afabench.common.initializers.MutualInformationInitializer",
    "LeastInformativeInitializer": "afabench.common.initializers.LeastInformativeInitializer",
    "RandomInitializer": "afabench.common.initializers.RandomInitializer",
    "MissingnessInitializer": "afabench.common.initializers.MissingnessInitializer",
    "CubeNMARInitializer": "afabench.common.initializers.CubeNMARInitializer",
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


# Backward compatibility functions - these delegate to the unified get_class function
# def get_afa_method_class(class_name: str) -> type[AFAMethod]:
#     """Return a reference to an AFAMethod class, given the class name."""
#     return get_class(class_name)


# def get_afa_dataset_class(class_name: str) -> type[AFADataset]:
#     """Return a reference to an AFADataset class, given the class name."""
#     return get_class(class_name)


# def get_afa_classifier_class(class_name: str) -> type[AFAClassifier]:
#     """Return a reference to an AFAClassifier class, given the class name."""
#     return get_class(class_name)


# def get_afa_unmasker_class(class_name: str) -> type[AFAUnmasker]:
#     """Return a reference to an AFAUnmasker class, given the class name."""
#     return get_class(class_name)


# def get_afa_initializer_class(class_name: str) -> type[AFAInitializer]:
#     """Return a reference to an AFAInitializer class, given the class name."""
#     return get_class(class_name)
