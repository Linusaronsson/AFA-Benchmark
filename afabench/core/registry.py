# from afabench.core.types import (
#     AFAClassifier,
#     AFADataset,
#     AFAInitializer,
#     AFAMethod,
#     AFAUnmasker,
# )

REGISTERED_CLASSES = {
    # AFA Method Classes
    "RLAFAMethod": "afabench.components.methods.rl.common.afa_methods.RLAFAMethod",
    "GDFSAFAMethod": "afabench.components.methods.discriminative.gdfs.afa_methods.GDFSAFAMethod",
    "DIMEAFAMethod": "afabench.components.methods.discriminative.dime.afa_methods.DIMEAFAMethod",
    "EDDIAFAMethod": "afabench.components.methods.generative.eddi.afa_methods.EDDIAFAMethod",
    "AACOAFAMethod": "afabench.components.methods.oracle.aaco.afa_methods.AACOAFAMethod",
    "AACONNAFAMethod": "afabench.components.methods.oracle.aaco.nn.AACONNAFAMethod",
    "StaticBaseMethod": "afabench.components.methods.static.pt.StaticBaseMethod",
    "RandomWithoutClassifierAFAMethod": "afabench.components.methods.dummy.without_classifier.RandomWithoutClassifierAFAMethod",
    "SequentialWithoutClassifierAFAMethod": "afabench.components.methods.dummy.without_classifier.SequentialWithoutClassifierAFAMethod",
    "RandomWithClassifierAFAMethod": "afabench.components.methods.dummy.with_classifier.RandomWithClassifierAFAMethod",
    "SequentialWithClassifierAFAMethod": "afabench.components.methods.dummy.with_classifier.SequentialWithClassifierAFAMethod",
    # Legacy aliases for backward compatibility with saved bundles
    "RandomDummyAFAMethod": "afabench.components.methods.dummy.without_classifier.RandomWithoutClassifierAFAMethod",
    "SequentialDummyAFAMethod": "afabench.components.methods.dummy.without_classifier.SequentialWithoutClassifierAFAMethod",
    "RandomSelectionAFAMethod": "afabench.components.methods.dummy.with_classifier.RandomWithClassifierAFAMethod",
    "SequentialSelectionAFAMethod": "afabench.components.methods.dummy.with_classifier.SequentialWithClassifierAFAMethod",
    # AFA Dataset Classes
    "CubeDataset": "afabench.datasets.datasets.CubeDataset",
    "CubeNonUniformCostsDataset": "afabench.datasets.datasets.CubeNonUniformCostsDataset",
    "CubeNMDataset": "afabench.datasets.datasets.CubeNMDataset",
    "AFAContextDataset": "afabench.datasets.datasets.CubeNMDataset",  # legacy alias to CubeNM
    "MNISTDataset": "afabench.datasets.datasets.MNISTDataset",
    "SyntheticMNISTDataset": "afabench.datasets.datasets.SyntheticMNISTDataset",
    "DiabetesDataset": "afabench.datasets.datasets.DiabetesDataset",
    "PhysionetDataset": "afabench.datasets.datasets.PhysionetDataset",
    "MiniBooNEDataset": "afabench.datasets.datasets.MiniBooNEDataset",
    "FashionMNISTDataset": "afabench.datasets.datasets.FashionMNISTDataset",
    "BankMarketingDataset": "afabench.datasets.datasets.BankMarketingDataset",
    "CKDDataset": "afabench.datasets.datasets.CKDDataset",
    "HeartDiseaseDataset": "afabench.datasets.datasets.HeartDiseaseDataset",
    "ACTG175Dataset": "afabench.datasets.datasets.ACTG175Dataset",
    "NHANESMortalityDataset": "afabench.datasets.datasets.NHANESMortalityDataset",
    "ImagenetteDataset": "afabench.datasets.datasets.ImagenetteDataset",
    "TrainingDatasetView": "afabench.datasets.training_views.TrainingDatasetView",
    # AFA Classifier Classes
    "RandomDummyAFAClassifier": "afabench.components.classifiers.RandomDummyAFAClassifier",
    "UniformDummyAFAClassifier": "afabench.components.classifiers.UniformDummyAFAClassifier",
    "WrappedMaskedMLPClassifier": "afabench.components.classifiers.WrappedMaskedMLPClassifier",
    "WrappedMaskedViTClassifier": "afabench.components.classifiers.WrappedMaskedViTClassifier",
    "JAFAAFAClassifier": "afabench.components.methods.rl.jafa.models.JAFAAFAClassifier",
    "ODINAFAClassifier": "afabench.components.methods.rl.odin.models.ODINAFAClassifier",
    "OLAFAClassifier": "afabench.components.methods.rl.ol.models.OLAFAClassifier",
    "GreedyAFAClassifier": "afabench.components.methods.discriminative.common.models.GreedyAFAClassifier",
    # AFA Unmasker Classes
    "DirectUnmasker": "afabench.components.unmaskers.DirectUnmasker",
    "ImagePatchUnmasker": "afabench.components.unmaskers.ImagePatchUnmasker",
    "CubeNMUnmasker": "afabench.components.unmaskers.CubeNMUnmasker",
    "GroupedFeatureUnmasker": "afabench.components.unmaskers.GroupedFeatureUnmasker",
    # AFA Initializer Classes
    "ZeroInitializer": "afabench.components.initializers.ZeroInitializer",
    "FixedRandomInitializer": "afabench.components.initializers.FixedRandomInitializer",
    "ManualInitializer": "afabench.components.initializers.ManualInitializer",
    "MutualInformationInitializer": "afabench.components.initializers.MutualInformationInitializer",
    "LeastInformativeInitializer": "afabench.components.initializers.LeastInformativeInitializer",
    "RandomInitializer": "afabench.components.initializers.RandomInitializer",
    # General PyTorch Model Bundle
    "TorchModelBundle": "afabench.core.bundle_system.torch_bundle.TorchModelBundle",
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
