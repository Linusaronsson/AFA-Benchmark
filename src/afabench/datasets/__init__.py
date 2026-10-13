from .actg import ACTG175Dataset
from .aliases import DATASET_KEY_ALIASES
from .bank_marketing import BankMarketingDataset
from .ckd import CKDDataset
from .cube import CubeDataset
from .cube_nm import CubeNMDataset
from .cube_nonuniform_costs import CubeNonUniformCostsDataset
from .diabetes import DiabetesDataset
from .fashion_mnist import FashionMNISTDataset
from .imagenette import ImagenetteDataset
from .metabric import MetabricDataset
from .miniboone import MiniBooNEDataset
from .miniboone_subset import MiniBooNESubsetDataset
from .mnist import MNISTDataset
from .physionet_challenge_2012 import PhysionetChallenge2012Dataset
from .synthetic_mnist import SyntheticMNISTDataset

__all__ = [
    "DATASET_KEY_ALIASES",
    "ACTG175Dataset",
    "BankMarketingDataset",
    "CKDDataset",
    "CubeDataset",
    "CubeNMDataset",
    "CubeNonUniformCostsDataset",
    "DiabetesDataset",
    "FashionMNISTDataset",
    "ImagenetteDataset",
    "MNISTDataset",
    "MetabricDataset",
    "MiniBooNEDataset",
    "MiniBooNESubsetDataset",
    "PhysionetChallenge2012Dataset",
    "SyntheticMNISTDataset",
]
