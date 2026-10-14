from typing import ClassVar, final, override

import pandas as pd
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class ACTG175Dataset(UCIDataset):
    """AIDS Clinical Trials Group Study 175 (UCI 890), labelled with the failure indicator `cid`."""

    SHA256: ClassVar[str] = (
        "71493efd4ac3d8edf9edb43df83f0cd91b4cfe73bc9f96278fa8b536e8131cda"
    )
    # UCI links to ClinicalTrials.gov for the license, which gives none
    SOURCE_COMMITTED: ClassVar[bool] = False

    @classmethod
    @override
    def _fetch_uci(cls) -> pd.DataFrame:
        actg_data = fetch_ucirepo(id=890)
        assert actg_data is not None
        assert actg_data.data is not None
        features_df = actg_data.data.features.copy()
        target_df = actg_data.data.targets.copy()
        target_series = target_df.iloc[:, 0].astype(int)
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        return df_data
