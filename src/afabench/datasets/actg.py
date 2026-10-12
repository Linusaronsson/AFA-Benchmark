from typing import final, override

import pandas as pd
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class ACTG175Dataset(UCIDataset):
    @override
    def _fetch_uci(self) -> pd.DataFrame:
        actg_data = fetch_ucirepo(id=890)
        assert actg_data is not None
        assert actg_data.data is not None
        features_df = actg_data.data.features.copy()
        target_df = actg_data.data.targets.copy()
        target_series = target_df.iloc[:, 0].astype(int)
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        return df_data
