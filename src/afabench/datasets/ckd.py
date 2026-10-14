from typing import ClassVar, final, override

import pandas as pd
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class CKDDataset(UCIDataset):
    """Chronic Kidney Disease (UCI 336), labelled 1 for chronic kidney disease."""

    SHA256: ClassVar[str] = (
        "45fb1b3b76b1a1a0f3bf1eab70796c76137ecc922e0f7c4c338141ae6e258f5b"
    )

    @classmethod
    @override
    def _fetch_uci(cls) -> pd.DataFrame:
        ckd_data = fetch_ucirepo(id=336)
        assert ckd_data is not None
        assert ckd_data.data is not None
        features_df = ckd_data.data.features.copy()
        target_df = ckd_data.data.targets.copy()
        target_series = (
            target_df.iloc[:, 0].astype(str).str.strip().str.lower()
        )
        target_series = target_series.map({"ckd": 1, "notckd": 0})
        df_data = features_df.copy()
        df_data["target"] = target_series.to_numpy()
        return df_data
