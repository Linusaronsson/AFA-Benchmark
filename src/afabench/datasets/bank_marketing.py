from typing import cast, final, override

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class BankMarketingDataset(UCIDataset):
    @override
    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        df_data = pd.read_csv(self.source, sep=";")
        target_col = "y" if "y" in df_data.columns else "deposit"
        features_df = df_data.drop(columns=[target_col])
        pd.set_option("future.no_silent_downcasting", True)
        labels = cast(
            "pd.Series",
            df_data[target_col].replace({"yes": 1, "no": 0}).astype("int64"),
        )
        # Missing categorical values are a category of their own
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                features_df[col] = LabelEncoder().fit_transform(
                    features_df[col].astype(str)
                )
        return features_df, labels

    @override
    def _fetch_source(self) -> None:
        self.source.parent.mkdir(parents=True, exist_ok=True)
        self._fetch_uci().to_csv(self.source, sep=";", index=False)

    @override
    def _fetch_uci(self) -> pd.DataFrame:
        bank_data = fetch_ucirepo(id=222)
        assert bank_data is not None
        assert bank_data.data is not None
        return pd.concat(
            [bank_data.data.features, bank_data.data.targets],
            axis=1,
        )
