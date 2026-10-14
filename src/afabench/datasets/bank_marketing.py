from typing import ClassVar, cast, final, override

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

from afabench.datasets.uci import UCIDataset


@final
class BankMarketingDataset(UCIDataset):
    """Bank Marketing (UCI 222, `bank-full`), labelled 1 when the client subscribed a term deposit."""

    SHA256: ClassVar[str] = (
        "7e3c9e305e7871ef398254dd01c1e3a1ff2b66d6c3e36666367cf07fdb0f8f03"
    )

    @override
    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        df_data = pd.read_csv(self.source, sep=";")
        features_df = df_data.drop(columns=["y"])
        pd.set_option("future.no_silent_downcasting", True)
        labels = cast(
            "pd.Series",
            df_data["y"].replace({"yes": 1, "no": 0}).astype("int64"),
        )
        # Missing categorical values are a category of their own
        for col in features_df.columns:
            if features_df[col].dtype == "object":
                features_df[col] = LabelEncoder().fit_transform(
                    features_df[col].astype(str)
                )
        return features_df, labels

    @classmethod
    @override
    def make_prepared_source(cls) -> bytes:
        return cls._fetch_uci().to_csv(sep=";", index=False).encode()

    @classmethod
    @override
    def _fetch_uci(cls) -> pd.DataFrame:
        bank_data = fetch_ucirepo(id=222)
        assert bank_data is not None
        assert bank_data.data is not None
        return pd.concat(
            [bank_data.data.features, bank_data.data.targets],
            axis=1,
        )
