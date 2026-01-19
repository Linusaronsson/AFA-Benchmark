from collections.abc import Sequence

import pandas as pd


def assert_where_selections_there_cost(
    df: pd.DataFrame, selections: Sequence[int], cost: float
) -> None:
    """Assert that wherever "prev_selections_performed" in the dataframe is equal to `selections`, "accumulated_cost" has to be equal to `cost`."""
    raise NotImplementedError
