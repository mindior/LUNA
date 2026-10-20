from __future__ import annotations

from typing import Any
import pandas as pd


def ensure_dataframe(instance: Any) -> pd.DataFrame:
    """
    Normalize an input instance into a single-row pandas DataFrame.

    Valid inputs:
        - pd.Series      → converted to a single-row DataFrame
        - pd.DataFrame   → must contain exactly one row

    All other input types are rejected with a TypeError,
    matching the contract expected by the LUNA test suite.
    """
    # Case 1: DataFrame with exactly one row
    if isinstance(instance, pd.DataFrame):
        if len(instance) != 1:
            raise ValueError(
                "ensure_dataframe expects a DataFrame with exactly one row; "
                f"received {len(instance)} rows."
            )
        return instance

    # Case 2: Series → convert to single-row DataFrame
    if isinstance(instance, pd.Series):
        return instance.to_frame().T

    # Everything else is invalid (required by tests)
    raise TypeError(
        "ensure_dataframe expects a pandas DataFrame or Series. "
        f"Received: {type(instance).__name__}."
    )
