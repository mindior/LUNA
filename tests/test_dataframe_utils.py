# tests/test_dataframe_utils.py
import pandas as pd
import pytest

from luna_xai.utils.dataframe import ensure_dataframe


def test_ensure_dataframe_from_series():
    s = pd.Series({"a": 1, "b": 2})
    df = ensure_dataframe(s)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["a", "b"]


def test_ensure_dataframe_single_row_ok():
    df_in = pd.DataFrame({"a": [1], "b": [2]})
    df = ensure_dataframe(df_in)
    assert df is df_in  # should return the object itself without copying it
    assert df.shape == (1, 2)


def test_ensure_dataframe_multirow_raises():
    df_in = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        ensure_dataframe(df_in)


def test_ensure_dataframe_wrong_type_raises():
    with pytest.raises(TypeError):
        ensure_dataframe({"a": 1})  # type: ignore[arg-type]
