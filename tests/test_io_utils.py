# tests/test_io_utils.py
import pytest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from luna_xai.utils.io_utils import (
    ensure_directory,
    load_model,
    load_data,
    save_dataframe_to_csv,
    save_dataframe_to_excel,
    safe_format,
    auto_convert_numeric,
)


def test_ensure_directory_creates(tmp_path):
    d = tmp_path / "subdir"
    assert not d.exists()
    out = ensure_directory(d, verbose=False)
    assert out == d
    assert d.exists()
    assert d.is_dir()


def test_load_model_and_data(tmp_path):
    # simple model
    model_path = tmp_path / "model.joblib"
    model = {"a": 1}
    joblib.dump(model, model_path)

    loaded = load_model(model_path)
    assert loaded == model

    # CSV
    df = pd.DataFrame({"x": [1, 2], "y": ["3", "4"]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    loaded_df = load_data(csv_path)
    assert loaded_df.shape == df.shape


def test_save_dataframe_to_csv_and_excel(tmp_path):
    pytest.importorskip("openpyxl")

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    csv_path = tmp_path / "out" / "data.csv"
    xlsx_path = tmp_path / "out" / "data.xlsx"

    save_dataframe_to_csv(df, csv_path, verbose=False)
    save_dataframe_to_excel(df, xlsx_path, verbose=False)

    assert csv_path.exists()
    assert xlsx_path.exists()


def test_safe_format_numeric_and_string():
    assert safe_format(1.23456, precision=3) == "1.23"
    assert safe_format("abc") == "abc"


def test_auto_convert_numeric_threshold():
    df = pd.DataFrame({"a": ["1", "2", "x"], "b": ["foo", "bar", "baz"]})
    conv = auto_convert_numeric(df, threshold=0.5)
    assert conv["a"].dtype != object
    assert conv["b"].dtype == object

    with pytest.raises(ValueError):
        auto_convert_numeric(df, threshold=1.5)
