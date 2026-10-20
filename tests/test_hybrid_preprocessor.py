# tests/test_hybrid_preprocessor.py
import numpy as np
import pandas as pd
import pytest

from luna_xai.utils.preprocessing import HybridPreprocessor


def test_hybrid_preprocessor_fit_transform_numeric_and_categorical(tmp_path):
    df = pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "c"],
        }
    )

    pre = HybridPreprocessor(use_cane=False, verbose=False)
    transformed, idf_dict, scaler = pre.fit_transform(df)

    assert transformed.shape == df.shape
    assert isinstance(idf_dict, dict)
    assert pre.output_dim_ == df.shape[1]

    # Everything must be numerical
    assert np.issubdtype(transformed.values.dtype, np.number)

    # Persistence test
    path = tmp_path / "preprocessor.pkl"
    pre.save(path)
    loaded = HybridPreprocessor.load(path)

    assert loaded.is_fitted_
    assert loaded.feature_names_ == pre.feature_names_

    transformed2 = loaded.transform(df)
    assert transformed2.shape == transformed.shape
