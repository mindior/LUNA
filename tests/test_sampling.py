# tests/test_sampling.py
import numpy as np
import pandas as pd
import pytest

from luna_xai.utils.sampling import (
    create_lhs_dataset_empirical,
    create_synthetic_samples,
)


def _make_mixed_neighbors():
    return pd.DataFrame(
        {
            "num1": [1.0, 2.0, 3.0, 4.0],
            "num2": [10.0, 20.0, 30.0, 40.0],
            "cat": ["a", "b", "a", "c"],
        }
    )


def test_create_lhs_dataset_empirical_basic_shape_and_types():
    neighbors = _make_mixed_neighbors()
    n_samples = 50

    synthetic = create_lhs_dataset_empirical(neighbors, n_samples=n_samples)
    assert isinstance(synthetic, pd.DataFrame)
    assert synthetic.shape == (n_samples, neighbors.shape[1])
    assert list(synthetic.columns) == list(neighbors.columns)


def test_create_lhs_dataset_empirical_respects_numeric_ranges():
    neighbors = _make_mixed_neighbors()
    synthetic = create_lhs_dataset_empirical(neighbors, n_samples=100)

    for col in ["num1", "num2"]:
        assert synthetic[col].min() >= neighbors[col].min() - 1e-6
        assert synthetic[col].max() <= neighbors[col].max() + 1e-6


def test_create_lhs_dataset_empirical_raises_on_empty():
    neighbors = pd.DataFrame(columns=["a", "b"])
    with pytest.raises(ValueError):
        create_lhs_dataset_empirical(neighbors, n_samples=10)


def test_create_synthetic_samples_lhs_dispatch():
    neighbors = _make_mixed_neighbors()
    synthetic = create_synthetic_samples(neighbors, n_samples=20, method="lhs", seed=42)
    assert synthetic.shape[0] == 20
    assert list(synthetic.columns) == list(neighbors.columns)


def test_create_synthetic_samples_invalid_method_raises():
    neighbors = _make_mixed_neighbors()
    with pytest.raises(ValueError):
        create_synthetic_samples(neighbors, n_samples=10, method="foo")  # type: ignore[arg-type]
