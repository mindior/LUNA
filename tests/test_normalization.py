# tests/test_normalization.py
import math

import pytest

from luna_xai.utils.normalization import arctan_normalize


def test_arctan_normalize_range_and_sign():
    raw = {"a": 0.0, "b": 100.0, "c": -100.0}
    norm = arctan_normalize(raw, scale_factor=10.0)

    assert set(norm.keys()) == set(raw.keys())
    for v in norm.values():
        assert -1.0 <= v <= 1.0

    assert norm["b"] > 0
    assert norm["c"] < 0
    assert math.isclose(norm["a"], 0.0, abs_tol=1e-9)


def test_arctan_normalize_invalid_importances_type():
    with pytest.raises(TypeError):
        arctan_normalize(["not", "a", "dict"])  # type: ignore[arg-type]


def test_arctan_normalize_invalid_scale_factor():
    with pytest.raises(ValueError):
        arctan_normalize({"a": 1.0}, scale_factor=0.0)
