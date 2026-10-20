# tests/test_metrics_ch.py
import pytest
import numpy as np

from luna_xai.utils.metrics import calculate_ch_index, adjust_signal
from luna_xai.metrics.ch_metric import CHMetric


def _make_sensitivity_dict():
    # feature A with two clearly separated levels
    # feature B with mixed levels (lower CH)
    return {
        "A": {
            0.0: [0.1, 0.2, 0.15],
            1.0: [0.9, 1.0, 1.1],
        },
        "B": {
            0.0: [0.4, 0.5, 0.6],
            1.0: [0.45, 0.55, 0.65],
        },
    }


def test_calculate_ch_index_basic_order_and_nonzero():
    sens = _make_sensitivity_dict()
    ch, means = calculate_ch_index(sens)

    assert set(ch.keys()) == {"A", "B"}
    assert set(means.keys()) == {"A", "B"}

    # feature A must have a higher CH than B
    assert ch["A"] > ch["B"]
    assert ch["A"] > 0.0


def test_adjust_signal_respects_trend():
    # A crescendo -> positive, B decrescendo -> negative
    means = {
        "A": [0.1, 0.2, 0.3],
        "B": [1.0, 0.5, 0.2],
    }
    ch_values = {
        "A": 2.0,
        "B": 3.0,
    }
    adjusted = adjust_signal(means, ch_values)
    assert adjusted["A"] == pytest.approx(2.0)
    assert adjusted["B"] == pytest.approx(-3.0)


def test_chmetric_compute_wraps_metric_result():
    sens = _make_sensitivity_dict()
    metric = CHMetric()
    result = metric.compute(sens)

    assert isinstance(result.importances, dict)
    assert "A" in result.importances
    # should carry raw sensitivity along with it
    assert result.sensitivity_data is sens
    assert result.metadata["metric"] == "CH"
