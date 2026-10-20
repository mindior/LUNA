# tests/test_plotting_renderer.py
import matplotlib
matplotlib.use("Agg")  # força backend sem GUI para os testes

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from luna_xai.utils.plotting import (
    plot_importances,
    plot_feature_trends,
    plot_feature_boxplots,
)
from luna_xai.renderers.plot_renderer import PlotRenderer
from luna_xai.core.plot_exporter import PlotExportStrategy
from luna_xai.core.results import MetricResult


class DummyExporter(PlotExportStrategy):
    """Exporter que não grava em disco, só registra as chamadas."""
    def __init__(self):
        self.saved = []

    def save(self, fig, path: str) -> None:
        self.saved.append(path)
        # We don't need to close fig here; pytest already runs fast and small.


def _dummy_sensitivity_data():
    # format: feature -> level -> [preds...]
    return {
        "f0": {0.0: [0.1, 0.2], 1.0: [0.8, 0.9]},
        "f1": {5.0: [1.0, 1.1], 10.0: [2.0, 2.1]},
    }


def test_plot_importances_returns_figure():
    from matplotlib.figure import Figure

    importances = {"a": 0.5, "b": -0.2, "c": 0.1}
    fig = plot_importances(
        idx="test",
        importances_dict=importances,
        type_="local",
        title="Test",
        top_k=2,
    )
    assert isinstance(fig, Figure)


def test_plot_feature_trends_and_boxplots_return_figs():
    sens = _dummy_sensitivity_data()
    instance = pd.Series({"f0": 0.0, "f1": 10.0})

    trend_figs = plot_feature_trends(
        idx="test",
        sensitivity_data=sens,
        instance=instance,
        features=None,
    )
    assert "f0" in trend_figs
    assert "f1" in trend_figs

    box_figs = plot_feature_boxplots(
        idx="test",
        predictions=sens,
        instance_prediction=1.0,
        instance=instance,
        features=None,
    )
    assert "f0" in box_figs
    assert "f1" in box_figs


def test_plot_renderer_importance_line_boxplot(tmp_path):
    sens = _dummy_sensitivity_data()
    importances = {"f0": 0.5, "f1": -0.3}
    result = MetricResult(
        importances=importances,
        sensitivity_data=sens,
        metadata={},
    )

    exporter = DummyExporter()
    renderer = PlotRenderer(
        plot_type="importance",
        output_dir=str(tmp_path),
        exporter=exporter,
        features=["f0"],
        top_k_importances=2,
        debug=False,
    )

    # importance
    renderer.render(result, context={"scope": "local", "idx": "42"})
    assert any("importances" in p for p in exporter.saved)

    # line
    renderer.plot_type = "line"
    renderer.render_from_data(
        sensitivity_data=sens,
        plot_type="line",
        instance=pd.Series({"f0": 0.0}),
        idx="42",
        instance_prediction=None,
    )
    assert any("lineplot" in p for p in exporter.saved)

    # boxplot
    renderer.plot_type = "boxplot"
    renderer.render_from_data(
        sensitivity_data=sens,
        plot_type="boxplot",
        instance=pd.Series({"f0": 0.0}),
        idx="42",
        instance_prediction=1.0,
    )
    assert any("boxplot" in p for p in exporter.saved)
