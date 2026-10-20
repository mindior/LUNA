# tests/test_explainers_tabular.py
import numpy as np
import pandas as pd
import pytest

from luna_xai.methods.dsa_method import DSAMethod
from luna_xai.metrics.ch_metric import CHMetric
from luna_xai.renderers.plot_renderer import PlotRenderer
from luna_xai.core.plot_exporter import PlotExportStrategy
from luna_xai.explainers.tabular_regression import TabularRegressionExplainer
from luna_xai.explainers.tabular_classification import TabularClassificationExplainer
from luna_xai.utils.normalization import arctan_normalize
from luna_xai.core.results import MetricResult


class DummyExporter(PlotExportStrategy):
    def __init__(self):
        self.calls = []

    def save(self, fig, path: str) -> None:
        self.calls.append(path)


def test_tabular_regression_explainer_end_to_end(
    dummy_regression_domain,
    toy_regression_data,
    tmp_path,
):
    X_df, _ = toy_regression_data
    domain = dummy_regression_domain
    method = DSAMethod(n_neighbors=5, n_samples=50, seed=42)
    method.domain = domain
    method.neighbor_finder = domain.get_neighbor_finder()

    metric = CHMetric()
    exporter = DummyExporter()
    renderer = PlotRenderer(
        plot_type="importance",
        output_dir=str(tmp_path),
        exporter=exporter,
        features=None,
        top_k_importances=3,
        debug=False,
    )

    explainer = TabularRegressionExplainer(
        model=domain.model,
        domain=domain,
        method=method,
        metric=metric,
        renderer=renderer,
        output_dir=str(tmp_path),
        normalizer=arctan_normalize,
    )

    instance = X_df.iloc[0]
    result = explainer.explain_instance(instance, idx="0", seed=123)

    assert isinstance(result, MetricResult)
    assert result.importances
    # After arctan normalization, everything should fall within [-1, 1]
    for v in result.importances.values():
        assert -1.0 <= v <= 1.0


def test_tabular_classification_explainer_end_to_end(
    dummy_classification_domain,
    toy_classification_data,
    tmp_path,
):
    X_df, _ = toy_classification_data
    domain = dummy_classification_domain

    method = DSAMethod(n_neighbors=5, n_samples=50, seed=42)
    method.domain = domain
    method.neighbor_finder = domain.get_neighbor_finder()

    metric = CHMetric()
    exporter = DummyExporter()
    renderer = PlotRenderer(
        plot_type="importance",
        output_dir=str(tmp_path),
        exporter=exporter,
        features=None,
        top_k_importances=3,
        debug=False,
    )

    explainer = TabularClassificationExplainer(
        model=domain.model,
        domain=domain,
        method=method,
        metric=metric,
        renderer=renderer,
        output_dir=str(tmp_path),
        normalizer=arctan_normalize,
    )

    instance = X_df.iloc[0]
    result = explainer.explain_instance(instance, idx="0", seed=123)

    assert isinstance(result, MetricResult)
    assert result.importances
    for v in result.importances.values():
        assert -1.0 <= v <= 1.0
