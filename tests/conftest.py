# tests/conftest.py
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Ensure that the “luna_xai” package (in src/) is importable in tests.
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]  # folder LUNA/
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------
# Existing fixtures and dummies
# ---------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification


class DummyDataInterface:
    """Minimal interface for explainers/tabular domain."""
    def transform_input(self, df: pd.DataFrame):
        # Here we assume that the data is already in the model space.
        return df.values


class DummyNeighborFinder:
    """Simple (Euclidean) nearest neighbor in raw tabular space."""
    def __init__(self, X: pd.DataFrame):
        self.X = X.reset_index(drop=True)

    def find_neighbors(self, instance: pd.DataFrame, n_neighbors: int = 7, k_factor: int = 3):
        if isinstance(instance, pd.Series):
            inst_df = instance.to_frame().T
        else:
            inst_df = instance

        inst_vec = inst_df.values.astype(float)
        base_vecs = self.X.values.astype(float)

        dists = np.linalg.norm(base_vecs - inst_vec, axis=1)
        # Ordena por distância e pega top-k (ignorando o próprio ponto se existir)
        order = np.argsort(dists)
        order = order[dists[order] > 0.0][:n_neighbors]

        neighbors = self.X.iloc[order].copy().reset_index(drop=True)
        neighbors["distance"] = dists[order]
        return neighbors


class DummyDomain:
    """Minimum domain for DSA testing + tabular explainers."""
    def __init__(self, model, X_train: pd.DataFrame, task: str = "regression"):
        self.model = model
        self.X_train = X_train.reset_index(drop=True)
        self.data_interface = DummyDataInterface()
        self.task = task
        self.neighbor_finder = DummyNeighborFinder(self.X_train)

    def get_neighbor_finder(self):
        return self.neighbor_finder

    def predict(self, df: pd.DataFrame):
        X = self.data_interface.transform_input(df)
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            # For DSA in classification, we will use class 1 probability
            proba = self.model.predict_proba(X)
            return proba[:, 1]
        return self.model.predict(X)


@pytest.fixture
def toy_regression_data():
    X, y = make_regression(
        n_samples=200,
        n_features=5,
        noise=0.1,
        random_state=42,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=cols)
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser


@pytest.fixture
def toy_regression_model(toy_regression_data):
    X_df, y_ser = toy_regression_data
    model = LinearRegression()
    model.fit(X_df, y_ser)
    return model


@pytest.fixture
def toy_classification_data():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    cols = [f"f{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=cols)
    y_ser = pd.Series(y, name="target")
    return X_df, y_ser


@pytest.fixture
def toy_classification_model(toy_classification_data):
    X_df, y_ser = toy_classification_data
    model = LogisticRegression(max_iter=1000)
    model.fit(X_df, y_ser)
    return model


@pytest.fixture
def dummy_regression_domain(toy_regression_model, toy_regression_data):
    X_df, _ = toy_regression_data
    return DummyDomain(model=toy_regression_model, X_train=X_df, task="regression")


@pytest.fixture
def dummy_classification_domain(toy_classification_model, toy_classification_data):
    X_df, _ = toy_classification_data
    return DummyDomain(model=toy_classification_model, X_train=X_df, task="classification")
