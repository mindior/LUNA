# tests/test_tabular_domain_manager_parquet.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from luna_xai.domains.tabular_domain_manager import TabularDomainManager
from luna_xai.domains.tabular_domain import TabularDomain

import pytest

pytestmark = pytest.mark.integration

def _make_tiny_dataset():
    """Create a small numeric dataset for fast tests."""
    df = pd.DataFrame(
        {
            "f0": [1.0, 2.0, 3.0],
            "f1": [10.0, 20.0, 30.0],
        }
    )
    y = [0.0, 1.0, 2.0]
    return df, y


def test_tabular_domain_manager_save_and_load_parquet_ok(tmp_path: Path):
    """End-to-end check that create_and_save + load work with Parquet + hash."""
    # --- Data and simple model ---
    X_train, y = _make_tiny_dataset()

    model = LinearRegression()
    model.fit(X_train, y)

    # Minimal data pipeline (just to satisfy the interface)
    pipeline_data = StandardScaler()
    pipeline_data.fit(X_train)
    pipeline_label = None

    save_dir = tmp_path / "saved_domain"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: create_and_save (should write parquet + hash) ---
    TabularDomainManager.create_and_save(
        model=model,
        pipeline_data=pipeline_data,
        pipeline_label=pipeline_label,
        X_train=X_train,
        categorical_features=[],  # no categoricals to avoid CANE in this test
        save_dir=save_dir,
        use_cane=False,
        space="l2",
    )

    # Parquet file must exist
    parquet_path = save_dir / "X_train.parquet"
    assert parquet_path.exists(), "X_train.parquet was not created by create_and_save."

    # Hash must be present in domain.json
    meta_path = save_dir / "domain.json"
    assert meta_path.exists(), "domain.json was not created."

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "train_parquet_path" in meta
    assert "train_hash" in meta
    assert isinstance(meta["train_hash"], str)
    assert meta["train_hash"], "train_hash should be a non-empty string."

    # --- Stage 2: load (must use saved parquet and verify hash) ---
    domain = TabularDomainManager.load(
        save_dir=save_dir,
        model=model,
        pipeline_data=pipeline_data,
        pipeline_label=pipeline_label,
        categorical_features=[],
        output_dir=tmp_path / "results",
        debug=False,
        strict_hash_check=True,
    )

    assert isinstance(domain, TabularDomain)

    # data_original must match original X_train (ignoring index)
    pd.testing.assert_frame_equal(
        domain.data_original.reset_index(drop=True),
        X_train.reset_index(drop=True),
    )


def test_tabular_domain_manager_load_parquet_hash_mismatch_raises(tmp_path: Path):
    """If the parquet is tampered with, strict_hash_check=True must raise an error."""
    # --- Data and simple model ---
    X_train, y = _make_tiny_dataset()

    model = LinearRegression()
    model.fit(X_train, y)

    pipeline_data = StandardScaler()
    pipeline_data.fit(X_train)
    pipeline_label = None

    save_dir = tmp_path / "saved_domain"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: create_and_save ---
    TabularDomainManager.create_and_save(
        model=model,
        pipeline_data=pipeline_data,
        pipeline_label=pipeline_label,
        X_train=X_train,
        categorical_features=[],
        save_dir=save_dir,
        use_cane=False,
        space="l2",
    )

    parquet_path = save_dir / "X_train.parquet"
    assert parquet_path.exists()

    # --- Corrupt the parquet after it was saved (to break the hash) ---
    df_corrupt = pd.read_parquet(parquet_path)
    df_corrupt.iloc[0, 0] += 9999  # change a single value
    df_corrupt.to_parquet(parquet_path, index=False)

    # --- load with strict_hash_check=True must fail ---
    with pytest.raises(ValueError) as excinfo:
        TabularDomainManager.load(
            save_dir=save_dir,
            model=model,
            pipeline_data=pipeline_data,
            pipeline_label=pipeline_label,
            categorical_features=[],
            output_dir=tmp_path / "results_bad",
            debug=False,
            strict_hash_check=True,
        )

    msg = str(excinfo.value).lower()
    assert "hash" in msg or "integrity" in msg

    # --- With strict_hash_check=False, load must succeed even with mismatched hash ---
    domain = TabularDomainManager.load(
        save_dir=save_dir,
        model=model,
        pipeline_data=pipeline_data,
        pipeline_label=pipeline_label,
        categorical_features=[],
        output_dir=tmp_path / "results_ok",
        debug=False,
        strict_hash_check=False,
    )

    assert isinstance(domain, TabularDomain)
